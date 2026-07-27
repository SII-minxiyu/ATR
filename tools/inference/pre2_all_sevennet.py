import os
import warnings
import sys
import time

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write as ase_write
from tqdm import tqdm

import torch

# SevenNet ASE calculator
try:
    # 新版本（GitHub / 0.11+）
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    # 老版本（0.10.x）
    from sevenn.sevennet_calculator import SevenNetCalculator


# ---------- 配置 ----------
# parquet 所在目录（你现在这些 train-xxxxx-of-00371.parquet 就在当前目录）
DATA_DIR = "."

# extxyz 输出目录（随便起个新名字，避免跟之前 CHGNet 结果混在一起）
OUTPUT_DIR = "./pbe_sevennet"

# 要处理的 shard 范围（含两端）
FIRST_IDX = 21
LAST_IDX = 22

# 只保留每条弛豫轨迹的最后一步？和你之前脚本含义一样
ONLY_FINAL_STEP = False

# 只读取推理需要的 parquet 列，减少 IO 和内存占用
PARQUET_COLUMNS = [
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "dimension_types",
    "id",
    "immutable_id",
]

# SevenNet 模型
# 1）如果你已经从作者那拿到 SevenNet-MF-0（r2SCAN 精度）的 checkpoint：
#    把下面改成实际路径，比如：
#    SEVENNET_MODEL = "/path/to/SevenNet-MF-0-checkpoint.pth"
#    SEVENNET_MODAL = None
#
# 2）如果暂时没有 MF-0，可以先用官方的 SevenNet-MF-ompa：
#    SEVENNET_MODEL = "7net-mf-ompa"
#    SEVENNET_MODAL = "mpa"
SEVENNET_MODEL = "/inspire/hdd/global_user/luomingxiang-240108540155/test1124/sevennet/checkpoint_sevennet_mf_0.pth" 
SEVENNET_FILE_TYPE = "checkpoint"
# 只有多模态模型才需要 modal，比如 "mpa" / "omat24" / "matpes_r2scan"。
# 单模态模型或自训练单任务 checkpoint 通常设成 None。
SEVENNET_MODAL = "R2SCAN"
# 官方提供的 ASE 推理加速后端。通常只开一个即可。
SEVENNET_ENABLE_CUEQ = False
SEVENNET_ENABLE_FLASH = False
SEVENNET_ENABLE_OEQ = False

# 记录失败结构的 log
SKIP_LOG = os.path.join(OUTPUT_DIR, "skipped_sevennet6.txt")
# -------------------------


def ensure_dir_for_file(path: str):
    """确保保存文件的目录存在。"""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _flatten_to_float_list(x):
    """把嵌套 list/tuple/np.ndarray 展平成 float list。"""
    out = []
    stack = [x]
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple, np.ndarray)):
            stack.extend(reversed(list(v)))
        else:
            out.append(float(v))
    out.reverse()
    return out


def parquet_columns_to_read():
    columns = list(PARQUET_COLUMNS)
    if ONLY_FINAL_STEP:
        columns.append("relaxation_step")
    return columns


def atoms_from_row(row):
    """从一行 parquet 构造 ASE Atoms + id 信息。"""
    # cell
    cell_vals = _flatten_to_float_list(row["lattice_vectors"])
    cell = np.array(cell_vals, dtype=float).reshape(3, 3)

    # positions（笛卡尔）
    pos_vals = _flatten_to_float_list(row["cartesian_site_positions"])
    pos = np.array(pos_vals, dtype=float).reshape(-1, 3)

    symbols = list(row["species_at_sites"])
    if len(symbols) != pos.shape[0]:
        raise ValueError(
            f"len(symbols)={len(symbols)} != positions={pos.shape[0]} for id={row.get('id')}"
        )

    dim_types = row.get("dimension_types")
    if dim_types is not None:
        pbc = [bool(d) for d in dim_types]
    else:
        pbc = [True, True, True]

    atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=pbc)

    # id：pbe -> r2scan（和你原来一致）
    raw_id = row.get("id")
    if isinstance(raw_id, str):
        sid = raw_id.replace("pbe", "r2scan")
    else:
        sid = raw_id
    immid = row.get("immutable_id")

    return atoms, sid, immid


def build_output_atoms(atoms: Atoms, sid, immid, energy, forces):
    atoms_out = atoms.copy()
    atoms_out.info = {
        "id": sid,
        "immutable_id": immid,
        "energy": energy,
    }
    atoms_out.set_array("forces", forces.astype(float))
    return atoms_out


def validate_accelerator_flags():
    enabled = [
        name
        for name, flag in (
            ("cueq", SEVENNET_ENABLE_CUEQ),
            ("flash", SEVENNET_ENABLE_FLASH),
            ("oeq", SEVENNET_ENABLE_OEQ),
        )
        if flag
    ]
    if len(enabled) > 1:
        raise ValueError(
            f"Only one SevenNet accelerator should be enabled at a time, got {enabled}"
        )


def build_sevennet_calculators():
    """建一个 GPU SevenNet 和一个 CPU SevenNet，方便 OOM 时降级。"""
    calc_kwargs = {
        "model": SEVENNET_MODEL,
        "file_type": SEVENNET_FILE_TYPE,
    }
    if SEVENNET_MODAL is not None:
        calc_kwargs["modal"] = SEVENNET_MODAL

    # GPU
    calc_gpu = None
    if torch.cuda.is_available():
        calc_gpu = SevenNetCalculator(
            device="cuda",
            enable_cueq=SEVENNET_ENABLE_CUEQ,
            enable_flash=SEVENNET_ENABLE_FLASH,
            enable_oeq=SEVENNET_ENABLE_OEQ,
            **calc_kwargs,
        )

    # CPU
    calc_cpu = SevenNetCalculator(device="cpu", **calc_kwargs)

    return calc_gpu, calc_cpu


def sevennet_predict(atoms: Atoms, calc_gpu, calc_cpu):
    """
    用 SevenNet 预测单个结构的能量和力。
    返回: (E_tot, forces)，单位 eV / eV/Å
    """
    # 没有 GPU 的情况：直接走 CPU
    if calc_gpu is None:
        atoms.calc = calc_cpu
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        return float(e), np.array(f, dtype=float), "cpu"

    # 有 GPU：先尝试 GPU，OOM 再降级到 CPU
    try:
        atoms.calc = calc_gpu
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        return float(e), np.array(f, dtype=float), "cuda"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        atoms.calc = calc_cpu
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        return float(e), np.array(f, dtype=float), "cpu_fallback"
    except RuntimeError as e:
        msg = str(e)
        if "CUDA out of memory" in msg:
            torch.cuda.empty_cache()
            atoms.calc = calc_cpu
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            return float(e), np.array(f, dtype=float), "cpu_fallback"
        # 其他错误往外抛，让外面决定是跳过还是停掉
        raise


def process_parquet(parquet_path, output_xyz_path, calc_gpu, calc_cpu, skip_fh):
    """处理单个 parquet：读入 -> SevenNet 推理 -> 写出 extxyz。"""
    print(f"Processing {os.path.basename(parquet_path)}", flush=True)
    start_time = time.perf_counter()

    df = pd.read_parquet(parquet_path, columns=parquet_columns_to_read())

    if ONLY_FINAL_STEP and {"id", "relaxation_step"}.issubset(df.columns):
        df = df.sort_values(["id", "relaxation_step"])
        df = df.groupby("id", as_index=False).tail(1)

    frames = []
    stats = {
        "rows": len(df),
        "written": 0,
        "skipped": 0,
        "cuda": 0,
        "cpu": 0,
        "cpu_fallback": 0,
    }
    
    use_tqdm = sys.stdout.isatty()  # 只有在终端里才用进度条


    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=os.path.basename(parquet_path),
        disable=not use_tqdm,       # 重定向时自动关闭进度条
    ):
        atoms, sid, immid = atoms_from_row(row)

        try:
            e_tot, forces, backend = sevennet_predict(atoms, calc_gpu, calc_cpu)
        except Exception as e:
            sid_str = sid if sid is not None else "None"
            err_msg = str(e)
            print(f"[SKIP] id={sid_str} error={err_msg}")
            if skip_fh is not None:
                skip_fh.write(f"{sid_str}\t{err_msg}\n")
            stats["skipped"] += 1
            continue

        stats[backend] += 1
        stats["written"] += 1
        frames.append(build_output_atoms(atoms, sid, immid, e_tot, forces))

    if frames:
        ensure_dir_for_file(output_xyz_path)
        ase_write(output_xyz_path, frames, format="extxyz")

    elapsed = time.perf_counter() - start_time
    rows_per_s = stats["rows"] / elapsed if elapsed > 0 else 0.0
    written_per_s = stats["written"] / elapsed if elapsed > 0 else 0.0
    print(
        "[SUMMARY] "
        f"rows={stats['rows']} written={stats['written']} skipped={stats['skipped']} "
        f"cuda={stats['cuda']} cpu={stats['cpu']} cpu_fallback={stats['cpu_fallback']} "
        f"elapsed_s={elapsed:.2f} rows_per_s={rows_per_s:.2f} written_per_s={written_per_s:.2f}",
        flush=True,
    )
    return stats, elapsed


def main():
    warnings.filterwarnings("ignore", module="ase")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    validate_accelerator_flags()

    print(
        "Loading SevenNet: "
        f"model={SEVENNET_MODEL}, file_type={SEVENNET_FILE_TYPE}, "
        f"modal={SEVENNET_MODAL}, cuda={torch.cuda.is_available()}, "
        f"flash={SEVENNET_ENABLE_FLASH}, cueq={SEVENNET_ENABLE_CUEQ}, "
        f"oeq={SEVENNET_ENABLE_OEQ}",
        flush=True,
    )

    calc_gpu, calc_cpu = build_sevennet_calculators()

    ensure_dir_for_file(SKIP_LOG)
    skip_fh = open(SKIP_LOG, "w", buffering=1)  # 行缓冲，方便 tail
    overall_stats = {
        "rows": 0,
        "written": 0,
        "skipped": 0,
        "cuda": 0,
        "cpu": 0,
        "cpu_fallback": 0,
    }
    overall_start = time.perf_counter()

    try:
        for idx in range(FIRST_IDX, LAST_IDX + 1):
            fname = f"train-{idx:05d}-of-00023.parquet"
            parquet_path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(parquet_path):
                print(f"[WARN] {parquet_path} not found, skip.")
                continue

            output_xyz_path = os.path.join(
                OUTPUT_DIR,
                fname.replace(".parquet", ".xyz"),
            )

            shard_stats, _ = process_parquet(
                parquet_path, output_xyz_path, calc_gpu, calc_cpu, skip_fh
            )
            for key in overall_stats:
                overall_stats[key] += shard_stats[key]
    finally:
        skip_fh.close()

    overall_elapsed = time.perf_counter() - overall_start
    overall_rows_per_s = overall_stats["rows"] / overall_elapsed if overall_elapsed > 0 else 0.0
    overall_written_per_s = (
        overall_stats["written"] / overall_elapsed if overall_elapsed > 0 else 0.0
    )
    print(
        "[OVERALL] "
        f"rows={overall_stats['rows']} written={overall_stats['written']} "
        f"skipped={overall_stats['skipped']} cuda={overall_stats['cuda']} "
        f"cpu={overall_stats['cpu']} cpu_fallback={overall_stats['cpu_fallback']} "
        f"elapsed_s={overall_elapsed:.2f} rows_per_s={overall_rows_per_s:.2f} "
        f"written_per_s={overall_written_per_s:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
