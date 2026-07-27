import os
import warnings
import sys


import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import write as ase_write
from tqdm import tqdm

import torch
from pymatgen.core import Lattice, Structure
from chgnet.model.model import CHGNet

# ---------- 配置 ----------
# 所有 parquet 所在的目录
DATA_DIR = "../"   # 比如 "/mnt/data/matbench" 之类的路径

# 要处理的 shard 范围：train-00000-of-00371.parquet ~ train-00370-of-00371.parquet
FIRST_SHARD = 00
LAST_SHARD = 10   # 包含 370

ONLY_FINAL_STEP = False            # False: 所有 relax step 都预测
BATCH_SIZE = 64                    # H200 可以先用 64，OOM 会自动兜底
SKIP_LOG = "./skipped_isolated_structs_00-10.txt"  # 所有 shard 共用一个跳过日志
PARQUET_COLUMNS = [
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "dimension_types",
    "id",
    "immutable_id",
]
# -------------------------


def ensure_dir_for_file(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _flatten_to_float_list(x):
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


def _to_numpy(x):
    """兼容 torch.Tensor / numpy / 标量 -> numpy.ndarray."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.array(x)


def parquet_columns_to_read():
    columns = list(PARQUET_COLUMNS)
    if ONLY_FINAL_STEP:
        columns.append("relaxation_step")
    return columns


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_models():
    """同时建一个 GPU 模型和一个 CPU 模型，方便需要时降级。"""
    gpu_device = get_device()
    chgnet_gpu = CHGNet.load(model_name="r2scan", use_device=gpu_device)
    chgnet_cpu = CHGNet.load(model_name="r2scan", use_device="cpu")
    return chgnet_gpu, chgnet_cpu


def atoms_and_structure_from_row(row):
    """从一行 parquet 生成 ASE Atoms + pymatgen Structure + id 信息。"""
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

    lattice = Lattice(cell)
    structure = Structure(lattice, symbols, pos, coords_are_cartesian=True)

    # id：pbe -> r2scan
    raw_id = row.get("id")
    if isinstance(raw_id, str):
        sid = raw_id.replace("pbe", "r2scan")
    else:
        sid = raw_id
    immid = row.get("immutable_id")

    return atoms, structure, sid, immid


def prediction_to_energy_forces(pred, natoms):
    e_arr = _to_numpy(pred["e"])
    e_per_atom = float(e_arr.reshape(-1).mean())
    e_tot = e_per_atom * natoms

    f_arr = _to_numpy(pred["f"])
    forces = f_arr.reshape(natoms, 3)
    return e_tot, forces


def build_output_atoms(atoms: Atoms, sid, immid, energy, forces):
    atoms_out = atoms.copy()
    atoms_out.info = {
        "id": sid,
        "immutable_id": immid,
        "energy": energy,
    }
    atoms_out.set_array("forces", forces.astype(float))
    return atoms_out


def predict_batch_gpu(chgnet_gpu, structures):
    preds = chgnet_gpu.predict_structure(
        structures,
        task="ef",
        batch_size=len(structures),
    )
    if isinstance(preds, dict):
        preds = [preds]
    return preds


def predict_single_cpu(chgnet_cpu, structure):
    pred = chgnet_cpu.predict_structure(
        structure,
        task="ef",
        batch_size=1,
    )
    return pred


def process_one(chgnet_gpu, chgnet_cpu, atoms_i, struct_i, sid_i, immid_i, frames, skip_fh):
    """处理单个结构：优先 GPU，不行再 CPU；孤立原子则跳过。"""
    # 先 GPU 单个结构
    try:
        pred = predict_batch_gpu(chgnet_gpu, [struct_i])[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        try:
            pred = predict_single_cpu(chgnet_cpu, struct_i)
        except ValueError as e:
            msg = str(e)
            if "isolated atom" in msg:
                sid_str = sid_i if sid_i is not None else "None"
                print(f"[SKIP][CPU isolated] id={sid_str}")
                if skip_fh is not None:
                    skip_fh.write(f"{sid_str}\tCPU_isolated\n")
                return
            else:
                raise
    except ValueError as e:
        msg = str(e)
        if "isolated atom" in msg:
            sid_str = sid_i if sid_i is not None else "None"
            print(f"[SKIP][GPU isolated] id={sid_str}")
            if skip_fh is not None:
                skip_fh.write(f"{sid_str}\tGPU_isolated\n")
            return
        else:
            raise

    # 走到这里说明有正常预测结果
    natoms = len(atoms_i)
    e_tot, forces = prediction_to_energy_forces(pred, natoms)
    frames.append(build_output_atoms(atoms_i, sid_i, immid_i, e_tot, forces))


def process_batch(chgnet_gpu, chgnet_cpu, atoms_buf, struct_buf, frames, skip_fh):
    """处理一个 batch：先尝试 GPU 一批跑，不行再单个处理。"""
    try:
        preds = predict_batch_gpu(chgnet_gpu, struct_buf)

        for (atoms_i, sid_i, immid_i), pred in zip(atoms_buf, preds):
            natoms = len(atoms_i)
            e_tot, forces = prediction_to_energy_forces(pred, natoms)
            frames.append(build_output_atoms(atoms_i, sid_i, immid_i, e_tot, forces))

    except torch.cuda.OutOfMemoryError:
        # 这一批太大：逐个结构处理（GPU/CPU fallback）
        torch.cuda.empty_cache()
        for (atoms_i, sid_i, immid_i), struct_i in zip(atoms_buf, struct_buf):
            process_one(chgnet_gpu, chgnet_cpu, atoms_i, struct_i, sid_i, immid_i, frames, skip_fh)

    except ValueError as e:
        msg = str(e)
        if "isolated atom" in msg:
            # 这一批里有孤立原子：逐个结构处理，分别判断谁该跳
            for (atoms_i, sid_i, immid_i), struct_i in zip(atoms_buf, struct_buf):
                process_one(chgnet_gpu, chgnet_cpu, atoms_i, struct_i, sid_i, immid_i, frames, skip_fh)
        else:
            raise


def process_one_parquet(parquet_path, output_xyz, chgnet_gpu, chgnet_cpu, skip_fh):
    """处理一个 parquet 文件，并写出对应的 xyz。"""
    if not os.path.exists(parquet_path):
        print(f"[WARN] {parquet_path} 不存在，跳过")
        return

    print(f"\n========== 处理文件：{parquet_path} ==========", flush=True)
    df = pd.read_parquet(parquet_path, columns=parquet_columns_to_read())

    if ONLY_FINAL_STEP and {"id", "relaxation_step"}.issubset(df.columns):
        df = df.sort_values(["id", "relaxation_step"])
        df = df.groupby("id", as_index=False).tail(1)

    frames = []
    atoms_buf = []   # [(atoms, id, immutable_id), ...]
    struct_buf = []  # [Structure, ...]

    use_tqdm = sys.stdout.isatty()  # 只有在终端里才用进度条

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Predicting {os.path.basename(parquet_path)}",
        disable=not use_tqdm,      # 重定向到文件时自动关闭 tqdm
    ):
        atoms, struct, sid, immid = atoms_and_structure_from_row(row)
        atoms_buf.append((atoms, sid, immid))
        struct_buf.append(struct)

        if len(struct_buf) >= BATCH_SIZE:
            process_batch(chgnet_gpu, chgnet_cpu, atoms_buf, struct_buf, frames, skip_fh)
            atoms_buf.clear()
            struct_buf.clear()

    # 处理最后不满一批的
    if struct_buf:
        process_batch(chgnet_gpu, chgnet_cpu, atoms_buf, struct_buf, frames, skip_fh)

    if frames:
        ensure_dir_for_file(output_xyz)
        ase_write(output_xyz, frames, format="extxyz")
        print(f"[OK] 写出 {len(frames)} 帧到 {output_xyz}", flush=True)
    else:
        print(f"[WARN] {parquet_path} 没有任何有效帧（可能全是孤立原子或出错）")


def main():
    warnings.filterwarnings("ignore", module="pymatgen")
    warnings.filterwarnings("ignore", module="ase")

    # 模型只加载一次
    print("加载 CHGNet 模型（GPU + CPU）...", flush=True)
    chgnet_gpu, chgnet_cpu = build_models()

    # 所有 shard 共用一个 skip log（追加写）
    ensure_dir_for_file(SKIP_LOG)
    skip_fh = open(SKIP_LOG, "w", buffering=1) if SKIP_LOG is not None else None

    try:
        for shard in range(FIRST_SHARD, LAST_SHARD + 1):
            # 构造文件名：train-00000-of-00371.parquet
            parquet_name = f"train-{shard:05d}-of-00023.parquet"
            parquet_path = os.path.join(DATA_DIR, parquet_name)

            # 对应的输出 xyz 名称
            xyz_name = parquet_name.replace(".parquet", ".xyz")
            output_xyz = os.path.join(".", xyz_name)

            process_one_parquet(parquet_path, output_xyz, chgnet_gpu, chgnet_cpu, skip_fh)

    finally:
        if skip_fh is not None:
            skip_fh.close()


if __name__ == "__main__":
    main()
