from __future__ import annotations

import argparse
import os
import sys
import time
import types
import warnings


MODEL_LABEL = "mace-mh-1-matpes_r2scan"

# ---------- 默认配置 ----------
DATA_DIR = "."
OUTPUT_DIR = f"./{MODEL_LABEL}/pred_xyz"
FIRST_IDX = 0
LAST_IDX = 22
ONLY_FINAL_STEP = False
FILENAME_TEMPLATE = "train-{idx:05d}-of-00023.parquet"
WRITE_CHUNK_SIZE = 256
SKIP_LOG = f"./{MODEL_LABEL}/skipped.txt"

MACE_MODEL = "mh-1"
MACE_HEAD = "matpes_r2scan"
DEFAULT_DTYPE = "float32"
DEVICE_PREFERENCE = "auto"  # auto / cuda / cpu
BATCH_SIZE = 8
ENABLE_CUEQ = False
ENABLE_OEQ = False

PARQUET_BASE_COLUMNS = [
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "dimension_types",
    "id",
    "immutable_id",
]
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run batched MACE-MH-1 (head=matpes_r2scan) on parquet shards and write extxyz labels."
        )
    )
    parser.add_argument("--data-dir", default=DATA_DIR, help="Parquet 所在目录。")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="extxyz 输出目录。")
    parser.add_argument(
        "--xyz-input",
        default=None,
        help="单个 xyz/extxyz 文件路径。设置后忽略 parquet shard 逻辑。",
    )
    parser.add_argument(
        "--first-idx",
        type=int,
        default=FIRST_IDX,
        help="起始 shard 序号（含）。",
    )
    parser.add_argument(
        "--last-idx",
        type=int,
        default=LAST_IDX,
        help="结束 shard 序号（含）。",
    )
    parser.add_argument(
        "--filename-template",
        default=FILENAME_TEMPLATE,
        help='Parquet 文件模板，例如 "train-{idx:05d}-of-00023.parquet"。',
    )
    parser.add_argument(
        "--only-final-step",
        action="store_true",
        default=ONLY_FINAL_STEP,
        help="如果 parquet 里有 relaxation_step，只保留每个 id 的最后一步。",
    )
    parser.add_argument(
        "--write-chunk-size",
        type=int,
        default=WRITE_CHUNK_SIZE,
        help="累计多少帧后刷写一次 extxyz，避免单个 shard 全堆在内存里。",
    )
    parser.add_argument(
        "--skip-log",
        default=SKIP_LOG,
        help="跳过样本日志文件路径。",
    )
    parser.add_argument(
        "--model",
        default=MACE_MODEL,
        help="MACE 模型名或本地模型路径。默认是官方 foundation model: mh-1。",
    )
    parser.add_argument(
        "--head",
        default=MACE_HEAD,
        help="多头模型的 head 名称。默认是 matpes_r2scan。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="真正的推理 batch size。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=DEVICE_PREFERENCE,
        help="推理设备偏好。",
    )
    parser.add_argument(
        "--default-dtype",
        choices=("float32", "float64"),
        default=DEFAULT_DTYPE,
        help="传给 MACE 的默认浮点精度。",
    )
    parser.add_argument(
        "--enable-cueq",
        action="store_true",
        default=ENABLE_CUEQ,
        help="启用 cuEquivariance CUDA 加速。",
    )
    parser.add_argument(
        "--enable-oeq",
        action="store_true",
        default=ENABLE_OEQ,
        help="启用 OpenEquivariance 加速。",
    )
    parser.add_argument(
        "--overwrite-output",
        dest="overwrite_output",
        action="store_true",
        default=True,
        help="如果输出 xyz 已存在，则覆盖重写。",
    )
    parser.add_argument(
        "--no-overwrite-output",
        dest="overwrite_output",
        action="store_false",
        help="如果输出 xyz 已存在，则在末尾继续追加。",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size 必须大于 0。")
    if args.write_chunk_size <= 0:
        raise SystemExit("--write-chunk-size 必须大于 0。")
    if args.enable_cueq and args.enable_oeq:
        raise SystemExit("--enable-cueq 和 --enable-oeq 不能同时开启。")

    return args


def ensure_dir_for_file(path: str):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def flatten_to_float_list(x, np_module):
    out = []

    def _walk(value):
        if isinstance(value, np_module.ndarray):
            for item in value.tolist():
                _walk(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
            return
        out.append(float(value))

    _walk(x)
    return out


def parquet_columns_to_read(only_final_step: bool):
    columns = list(PARQUET_BASE_COLUMNS)
    if only_final_step:
        columns.append("relaxation_step")
    return columns


def import_runtime_dependencies():
    missing = []

    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
        np = None

    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
        pd = None

    try:
        from ase import Atoms
        from ase.io import read as ase_read, write as ase_write
    except ImportError:
        missing.extend(["ase", "ase.io"])
        Atoms = None
        ase_read = None
        ase_write = None

    try:
        import torch
    except ImportError:
        missing.append("torch")
        torch = None
    else:
        compiler_ns = getattr(torch, "compiler", None)
        if compiler_ns is None:
            compiler_ns = types.SimpleNamespace()
            torch.compiler = compiler_ns
        if not hasattr(compiler_ns, "is_compiling"):
            fallback_is_compiling = None
            dynamo_ns = getattr(torch, "_dynamo", None)
            external_utils = getattr(dynamo_ns, "external_utils", None)
            if external_utils is not None:
                fallback_is_compiling = getattr(external_utils, "is_compiling", None)
            if fallback_is_compiling is None:
                fallback_is_compiling = lambda: False
            torch.compiler.is_compiling = fallback_is_compiling

    try:
        import mace as mace_pkg
        from mace import data as mace_data
        from mace.calculators import mace_mp
        from mace.tools import torch_geometric, torch_tools, utils
    except ImportError:
        missing.append("mace-torch")
        mace_pkg = None
        mace_data = None
        mace_mp = None
        torch_geometric = None
        torch_tools = None
        utils = None

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(iterable, **_kwargs):
            return iterable

    try:
        from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    except ImportError:
        run_e3nn_to_cueq = None

    try:
        from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq
    except ImportError:
        run_e3nn_to_oeq = None

    if missing:
        unique_missing = []
        for name in missing:
            if name not in unique_missing:
                unique_missing.append(name)
        joined = ", ".join(unique_missing)
        raise SystemExit(
            "缺少运行依赖："
            f"{joined}\n"
            "建议先进入带 PyTorch 的环境，再安装 MACE：\n"
            "  pip install mace-torch\n"
            "MACE 官方安装说明：\n"
            "  https://mace-docs.readthedocs.io/en/latest/guide/installation.html"
        )

    return {
        "np": np,
        "pd": pd,
        "Atoms": Atoms,
        "ase_read": ase_read,
        "ase_write": ase_write,
        "torch": torch,
        "mace_pkg": mace_pkg,
        "mace_data": mace_data,
        "mace_mp": mace_mp,
        "torch_geometric": torch_geometric,
        "torch_tools": torch_tools,
        "utils": utils,
        "tqdm": tqdm,
        "run_e3nn_to_cueq": run_e3nn_to_cueq,
        "run_e3nn_to_oeq": run_e3nn_to_oeq,
    }


def resolve_device(torch_module, preference: str):
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if torch_module.cuda.is_available():
            return "cuda"
        print("[WARN] --device=cuda，但当前环境没有可用 CUDA，自动回退到 CPU。", flush=True)
        return "cpu"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def load_raw_model_or_die(model_name, device, default_dtype, mace_mp, mace_version):
    try:
        return mace_mp(
            model=model_name,
            device=device,
            default_dtype=default_dtype,
            return_raw_model=True,
        )
    except Exception as exc:
        raise SystemExit(
            "MACE 模型加载失败。\n"
            f"model={model_name!r}, device={device}, mace_version={mace_version}\n"
            f"原始错误：{exc}\n\n"
            "常见原因有两个：\n"
            "1. 当前服务器不能联网，无法自动下载 foundation model。\n"
            "2. 当前 mace-torch 版本过旧，不支持 mh-1。\n\n"
            "建议你先检查：\n"
            "  python -c \"import mace; print(mace.__version__)\"\n\n"
            "如果版本低于 0.3.15，先升级：\n"
            "  pip install -U 'mace-torch>=0.3.15'\n\n"
            "如果服务器不能联网，就先手动下载 Hugging Face 上的 mace-mh-1.model，"
            "然后把 --model 改成本地文件路径，例如：\n"
            "  --model /path/to/mace-mh-1.model --head matpes_r2scan"
        ) from exc


def prepare_model_for_inference(
    model,
    device: str,
    default_dtype: str,
    enable_cueq: bool,
    enable_oeq: bool,
    run_e3nn_to_cueq,
    run_e3nn_to_oeq,
):
    if default_dtype == "float32":
        model = model.float()
    elif default_dtype == "float64":
        model = model.double()

    if device == "cuda" and enable_cueq:
        if run_e3nn_to_cueq is None:
            raise SystemExit(
                "你开启了 --enable-cueq，但当前环境没有可用的 cuEquivariance 转换入口。"
            )
        print("[INFO] Converting GPU model to cuEquivariance.", flush=True)
        model = run_e3nn_to_cueq(model, device=device)

    if device == "cuda" and enable_oeq:
        if run_e3nn_to_oeq is None:
            raise SystemExit(
                "你开启了 --enable-oeq，但当前环境没有可用的 OpenEquivariance 转换入口。"
            )
        print("[INFO] Converting GPU model to OpenEquivariance.", flush=True)
        model = run_e3nn_to_oeq(model, device=device)

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_raw_models(args, deps):
    mace_mp = deps["mace_mp"]
    torch_module = deps["torch"]
    torch_tools = deps["torch_tools"]
    utils = deps["utils"]
    mace_pkg = deps["mace_pkg"]

    torch_tools.set_default_dtype(args.default_dtype)
    mace_version = getattr(mace_pkg, "__version__", "unknown")

    model_cpu = load_raw_model_or_die(
        model_name=args.model,
        device="cpu",
        default_dtype=args.default_dtype,
        mace_mp=mace_mp,
        mace_version=mace_version,
    )
    model_cpu = prepare_model_for_inference(
        model_cpu,
        device="cpu",
        default_dtype=args.default_dtype,
        enable_cueq=False,
        enable_oeq=False,
        run_e3nn_to_cueq=deps["run_e3nn_to_cueq"],
        run_e3nn_to_oeq=deps["run_e3nn_to_oeq"],
    )

    model_heads = getattr(model_cpu, "heads", None)
    if model_heads is not None and args.head not in model_heads:
        raise SystemExit(
            f"请求的 head={args.head!r} 不在模型可用 heads 中，可选值：{list(model_heads)}"
        )

    model_gpu = None
    if args.device == "cuda" and torch_module.cuda.is_available():
        try:
            model_gpu = load_raw_model_or_die(
                model_name=args.model,
                device="cuda",
                default_dtype=args.default_dtype,
                mace_mp=mace_mp,
                mace_version=mace_version,
            )
            model_gpu = prepare_model_for_inference(
                model_gpu,
                device="cuda",
                default_dtype=args.default_dtype,
                enable_cueq=args.enable_cueq,
                enable_oeq=args.enable_oeq,
                run_e3nn_to_cueq=deps["run_e3nn_to_cueq"],
                run_e3nn_to_oeq=deps["run_e3nn_to_oeq"],
            )
        except Exception as exc:
            print(f"[WARN] CUDA 模型初始化失败，改用 CPU: {exc}", flush=True)
            model_gpu = None

    z_table = utils.AtomicNumberTable([int(z) for z in model_cpu.atomic_numbers])
    cutoff = float(model_cpu.r_max)

    return model_gpu, model_cpu, z_table, cutoff, model_heads


def atoms_from_row(row, np_module, Atoms):
    cell_vals = flatten_to_float_list(row["lattice_vectors"], np_module)
    cell = np_module.array(cell_vals, dtype=float).reshape(3, 3)

    pos_vals = flatten_to_float_list(row["cartesian_site_positions"], np_module)
    positions = np_module.array(pos_vals, dtype=float).reshape(-1, 3)

    symbols = list(row["species_at_sites"])
    if len(symbols) != positions.shape[0]:
        raise ValueError(
            f"len(symbols)={len(symbols)} != positions={positions.shape[0]} for id={row.get('id')}"
        )

    dim_types = row.get("dimension_types")
    if dim_types is not None:
        pbc = [bool(x) for x in dim_types]
    else:
        pbc = [True, True, True]

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)

    raw_id = row.get("id")
    if isinstance(raw_id, str):
        sid = raw_id.replace("pbe", "r2scan")
    else:
        sid = raw_id
    immid = row.get("immutable_id")

    return atoms, sid, immid


def atoms_from_xyz_frame(atoms_in, idx: int, xyz_path: str):
    atoms = atoms_in.copy()
    atoms.calc = None

    sid = atoms.info.get("id")
    if sid is None:
        base = os.path.splitext(os.path.basename(xyz_path))[0]
        sid = f"{base}-{idx}"
    if isinstance(sid, str):
        sid = sid.replace("pbe", "r2scan")

    immid = atoms.info.get("immutable_id")
    return atoms, sid, immid


def atomic_data_from_atoms(atoms, head_name, z_table, cutoff, heads, mace_data):
    config = mace_data.config_from_atoms(atoms, head_name=head_name)
    return mace_data.AtomicData.from_config(
        config,
        z_table=z_table,
        cutoff=cutoff,
        heads=heads,
    )


def build_output_atoms(atoms, sid, immid, energy, forces):
    atoms_out = atoms.copy()
    atoms_out.info = dict(atoms.info)
    if "energy" in atoms.info and "source_energy" not in atoms_out.info:
        atoms_out.info["source_energy"] = atoms.info["energy"]
    if "energy_corrected" in atoms.info and "source_energy_corrected" not in atoms_out.info:
        atoms_out.info["source_energy_corrected"] = atoms.info["energy_corrected"]
    atoms_out.info["id"] = sid
    atoms_out.info["immutable_id"] = immid
    atoms_out.info["energy"] = float(energy)
    atoms_out.set_array("forces", forces.astype(float))
    return atoms_out


def flush_frames(ase_write, output_xyz_path: str, frames):
    if not frames:
        return
    ensure_dir_for_file(output_xyz_path)
    ase_write(
        output_xyz_path,
        frames,
        format="extxyz",
        append=os.path.exists(output_xyz_path),
    )
    frames.clear()


def is_cuda_oom(exc, torch_module):
    return isinstance(exc, torch_module.cuda.OutOfMemoryError) or "CUDA out of memory" in str(
        exc
    )


def build_graph_batch(entries, torch_geometric):
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[entry["atomic_data"] for entry in entries],
        batch_size=len(entries),
        shuffle=False,
        drop_last=False,
    )
    return next(iter(loader))


def predict_with_model(entries, model, device, deps):
    np_module = deps["np"]
    torch_geometric = deps["torch_geometric"]
    torch_tools = deps["torch_tools"]

    graph_batch = build_graph_batch(entries, torch_geometric)
    graph_batch = graph_batch.to(device)

    output = model(graph_batch.to_dict(), compute_stress=False)

    energies = torch_tools.to_numpy(output["energy"]).reshape(-1)
    force_array = torch_tools.to_numpy(output["forces"])
    split_points = graph_batch.ptr[1:].detach().cpu().numpy().tolist()
    forces_list = np_module.split(force_array, split_points, axis=0)[:-1]

    if len(energies) != len(entries) or len(forces_list) != len(entries):
        raise RuntimeError(
            "批量推理输出数量与输入不一致："
            f"n_input={len(entries)} n_energy={len(energies)} n_forces={len(forces_list)}"
        )

    outputs = []
    for energy, forces in zip(energies, forces_list):
        outputs.append(
            {
                "ok": True,
                "energy": float(energy),
                "forces": np_module.asarray(forces, dtype=float),
            }
        )
    return outputs


def predict_entries(entries, model_gpu, model_cpu, deps):
    torch_module = deps["torch"]

    if not entries:
        return []

    if model_gpu is None:
        return predict_entries_cpu(entries, model_cpu, deps)

    try:
        outputs = predict_with_model(entries, model_gpu, "cuda", deps)
        for item in outputs:
            item["backend"] = "cuda"
        return outputs
    except Exception as exc:
        if is_cuda_oom(exc, torch_module):
            torch_module.cuda.empty_cache()

        if len(entries) == 1:
            try:
                outputs = predict_with_model(entries, model_cpu, "cpu", deps)
                outputs[0]["backend"] = "cpu_fallback"
                return outputs
            except Exception as cpu_exc:
                return [
                    {
                        "ok": False,
                        "backend": "failed",
                        "error": f"gpu_error={exc}; cpu_error={cpu_exc}",
                    }
                ]

        mid = len(entries) // 2
        left = predict_entries(entries[:mid], model_gpu, model_cpu, deps)
        right = predict_entries(entries[mid:], model_gpu, model_cpu, deps)
        return left + right


def predict_entries_cpu(entries, model_cpu, deps):
    if not entries:
        return []

    try:
        outputs = predict_with_model(entries, model_cpu, "cpu", deps)
        for item in outputs:
            item["backend"] = "cpu"
        return outputs
    except Exception as exc:
        if len(entries) == 1:
            return [{"ok": False, "backend": "failed", "error": str(exc)}]

        mid = len(entries) // 2
        left = predict_entries_cpu(entries[:mid], model_cpu, deps)
        right = predict_entries_cpu(entries[mid:], model_cpu, deps)
        return left + right


def log_skip(skip_fh, sid, err_msg):
    sid_str = sid if sid is not None else "None"
    print(f"[SKIP] id={sid_str} error={err_msg}", flush=True)
    if skip_fh is not None:
        skip_fh.write(f"{sid_str}\t{err_msg}\n")


def process_prediction_batch(
    entries,
    frames,
    output_xyz_path,
    write_chunk_size,
    model_gpu,
    model_cpu,
    skip_fh,
    stats,
    deps,
):
    results = predict_entries(entries, model_gpu, model_cpu, deps)

    for entry, result in zip(entries, results):
        if not result["ok"]:
            log_skip(skip_fh, entry["sid"], result["error"])
            stats["skipped"] += 1
            continue

        stats["written"] += 1
        stats[result["backend"]] += 1
        frames.append(
            build_output_atoms(
                atoms=entry["atoms"],
                sid=entry["sid"],
                immid=entry["immid"],
                energy=result["energy"],
                forces=result["forces"],
            )
        )

    if len(frames) >= write_chunk_size:
        flush_frames(deps["ase_write"], output_xyz_path, frames)


def process_parquet(
    parquet_path: str,
    output_xyz_path: str,
    args,
    model_gpu,
    model_cpu,
    z_table,
    cutoff,
    model_heads,
    skip_fh,
    deps,
):
    np_module = deps["np"]
    pd_module = deps["pd"]
    Atoms = deps["Atoms"]
    mace_data = deps["mace_data"]
    tqdm = deps["tqdm"]

    if not os.path.exists(parquet_path):
        print(f"[WARN] {parquet_path} 不存在，跳过。", flush=True)
        return None

    print(f"Processing {os.path.basename(parquet_path)}", flush=True)
    start_time = time.perf_counter()

    if args.overwrite_output and os.path.exists(output_xyz_path):
        os.remove(output_xyz_path)

    df = pd_module.read_parquet(
        parquet_path,
        columns=parquet_columns_to_read(args.only_final_step),
    )

    if args.only_final_step and {"id", "relaxation_step"}.issubset(df.columns):
        df = df.sort_values(["id", "relaxation_step"])
        df = df.groupby("id", as_index=False).tail(1)

    stats = {
        "rows": len(df),
        "written": 0,
        "skipped": 0,
        "cuda": 0,
        "cpu": 0,
        "cpu_fallback": 0,
    }
    frames = []
    entries = []
    use_tqdm = sys.stdout.isatty()

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=os.path.basename(parquet_path),
        disable=not use_tqdm,
    ):
        sid = row.get("id")
        if isinstance(sid, str):
            sid = sid.replace("pbe", "r2scan")

        try:
            atoms, sid, immid = atoms_from_row(row, np_module, Atoms)
        except Exception as exc:
            log_skip(skip_fh, sid, f"build_atoms_failed: {exc}")
            stats["skipped"] += 1
            continue

        try:
            atomic_data = atomic_data_from_atoms(
                atoms=atoms,
                head_name=args.head,
                z_table=z_table,
                cutoff=cutoff,
                heads=model_heads,
                mace_data=mace_data,
            )
        except Exception as exc:
            log_skip(skip_fh, sid, f"build_atomic_data_failed: {exc}")
            stats["skipped"] += 1
            continue

        entries.append(
            {
                "atoms": atoms,
                "sid": sid,
                "immid": immid,
                "atomic_data": atomic_data,
            }
        )

        if len(entries) >= args.batch_size:
            process_prediction_batch(
                entries=entries,
                frames=frames,
                output_xyz_path=output_xyz_path,
                write_chunk_size=args.write_chunk_size,
                model_gpu=model_gpu,
                model_cpu=model_cpu,
                skip_fh=skip_fh,
                stats=stats,
                deps=deps,
            )
            entries.clear()

    if entries:
        process_prediction_batch(
            entries=entries,
            frames=frames,
            output_xyz_path=output_xyz_path,
            write_chunk_size=args.write_chunk_size,
            model_gpu=model_gpu,
            model_cpu=model_cpu,
            skip_fh=skip_fh,
            stats=stats,
            deps=deps,
        )

    flush_frames(deps["ase_write"], output_xyz_path, frames)

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
    return stats


def process_xyz(
    xyz_path: str,
    output_xyz_path: str,
    args,
    model_gpu,
    model_cpu,
    z_table,
    cutoff,
    model_heads,
    skip_fh,
    deps,
):
    ase_read = deps["ase_read"]
    mace_data = deps["mace_data"]
    tqdm = deps["tqdm"]

    if not os.path.exists(xyz_path):
        print(f"[WARN] {xyz_path} 不存在，跳过。", flush=True)
        return None

    print(f"Processing {os.path.basename(xyz_path)}", flush=True)
    start_time = time.perf_counter()

    if args.overwrite_output and os.path.exists(output_xyz_path):
        os.remove(output_xyz_path)

    frames_in = ase_read(xyz_path, index=":")
    if not isinstance(frames_in, list):
        frames_in = [frames_in]

    stats = {
        "rows": len(frames_in),
        "written": 0,
        "skipped": 0,
        "cuda": 0,
        "cpu": 0,
        "cpu_fallback": 0,
    }
    frames = []
    entries = []
    use_tqdm = sys.stdout.isatty()

    for idx, atoms_in in tqdm(
        enumerate(frames_in),
        total=len(frames_in),
        desc=os.path.basename(xyz_path),
        disable=not use_tqdm,
    ):
        sid = atoms_in.info.get("id")
        if isinstance(sid, str):
            sid = sid.replace("pbe", "r2scan")

        try:
            atoms, sid, immid = atoms_from_xyz_frame(atoms_in, idx, xyz_path)
        except Exception as exc:
            log_skip(skip_fh, sid, f"build_atoms_failed: {exc}")
            stats["skipped"] += 1
            continue

        try:
            atomic_data = atomic_data_from_atoms(
                atoms=atoms,
                head_name=args.head,
                z_table=z_table,
                cutoff=cutoff,
                heads=model_heads,
                mace_data=mace_data,
            )
        except Exception as exc:
            log_skip(skip_fh, sid, f"build_atomic_data_failed: {exc}")
            stats["skipped"] += 1
            continue

        entries.append(
            {
                "atoms": atoms,
                "sid": sid,
                "immid": immid,
                "atomic_data": atomic_data,
            }
        )

        if len(entries) >= args.batch_size:
            process_prediction_batch(
                entries=entries,
                frames=frames,
                output_xyz_path=output_xyz_path,
                write_chunk_size=args.write_chunk_size,
                model_gpu=model_gpu,
                model_cpu=model_cpu,
                skip_fh=skip_fh,
                stats=stats,
                deps=deps,
            )
            entries.clear()

    if entries:
        process_prediction_batch(
            entries=entries,
            frames=frames,
            output_xyz_path=output_xyz_path,
            write_chunk_size=args.write_chunk_size,
            model_gpu=model_gpu,
            model_cpu=model_cpu,
            skip_fh=skip_fh,
            stats=stats,
            deps=deps,
        )

    flush_frames(deps["ase_write"], output_xyz_path, frames)

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
    return stats


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", module="ase")

    deps = import_runtime_dependencies()
    args.device = resolve_device(deps["torch"], args.device)

    print(
        "Loading batched MACE: "
        f"label={MODEL_LABEL}, model={args.model}, head={args.head}, "
        f"device={args.device}, default_dtype={args.default_dtype}, batch_size={args.batch_size}, "
        f"enable_cueq={args.enable_cueq}, enable_oeq={args.enable_oeq}",
        flush=True,
    )
    print(
        "[INFO] 首次运行若本地没有缓存，MACE 会自动下载模型到 ~/.cache/mace",
        flush=True,
    )

    model_gpu, model_cpu, z_table, cutoff, model_heads = build_raw_models(args, deps)
    if model_heads is not None:
        print(f"[INFO] available_heads={list(model_heads)}", flush=True)

    ensure_dir_for_file(args.skip_log)
    skip_fh = open(args.skip_log, "w", buffering=1)

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
        if args.xyz_input is not None:
            output_xyz_path = os.path.join(
                args.output_dir,
                os.path.basename(args.xyz_input),
            )
            xyz_stats = process_xyz(
                xyz_path=args.xyz_input,
                output_xyz_path=output_xyz_path,
                args=args,
                model_gpu=model_gpu,
                model_cpu=model_cpu,
                z_table=z_table,
                cutoff=cutoff,
                model_heads=model_heads,
                skip_fh=skip_fh,
                deps=deps,
            )
            if xyz_stats is not None:
                for key in overall_stats:
                    overall_stats[key] += xyz_stats[key]
        else:
            for idx in range(args.first_idx, args.last_idx + 1):
                filename = args.filename_template.format(idx=idx)
                parquet_path = os.path.join(args.data_dir, filename)
                output_xyz_path = os.path.join(
                    args.output_dir,
                    filename.replace(".parquet", ".xyz"),
                )
                shard_stats = process_parquet(
                    parquet_path=parquet_path,
                    output_xyz_path=output_xyz_path,
                    args=args,
                    model_gpu=model_gpu,
                    model_cpu=model_cpu,
                    z_table=z_table,
                    cutoff=cutoff,
                    model_heads=model_heads,
                    skip_fh=skip_fh,
                    deps=deps,
                )
                if shard_stats is None:
                    continue
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
