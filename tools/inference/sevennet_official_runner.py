import importlib
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from torch_geometric.data import Batch
from tqdm import tqdm

try:
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    from sevenn.sevennet_calculator import SevenNetCalculator
import sevenn._keys as KEY
from sevenn.atom_graph_data import AtomGraphData
from sevenn.train.dataload import unlabeled_atoms_to_graph


PARQUET_BASE_COLUMNS = [
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "dimension_types",
    "id",
    "immutable_id",
]


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


def parquet_columns_to_read(only_final_step: bool):
    columns = list(PARQUET_BASE_COLUMNS)
    if only_final_step:
        columns.append("relaxation_step")
    return columns


def atoms_from_row(row):
    cell_vals = _flatten_to_float_list(row["lattice_vectors"])
    cell = np.array(cell_vals, dtype=float).reshape(3, 3)

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

    raw_id = row.get("id")
    if isinstance(raw_id, str):
        sid = raw_id.replace("pbe", "r2scan")
    else:
        sid = raw_id
    immid = row.get("immutable_id")

    return atoms, sid, immid


def atoms_from_extxyz(atoms: Atoms):
    atoms = atoms.copy()
    raw_id = atoms.info.get("id")
    if isinstance(raw_id, str):
        sid = raw_id.replace("pbe", "r2scan")
    else:
        sid = raw_id
    immid = atoms.info.get("immutable_id")
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


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x)


def _helper_available(module_name: str, func_name: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        checker = getattr(module, func_name)
        return bool(checker())
    except Exception:
        return False


def resolve_accelerator(preference: str):
    availability = {
        "flash": _helper_available("sevenn.nn.flash_helper", "is_flash_available"),
        "cueq": _helper_available("sevenn.nn.cue_helper", "is_cue_available"),
        "oeq": _helper_available("sevenn.nn.oeq_helper", "is_oeq_available"),
    }

    pref = (preference or "auto").lower()
    if pref == "none":
        selected = "none"
    elif pref == "auto":
        selected = "none"
        for name in ("flash", "cueq", "oeq"):
            if availability[name]:
                selected = name
                break
    elif pref in availability and availability[pref]:
        selected = pref
    else:
        selected = "none"

    flags = {
        "enable_flash": selected == "flash",
        "enable_cueq": selected == "cueq",
        "enable_oeq": selected == "oeq",
    }
    return selected, flags, availability


def create_calculator(model, file_type, modal, device, accelerator_flags):
    kwargs = {"model": model, "device": device}
    if file_type is not None:
        kwargs["file_type"] = file_type
    if modal is not None:
        kwargs["modal"] = modal
    if device == "cuda":
        kwargs.update(accelerator_flags)
    return SevenNetCalculator(**kwargs)


def build_calculators(model, file_type, modal_candidates, accelerator_flags):
    candidates = list(modal_candidates) if modal_candidates else [None]
    cpu_errors = []
    calc_cpu = None
    selected_modal = None

    for modal in candidates:
        try:
            calc_cpu = create_calculator(model, file_type, modal, "cpu", {})
            selected_modal = modal
            break
        except Exception as exc:
            cpu_errors.append(f"modal={modal!r}: {exc}")

    if calc_cpu is None:
        joined = "\n".join(cpu_errors)
        raise RuntimeError(f"Failed to build CPU SevenNet calculator.\n{joined}")

    calc_gpu = None
    if torch.cuda.is_available():
        try:
            calc_gpu = create_calculator(
                model, file_type, selected_modal, "cuda", accelerator_flags
            )
        except Exception as exc:
            print(
                f"[WARN] Failed to build CUDA calculator with modal={selected_modal!r}: {exc}",
                flush=True,
            )

    return calc_gpu, calc_cpu, selected_modal


def sevennet_predict(atoms: Atoms, calc_gpu, calc_cpu):
    if calc_gpu is None:
        atoms.calc = calc_cpu
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        return float(e), np.array(f, dtype=float), "cpu"

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
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            torch.cuda.empty_cache()
            atoms.calc = calc_cpu
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            return float(e), np.array(f, dtype=float), "cpu_fallback"
        raise


def atoms_to_graph_data(atoms: Atoms, calc) -> AtomGraphData:
    graph = unlabeled_atoms_to_graph(atoms, calc.cutoff)
    data = AtomGraphData.from_numpy_dict(graph)
    if getattr(calc, "modal", None) is not None:
        data[KEY.DATA_MODALITY] = calc.modal
    return data


def split_batched_output(output, natoms_list):
    total_atoms = sum(natoms_list)
    forces = _to_numpy(output[KEY.PRED_FORCE]).reshape(-1, 3)
    if forces.shape[0] < total_atoms:
        raise RuntimeError(
            f"Unexpected batched force shape {forces.shape}, expected at least {total_atoms} atoms"
        )

    energies = None
    pred_total = output.get(KEY.PRED_TOTAL_ENERGY)
    if pred_total is not None:
        pred_total = _to_numpy(pred_total).reshape(-1)
        if pred_total.size == len(natoms_list):
            energies = [float(v) for v in pred_total]
        elif pred_total.size == 1 and len(natoms_list) == 1:
            energies = [float(pred_total[0])]

    if energies is None:
        atomic_energy = output.get(KEY.ATOMIC_ENERGY)
        if atomic_energy is None:
            raise RuntimeError("Batched output does not include usable energy keys")
        atomic_energy = _to_numpy(atomic_energy).reshape(-1)
        if atomic_energy.size < total_atoms:
            raise RuntimeError(
                f"Unexpected batched atomic energy shape {atomic_energy.shape}, "
                f"expected at least {total_atoms} atoms"
            )
        energies = []
        start = 0
        for natoms in natoms_list:
            stop = start + natoms
            energies.append(float(atomic_energy[start:stop].sum()))
            start = stop

    results = []
    start = 0
    for energy, natoms in zip(energies, natoms_list):
        stop = start + natoms
        results.append((energy, forces[start:stop].astype(float)))
        start = stop
    return results


def sevennet_predict_batch(items, calc, backend_label):
    if not items:
        return []

    model = calc.model
    prev_is_batch = getattr(model, "is_batch_data", None)
    if hasattr(model, "set_is_batch_data"):
        model.set_is_batch_data(True)

    try:
        graph_list = [item["graph"] for item in items]
        natoms_list = [len(item["atoms"]) for item in items]
        batch = Batch.from_data_list(graph_list)
        batch = batch.to(calc.device)
        output = model(batch)
        split = split_batched_output(output, natoms_list)
        return [(energy, forces, backend_label) for energy, forces in split]
    finally:
        if prev_is_batch is not None and hasattr(model, "set_is_batch_data"):
            model.set_is_batch_data(prev_is_batch)


def process_item_single(item, calc_gpu, calc_cpu, frames, skip_fh, stats):
    atoms = item["atoms"]
    sid = item["sid"]
    immid = item["immid"]
    try:
        e_tot, forces, backend = sevennet_predict(atoms, calc_gpu, calc_cpu)
    except Exception as exc:
        sid_str = sid if sid is not None else "None"
        err_msg = str(exc)
        print(f"[SKIP] id={sid_str} error={err_msg}", flush=True)
        if skip_fh is not None:
            skip_fh.write(f"{sid_str}\t{err_msg}\n")
        stats["skipped"] += 1
        return

    stats[backend] += 1
    stats["written"] += 1
    frames.append(build_output_atoms(atoms, sid, immid, e_tot, forces))


def process_items(items, calc_gpu, calc_cpu, frames, skip_fh, stats, preferred_batch_size):
    if not items:
        return

    if preferred_batch_size <= 1 or calc_gpu is None:
        for item in items:
            process_item_single(item, calc_gpu, calc_cpu, frames, skip_fh, stats)
        return

    start = 0
    chunk_size = min(preferred_batch_size, len(items))
    while start < len(items):
        end = min(start + chunk_size, len(items))
        chunk = items[start:end]
        try:
            batch_results = sevennet_predict_batch(chunk, calc_gpu, "cuda")
            for item, (e_tot, forces, backend) in zip(chunk, batch_results):
                stats[backend] += 1
                stats["written"] += 1
                frames.append(
                    build_output_atoms(item["atoms"], item["sid"], item["immid"], e_tot, forces)
                )
            start = end
            chunk_size = min(preferred_batch_size, len(items) - start) if start < len(items) else 0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk_size > 1:
                chunk_size = max(1, chunk_size // 2)
                continue
            process_item_single(chunk[0], calc_gpu, calc_cpu, frames, skip_fh, stats)
            start = end
            chunk_size = min(preferred_batch_size, len(items) - start) if start < len(items) else 0
        except RuntimeError as exc:
            msg = str(exc)
            if "CUDA out of memory" in msg and chunk_size > 1:
                torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                continue
            if chunk_size > 1:
                print(
                    f"[WARN] batch chunk_size={chunk_size} failed, retry smaller chunks: {msg}",
                    flush=True,
                )
                chunk_size = max(1, chunk_size // 2)
                continue
            process_item_single(chunk[0], calc_gpu, calc_cpu, frames, skip_fh, stats)
            start = end
            chunk_size = min(preferred_batch_size, len(items) - start) if start < len(items) else 0


def process_parquet(
    parquet_path,
    output_xyz_path,
    calc_gpu,
    calc_cpu,
    skip_fh,
    only_final_step,
    preferred_batch_size,
):
    print(f"Processing {os.path.basename(parquet_path)}", flush=True)
    start_time = time.perf_counter()

    df = pd.read_parquet(
        parquet_path, columns=parquet_columns_to_read(only_final_step)
    )

    if only_final_step and {"id", "relaxation_step"}.issubset(df.columns):
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
    use_tqdm = sys.stdout.isatty()
    graph_builder_calc = calc_gpu if calc_gpu is not None else calc_cpu
    pending_items = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=os.path.basename(parquet_path),
        disable=not use_tqdm,
    ):
        sid = None
        try:
            atoms, sid, immid = atoms_from_row(row)
            graph = (
                atoms_to_graph_data(atoms, graph_builder_calc)
                if preferred_batch_size > 1
                else None
            )
        except Exception as exc:
            sid_str = sid if sid is not None else "None"
            err_msg = str(exc)
            print(f"[SKIP] id={sid_str} error={err_msg}", flush=True)
            if skip_fh is not None:
                skip_fh.write(f"{sid_str}\t{err_msg}\n")
            stats["skipped"] += 1
            continue

        pending_items.append(
            {
                "atoms": atoms,
                "sid": sid,
                "immid": immid,
                "graph": graph,
            }
        )
        if len(pending_items) >= preferred_batch_size:
            process_items(
                pending_items,
                calc_gpu,
                calc_cpu,
                frames,
                skip_fh,
                stats,
                preferred_batch_size,
            )
            pending_items.clear()

    if pending_items:
        process_items(
            pending_items,
            calc_gpu,
            calc_cpu,
            frames,
            skip_fh,
            stats,
            preferred_batch_size,
        )

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
    return stats


def process_xyz_file(
    xyz_path,
    output_xyz_path,
    calc_gpu,
    calc_cpu,
    skip_fh,
    preferred_batch_size,
    xyz_index=":",
    xyz_format="extxyz",
):
    print(f"Processing {os.path.basename(xyz_path)}", flush=True)
    start_time = time.perf_counter()

    frames_in = ase_read(xyz_path, index=xyz_index, format=xyz_format)
    if isinstance(frames_in, Atoms):
        frames_in = [frames_in]
    else:
        frames_in = list(frames_in)

    frames_out = []
    stats = {
        "rows": len(frames_in),
        "written": 0,
        "skipped": 0,
        "cuda": 0,
        "cpu": 0,
        "cpu_fallback": 0,
    }
    use_tqdm = sys.stdout.isatty()
    graph_builder_calc = calc_gpu if calc_gpu is not None else calc_cpu
    pending_items = []

    for atoms_in in tqdm(
        frames_in,
        total=len(frames_in),
        desc=os.path.basename(xyz_path),
        disable=not use_tqdm,
    ):
        sid = None
        try:
            atoms, sid, immid = atoms_from_extxyz(atoms_in)
            graph = (
                atoms_to_graph_data(atoms, graph_builder_calc)
                if preferred_batch_size > 1
                else None
            )
        except Exception as exc:
            sid_str = sid if sid is not None else "None"
            err_msg = str(exc)
            print(f"[SKIP] id={sid_str} error={err_msg}", flush=True)
            if skip_fh is not None:
                skip_fh.write(f"{sid_str}\t{err_msg}\n")
            stats["skipped"] += 1
            continue

        pending_items.append(
            {
                "atoms": atoms,
                "sid": sid,
                "immid": immid,
                "graph": graph,
            }
        )
        if len(pending_items) >= preferred_batch_size:
            process_items(
                pending_items,
                calc_gpu,
                calc_cpu,
                frames_out,
                skip_fh,
                stats,
                preferred_batch_size,
            )
            pending_items.clear()

    if pending_items:
        process_items(
            pending_items,
            calc_gpu,
            calc_cpu,
            frames_out,
            skip_fh,
            stats,
            preferred_batch_size,
        )

    if frames_out:
        ensure_dir_for_file(output_xyz_path)
        ase_write(output_xyz_path, frames_out, format="extxyz")

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


def run_with_settings(settings):
    warnings.filterwarnings("ignore", module="ase")

    data_dir = settings.get("DATA_DIR", ".")
    output_dir = settings["OUTPUT_DIR"]
    first_idx = settings["FIRST_IDX"]
    last_idx = settings["LAST_IDX"]
    only_final_step = settings.get("ONLY_FINAL_STEP", False)
    model = settings["SEVENNET_MODEL"]
    file_type = settings.get("SEVENNET_FILE_TYPE")
    modal_candidates = settings.get("SEVENNET_MODAL_CANDIDATES", [None])
    accelerator_pref = settings.get("ACCELERATOR_PREFERENCE", "auto")
    preferred_batch_size = int(settings.get("SEVENNET_BATCH_SIZE", 1))
    skip_log = settings.get("SKIP_LOG", os.path.join(output_dir, "skipped.txt"))
    input_format = settings.get("INPUT_FORMAT", "parquet").lower()
    filename_template = settings.get(
        "FILENAME_TEMPLATE", "train-{idx:05d}-of-00023.parquet"
    )
    input_file = settings.get("INPUT_FILE")
    input_files = settings.get("INPUT_FILES")
    xyz_index = settings.get("XYZ_INDEX", ":")
    xyz_format = settings.get("XYZ_FORMAT", "extxyz")
    label = settings.get("MODEL_LABEL", model)

    os.makedirs(output_dir, exist_ok=True)
    selected_accel, accel_flags, availability = resolve_accelerator(accelerator_pref)
    print(
        "Loading SevenNet: "
        f"label={label}, model={model}, file_type={file_type}, "
        f"modal_candidates={modal_candidates}, cuda={torch.cuda.is_available()}, "
        f"selected_accelerator={selected_accel}, accelerator_availability={availability}, "
        f"batch_size={preferred_batch_size}",
        flush=True,
    )

    calc_gpu, calc_cpu, selected_modal = build_calculators(
        model, file_type, modal_candidates, accel_flags
    )
    print(f"[INFO] selected_modal={selected_modal!r}", flush=True)

    ensure_dir_for_file(skip_log)
    skip_fh = open(skip_log, "w", buffering=1)
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
        if input_format == "xyz":
            if input_files is None:
                input_paths = [input_file] if input_file else []
            else:
                input_paths = list(input_files)

            if not input_paths:
                raise ValueError(
                    "INPUT_FORMAT='xyz' requires INPUT_FILE or INPUT_FILES in SETTINGS"
                )

            for xyz_path in input_paths:
                resolved_xyz_path = (
                    xyz_path
                    if os.path.isabs(xyz_path)
                    else os.path.join(data_dir, xyz_path)
                )
                if not os.path.exists(resolved_xyz_path):
                    print(f"[WARN] {resolved_xyz_path} not found, skip.", flush=True)
                    continue

                output_name = os.path.basename(resolved_xyz_path)
                output_xyz_path = os.path.join(output_dir, output_name)
                shard_stats = process_xyz_file(
                    resolved_xyz_path,
                    output_xyz_path,
                    calc_gpu,
                    calc_cpu,
                    skip_fh,
                    preferred_batch_size,
                    xyz_index=xyz_index,
                    xyz_format=xyz_format,
                )
                for key in overall_stats:
                    overall_stats[key] += shard_stats[key]
        else:
            for idx in range(first_idx, last_idx + 1):
                fname = filename_template.format(idx=idx)
                parquet_path = os.path.join(data_dir, fname)
                if not os.path.exists(parquet_path):
                    print(f"[WARN] {parquet_path} not found, skip.", flush=True)
                    continue

                output_xyz_path = os.path.join(output_dir, fname.replace(".parquet", ".xyz"))
                shard_stats = process_parquet(
                    parquet_path,
                    output_xyz_path,
                    calc_gpu,
                    calc_cpu,
                    skip_fh,
                    only_final_step,
                    preferred_batch_size,
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
