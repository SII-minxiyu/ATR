#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


RUNS = {
    "finetuned_balanced": Path("runs/run_008_step2501_base_300K_100ps"),
    "baseline_random_e20": Path("runs/run_010_step2501_baseline_random_e20_300K_100ps"),
}
OUT_DIR = Path("runs/comparison_step2501_base_vs_baseline_100ps")


def parse_log(run_dir: Path) -> np.ndarray:
    rows = []
    for line in (run_dir / "mlmd.log").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("Time"):
            rows.append([float(x) for x in line.split()])
    return np.array(rows, dtype=float)


def load_series() -> dict[str, dict[str, np.ndarray]]:
    series = {}
    for label, run_dir in RUNS.items():
        log = parse_log(run_dir)
        natoms = len(read(run_dir / "final_structure.extxyz"))
        series[label] = {
            "time": log[:, 0],
            "T_K": log[:, 4],
            "Epot_atom": log[:, 2] / natoms,
            "Ekin": log[:, 3],
        }
    return series


def padded_ylim(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = max(high - low, 1e-8)
    return low - pad * span, high + pad * span


def plot_base_full_baseline_pre_spike(
    series: dict[str, dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    filename: str,
    spike_time: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    colors = {
        "finetuned_balanced": "#1f77b4",
        "baseline_random_e20": "#d62728",
    }

    y_for_ylim = []
    spike_val = None
    for label, data in series.items():
        t = data["time"]
        y = data[key]
        keep = t < spike_time if label == "baseline_random_e20" else np.ones_like(t, dtype=bool)
        ax.plot(t[keep], y[keep], lw=1.35, color=colors[label], label=label)
        if label == "baseline_random_e20":
            y_for_ylim.append(y[keep])
        else:
            y_for_ylim.append(y[keep])

    ylim = padded_ylim(np.concatenate(y_for_ylim))
    ax.set_ylim(*ylim)

    if key == "T_K":
        ax.axhline(300, color="#555555", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlim(-10.0, 100.0)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Time since source_step=2501 start (ps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series = load_series()
    baseline = series["baseline_random_e20"]
    spike_idx = int(np.argmax(baseline["T_K"]))
    spike_time = float(baseline["time"][spike_idx])

    plot_base_full_baseline_pre_spike(
        series,
        "T_K",
        "Temperature (K)",
        "temperature_finetuned_full_baseline_pre_spike_clipped.png",
        spike_time,
    )
    plot_base_full_baseline_pre_spike(
        series,
        "Epot_atom",
        "Epot/atom (eV)",
        "epot_atom_finetuned_full_baseline_pre_spike_clipped.png",
        spike_time,
    )
    plot_base_full_baseline_pre_spike(
        series,
        "Ekin",
        "Ekin (eV)",
        "ekin_finetuned_full_baseline_pre_spike_clipped.png",
        spike_time,
    )
    print(f"spike_time_ps: {spike_time}")
    print(f"wrote: {OUT_DIR}")


if __name__ == "__main__":
    main()
