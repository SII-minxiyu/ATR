#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from compare_step2501_pairwise_100ps_with_dft12p5 import load_dft, load_mlmd, plot_series


def main() -> None:
    out_dir = Path("runs/comparison_step2501_base_vs_baseline_100ps")
    series = {
        "DFT_r2SCAN_12p5ps_from_step2501": load_dft(),
        "finetuned_balanced": load_mlmd("finetuned_balanced", Path("runs/run_008_step2501_base_300K_100ps")),
        "baseline_random_e20": load_mlmd(
            "baseline_random_e20",
            Path("runs/run_010_step2501_baseline_random_e20_300K_100ps"),
        ),
    }
    plot_series(
        series,
        "Fmax",
        "max |F| (eV/A)",
        out_dir / "fmax_compare_with_dft12p5.png",
        ylim=(None, 15),
    )
    print(f"wrote: {out_dir / 'fmax_compare_with_dft12p5.png'}")


if __name__ == "__main__":
    main()
