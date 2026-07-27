#!/usr/bin/env python
from pathlib import Path
import runpy

SCRIPT = Path(
    "/inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_InSe_48atom_vasp/"
    "model_md/chgnet_r2scan_300K_from_dft_midframes/scripts/compare_three_models_with_dft_extxyz.py"
)

runpy.run_path(str(SCRIPT), run_name="__main__")
