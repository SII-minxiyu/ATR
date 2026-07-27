# HgInSn48 CHGNet ML-MD From DFT Midframes

This folder prepares CHGNet ML-MD runs from equilibrated 300 K r2SCAN DFT-MD frames.

## Inputs

DFT source dataset:

`../../ml_test_dataset_r2scan_HgInSn48/r2scan_300K_5ps/HgInSn48_r2scan_300K_5ps_prod_after500fs_all.extxyz`

Checkpoint:

`/inspire/hdd/global_user/luomingxiang-240108540155/teacher/chgnet_distill_pretrain/finetune_default_graph_balanced_from_e35_e10_ew3_fw1p5_lr5e5/best.pth.tar`

Selected start frames:

- `starts/HgInSn48_source_step_2501.extxyz`
  - 2.501 ps
  - energy `-1537.07453075 eV`
- `starts/HgInSn48_source_step_3501.extxyz`
  - 3.501 ps
  - energy `-1537.96560655 eV`
  - default start for the prepared run
- `starts/HgInSn48_source_step_4501.extxyz`
  - 4.501 ps
  - energy `-1537.65064370 eV`

The matching `.vasp` files are provided only as structure backups; the ML-MD script uses the `.extxyz` files.

## Environment

The Python environment must import all of:

- `torch`
- `ase`
- `pymatgen`
- `chgnet`

Check the environment first:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes
PYTHON=/path/to/python ./check_env.sh
```

The scripts add the local CHGNet source tree to `PYTHONPATH` by default:

`/inspire/hdd/global_user/luomingxiang-240108540155/chgnet`

On the current inspected shell, the default Python has `ase` but `torch` fails to load and `pymatgen` is missing, while the local `actor/data` conda envs have `torch` but miss `ase/pymatgen`. Use or create an environment where all four dependencies work.

## Run Default 300 K NVT ML-MD

Default run:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_HgInSn_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes
PYTHON=/path/to/python DEVICE=cpu ./run_step3501_300K_10ps.sh
```

Useful overrides:

```bash
STEPS=1000 LOGINTERVAL=5 PYTHON=/path/to/python DEVICE=cpu ./run_step3501_300K_10ps.sh
DEVICE=cuda STEPS=100000 PYTHON=/path/to/python ./run_step3501_300K_10ps.sh
```

Outputs go to:

`runs/run_001_step3501_300K_10ps/`

Important output files:

- `mlmd.log`
- `mlmd.traj`
- `mlmd_trajectory.extxyz`
- `final_structure.extxyz`
- `run_config.json`

## Notes

This prepared run uses:

- ensemble: NVT
- thermostat: Berendsen
- target temperature: 300 K
- initial velocity temperature: 300 K
- timestep: 1 fs
- default length: 10 ps, `10000` steps

For a first check, run `STEPS=1000` before a long production trajectory.
