# InSe48 CHGNet ML-MD From DFT Midframes

This folder prepares CHGNet ML-MD runs from equilibrated 300 K r2SCAN DFT-MD frames.

## Inputs

DFT source dataset:

`../../ml_test_dataset_r2scan_InSe48/r2scan_300K/InSe48_r2scan_300K_2ps_all.extxyz`

Checkpoint:

`/inspire/hdd/global_user/luomingxiang-240108540155/teacher/chgnet_distill_pretrain/finetune_default_graph_balanced_from_e35_e10_ew3_fw1p5_lr5e5/best.pth.tar`

Selected start frames:

- `starts/InSe48_frame_501.extxyz`
  - global frame 501
  - about 0.501 ps
  - energy `-903.00861100 eV`
- `starts/InSe48_frame_1001.extxyz`
  - global frame 1001
  - about 1.001 ps
  - energy `-902.28827615 eV`
  - default start for the prepared run
- `starts/InSe48_frame_1501.extxyz`
  - global frame 1501
  - about 1.501 ps
  - energy `-902.45039944 eV`

The InSe 2 ps dataset was combined from two 1 ps DFT runs, so `source_step` restarts in the second half. Use `global_frame` for ML-MD start selection.

## Environment

The Python environment must import all of:

- `torch`
- `ase`
- `pymatgen`
- `chgnet`

Check the environment first:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes
PYTHON=/path/to/python ./check_env.sh
```

The scripts add the local CHGNet source tree to `PYTHONPATH` by default:

`/inspire/hdd/global_user/luomingxiang-240108540155/chgnet`

## Run Default 300 K NVT ML-MD

Short 1 ps check:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_InSe_48atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes
STEPS=1000 LOGINTERVAL=5 PYTHON=/path/to/python DEVICE=cuda ./run_frame1001_300K_10ps.sh
```

Default 10 ps run:

```bash
PYTHON=/path/to/python DEVICE=cuda ./run_frame1001_300K_10ps.sh
```

Outputs go to:

`runs/run_001_frame1001_300K_10ps/`

This prepared run uses NVT, Berendsen thermostat, 300 K, 1 fs timestep, and default 10000 steps.
