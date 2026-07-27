# MgSi44 CHGNet ML-MD From DFT Midframes

This folder prepares CHGNet ML-MD runs from equilibrated 300 K r2SCAN DFT-MD frames.

## Inputs

DFT source dataset:

`../../ml_test_dataset_r2scan_MgSi44/r2scan_300K/MgSi44_r2scan_300K_2ps_all.extxyz`

Checkpoint:

`/inspire/hdd/global_user/luomingxiang-240108540155/teacher/chgnet_distill_pretrain/finetune_default_graph_balanced_from_e35_e10_ew3_fw1p5_lr5e5/best.pth.tar`

Selected start frames:

- `starts/MgSi44_frame_501.extxyz`
  - global frame 501
  - about 0.501 ps
  - energy `-246.02026143 eV`
- `starts/MgSi44_frame_1001.extxyz`
  - global frame 1001
  - about 1.001 ps
  - energy `-247.77450214 eV`
  - default start for the prepared run
- `starts/MgSi44_frame_1501.extxyz`
  - global frame 1501
  - about 1.501 ps
  - energy `-248.47290031 eV`

## Environment

The Python environment must import all of:

- `torch`
- `ase`
- `pymatgen`
- `chgnet`

Check the environment first:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes
PYTHON=/path/to/python ./check_env.sh
```

The scripts add the local CHGNet source tree to `PYTHONPATH` by default:

`/inspire/hdd/global_user/luomingxiang-240108540155/chgnet`

## Run Default 300 K NVT ML-MD

Short 1 ps check:

```bash
cd /inspire/hdd/global_user/luomingxiang-240108540155/luyouqi/md_MgSi_44atom_vasp/model_md/chgnet_r2scan_300K_from_dft_midframes
STEPS=1000 LOGINTERVAL=5 PYTHON=/path/to/python DEVICE=cuda ./run_frame1001_300K_10ps.sh
```

Default 10 ps run:

```bash
PYTHON=/path/to/python DEVICE=cuda ./run_frame1001_300K_10ps.sh
```

Outputs go to:

`runs/run_001_frame1001_300K_10ps/`

This prepared run uses NVT, Berendsen thermostat, 300 K, 1 fs timestep, and default 10000 steps.
