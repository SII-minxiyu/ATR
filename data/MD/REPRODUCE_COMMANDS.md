# Reproduce 300 K AIMD / MLMD

This file uses relative bundle paths so the package can be moved to another
machine. After unpacking the zip, define the bundle root first:

```bash
cd MD
export BUNDLE_ROOT="$PWD"
```

`MANIFEST.csv` is a relative-path index for the unpacked bundle.

## README Locations

- System notes: `$BUNDLE_ROOT/<System>/README_and_notes/README.md`
- DFT-AIMD notes: `$BUNDLE_ROOT/<System>/aimd_300K/README_reference_dft_md.md`
- MLMD notes: `$BUNDLE_ROOT/<System>/mlmd_300K/README_model_md.md`
- DFT extxyz notes: `$BUNDLE_ROOT/<System>/dft_extxyz_for_eval/**/README.md`

## Reproduce DFT-AIMD

Each formal AIMD run directory contains its own `run.sh`. The script has
site-specific defaults for `ONEAPI` and `VASP`, so override them on a new
machine.

Example for MgSi44:

```bash
cd "$BUNDLE_ROOT/MgSi44/aimd_300K/run_002_r2scan_nvt_300K_2ps"
ONEAPI=/path/to/oneapi/setvars.sh \
VASP=/path/to/vasp_std \
NP=16 \
./run.sh
```

Example for InSe48:

```bash
cd "$BUNDLE_ROOT/InSe48/aimd_300K/run_005_r2scan_nvt_300K_1ps"
ONEAPI=/path/to/oneapi/setvars.sh \
VASP=/path/to/vasp_std \
NP=16 \
./run.sh
```

Example for HgInSn48:

```bash
cd "$BUNDLE_ROOT/HgInSn48/aimd_300K/run_002_r2scan_nvt_300K_2ps"
ONEAPI=/path/to/oneapi/setvars.sh \
VASP=/path/to/vasp_std \
NP=16 \
./run.sh
```

Notes:

- `POTCAR` is intentionally not included. Regenerate it before rerunning VASP
  with the same element order as the local `POSCAR`.
- Use the corresponding `run.sh` in each formal 300 K AIMD directory.
- Existing AIMD output files are already included for inspection, so rerunning
  is only needed for reproduction.

## Reproduce MLMD

The bundle contains the needed start structures and scripts under each
`<System>/mlmd_300K/` directory. On a new machine, provide:

```bash
export PYTHON=/path/to/python
export CHGNET_ROOT=/path/to/chgnet_source_or_empty_if_installed
export CHECKPOINT_DIR=/path/to/checkpoints
```

Expected checkpoint filenames in the examples below:

- `$CHECKPOINT_DIR/atr_best.pth.tar`
- `$CHECKPOINT_DIR/baseline_random_e20.pth.tar`

Official CHGNet r2SCAN uses `--checkpoint r2scan` if the installed CHGNet
package provides that pretrained model.

### InSe48 ATR/best

```bash
cd "$BUNDLE_ROOT/InSe48/mlmd_300K"
PYTHONPATH="${CHGNET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"$PYTHON" scripts/run_chgnet_mlmd.py \
  --initial starts/InSe48_frame_1001.extxyz \
  --checkpoint "$CHECKPOINT_DIR/atr_best.pth.tar" \
  --out-dir runs/reproduce_InSe48_ATR_300K_100ps \
  --steps 100000 \
  --temperature 300 \
  --starting-temperature 300 \
  --timestep 1.0 \
  --loginterval 10 \
  --device cuda \
  --thermostat Berendsen
```

### MgSi44 ATR/best

```bash
cd "$BUNDLE_ROOT/MgSi44/mlmd_300K"
PYTHONPATH="${CHGNET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"$PYTHON" scripts/run_chgnet_mlmd.py \
  --initial starts/MgSi44_frame_1001.extxyz \
  --checkpoint "$CHECKPOINT_DIR/atr_best.pth.tar" \
  --out-dir runs/reproduce_MgSi44_ATR_300K_100ps \
  --steps 100000 \
  --temperature 300 \
  --starting-temperature 300 \
  --timestep 1.0 \
  --loginterval 10 \
  --device cuda \
  --thermostat Berendsen
```

### HgInSn48 ATR/best

```bash
cd "$BUNDLE_ROOT/HgInSn48/mlmd_300K"
PYTHONPATH="${CHGNET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"$PYTHON" scripts/run_chgnet_mlmd.py \
  --initial starts/HgInSn48_source_step_2501.extxyz \
  --checkpoint "$CHECKPOINT_DIR/atr_best.pth.tar" \
  --out-dir runs/reproduce_HgInSn48_ATR_300K_100ps \
  --steps 100000 \
  --temperature 300 \
  --starting-temperature 300 \
  --timestep 1.0 \
  --loginterval 10 \
  --device cuda \
  --thermostat Berendsen
```

For baseline comparisons, reuse the same command and change:

```bash
--checkpoint "$CHECKPOINT_DIR/baseline_random_e20.pth.tar"
--out-dir runs/reproduce_<System>_baseline_300K_100ps
```

For official CHGNet r2SCAN comparisons, use:

```bash
--checkpoint r2scan
--out-dir runs/reproduce_<System>_official_r2scan_300K_100ps
```

The copied `run_config.json` files under `mlmd_300K/runs/` record the exact
settings used for the original runs.
