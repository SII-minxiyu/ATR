# Active rejection enables reliable generalization of universal machine-learning interatomic potentials

This repository provides the code, trained models, data interfaces, and reproducibility
workflow for the manuscript:

**"Active rejection enables reliable generalization of universal machine-learning
interatomic potentials."**

We introduce **Adaptive Multi-Teacher Routing (ATR)**, a structure-wise routing
framework that converts predictions from multiple pretrained universal machine-learning
interatomic potentials (uMLIPs) into reliable and traceable r2SCAN-level pseudo-labels.
For each candidate structure, ATR either selects a high-confidence teacher prediction
or actively rejects the structure when no teacher is sufficiently reliable.

## Highlights

- Structure-wise teacher selection with an explicit Accept/Reject decision.
- Top-5 teacher deployment using an XGBoost A/R router and lightweight risk control.
- CHGNet student pretraining, MP-r2SCAN evaluation, and finite-temperature molecular
dynamics validation.
- Export of routing confidence, selected-teacher identity, and per-structure metadata.



## Repository layout

```text
ATR/
├── data/
│   ├── subset/                         # initial unlabeled structures for ATR routing
│   ├── train.extxyz                    # ATR-selected r2SCAN-level pseudo-labeled training structures
│   ├── test.extxyz                     # ATR-selected r2SCAN-level pseudo-labeled test structures
│   ├── heldout_test.extxyz             # real DFT-r2SCAN data for final model testing
│   ├── ATR-data/                       # calibration data and ATR training inputs
│   ├── MD/                             # MLMD and DFT-AIMD validation data
│   │   ├── HgInSn48/
│   │   ├── InSe48/
│   │   └── MgSi44/
│   ├── mp-r2scan/                      # MP-r2SCAN train/val/test splits and processed datasets
│   └── teacher-distillation label/     # aligned multi-teacher predictions
│       ├── 7net-mf_0-R2SCAN/
│       ├── 7net-omni-i12-matpes_r2scan/
│       ├── 7net-omni-i12-mp_r2scan/
│       ├── 7net-omni-matpes_r2scan/
│       ├── 7net-omni-mp_r2scan/
│       ├── chgnet/
│       ├── mace-matpes-r2scan-0/
│       ├── mace-mh-1-matpes-r2scan/
│       └── pet-mad/
├── models/
│   ├── ATR_seed2026.joblib             # trained top-5 XGBoost A/R router
│   ├── ATR_baseline-model/             # ATR student and baseline checkpoints
│   │   ├── chgnet_r2scan_ATR.pth.tar
│   │   ├── baseline.pth.tar
│   │   └── without_ATR.pth.tar
│   ├── chgnet/                         # CHGNet teacher checkpoints
│   ├── mace/                           # MACE teacher checkpoints
│   ├── pet/                            # PET-MAD teacher checkpoints
│   └── sevennet/                       # SevenNet teacher checkpoints
├── outputs/                            # routing decisions, logs, metrics, and trajectories
├── selective_router/                   # ATR features, training, routing, and analysis
├── tools/
│   ├── inference/                      # teacher-inference utilities
│   ├── mp-r2scan/                      # MP-r2SCAN preprocessing and split utilities
│   ├── route_extxyz_top5.py            # route aligned ExtXYZ teacher predictions
│   ├── export_accepted_from_decisions_extxyz.py
│   ├── export_route_decisions_to_extxyz.py
│   └── extxyz_to_ase_db.py
├── run_selective_router.py             # ATR training and application CLI
├── commands_examples.sh                # additional command templates
├── requirements.txt                    # Python dependencies
└── README.md
```

## Installation

```bash
cd ATR
conda create -n atr python=3.10 -y
conda activate atr
pip install -r requirements.txt
```

For MLMD workflows, install the additional packages required by CHGNet:

```bash
pip install torch ase pymatgen chgnet lmdb numpy pandas scikit-learn
```

Reproducing the DFT-AIMD calculations requires a separate VASP installation and
locally licensed `POTCAR` files. Proprietary VASP files are not included.

## Data preparation

The initial unlabeled candidate structures used by the ATR workflow are stored under:

```text
data/subset/
```

ATR routes these structures through the teacher ensemble, retains high-confidence
predictions, and produces the following r2SCAN-level pseudo-labeled splits:

```text
data/train.extxyz
data/test.extxyz
```

These two files contain ATR-selected teacher labels derived from the unlabeled subset;
they are not direct DFT calculations.

`data/heldout_test.extxyz` contains real DFT-r2SCAN calculations and is reserved for
the final evaluation of the trained model. Final testing should report results on both:

```text
data/heldout_test.extxyz
data/mp-r2scan/test.extxyz
```

The router calibration data under `data/ATR-data/` should contain real r2SCAN labels
and aligned predictions from the candidate teacher models. Teacher predictions must
refer to the same structures using consistent structure IDs or identical ordering.

The processed MP-r2SCAN data used for student-model fine-tuning and evaluation are
organized separately:

```text
data/mp-r2scan/train.extxyz
data/mp-r2scan/val.extxyz
data/mp-r2scan/test.extxyz
```

LMDB files, when provided, are derived training caches. The authoritative data split
remains the split definition together with the processed ExtXYZ files.

## Train the ATR router

Train the A/R router from the r2SCAN calibration set and aligned teacher predictions:

```bash
python run_selective_router.py ar-maxforce \
  --root data/ATR-data \
  --output-dir outputs/retrain_ar_router \
  --seed 2026 \
  --backend xgboost \
  --disagreement-mode top5 \
  --precision-target 0.90
```

The training command writes the router bundle and evaluation summaries to:

```text
outputs/retrain_ar_router/router_bundle.joblib
outputs/retrain_ar_router/summary.json
outputs/retrain_ar_router/all_structure_decisions.csv
```

To use the retrained router as the default deployment model:

```bash
cp outputs/retrain_ar_router/router_bundle.joblib models/ATR_seed2026.joblib
```



## Apply ATR to aligned teacher predictions

The repository includes the trained router at `models/ATR_seed2026.joblib`. If the
five teacher prediction files contain the same structures in the same order, route
them directly from ExtXYZ:

```bash
python tools/route_extxyz_top5.py \
  --bundle models/ATR_seed2026.joblib \
  --teacher 7net-omni-i12-mp_r2scan=/path/to/7net-omni-i12-mp_r2scan.extxyz \
  --teacher 7net-omni-mp_r2scan=/path/to/7net-omni-mp_r2scan.extxyz \
  --teacher 7net-omni-i12-matpes_r2scan=/path/to/7net-omni-i12-matpes_r2scan.extxyz \
  --teacher 7net-omni-matpes_r2scan=/path/to/7net-omni-matpes_r2scan.extxyz \
  --teacher mace-mh-1-matpes_r2scan=/path/to/mace-mh-1-matpes-r2scan.extxyz \
  --output-csv outputs/ar_route_decisions.csv \
  --output-accepted-xyz outputs/ar_route_labeled_accept_only.extxyz \
  --postprocess-light \
  --postprocess-confidence-lt 0.45 \
  --postprocess-force-dev-max-gt 0.15
```

The accepted ExtXYZ file contains the selected teacher energy and forces together
with routing metadata in `atoms.info`.

## Molecular dynamics validation

The 300 K validation systems are organized under:

```text
data/MD/HgInSn48/
data/MD/InSe48/
data/MD/MgSi44/
```

For example, run an HgInSn48 MLMD trajectory with the ATR-pretrained CHGNet model:

```bash
python data/MD/HgInSn48/mlmd_300K/scripts/run_chgnet_mlmd.py \
  --initial data/MD/HgInSn48/mlmd_300K/starts/HgInSn48_source_step_2501.extxyz \
  --checkpoint models/ATR_baseline-model/chgnet_r2scan_ATR.pth.tar \
  --out-dir outputs/mlmd/HgInSn48/atr_300K_100ps \
  --steps 100000 \
  --temperature 300 \
  --starting-temperature 300 \
  --timestep 1.0 \
  --loginterval 10 \
  --device cuda \
  --thermostat Berendsen \
  --seed 20260514
```

Use the corresponding scripts under the InSe48 and MgSi44 directories for the other
systems. DFT-AIMD input templates, reference trajectories, and comparison scripts are
stored within each system directory.

## Key outputs

- `models/ATR_seed2026.joblib`: trained top-5 ATR A/R router.
- `outputs/ar_route_decisions.csv`: per-structure decision, confidence, and selected teacher.
- `outputs/ar_route_labeled_accept_only.extxyz`: accepted r2SCAN-level pseudo-labels.
- `models/ATR_baseline-model/chgnet_r2scan_ATR.pth.tar`: ATR-pretrained CHGNet checkpoint.
- `data/MD/<system>/mlmd_300K/runs/`: finite-temperature MLMD trajectories and logs.
- `data/MD/<system>/summary/`: accuracy and stability summaries.



## Reproducibility notes

- Keep the ATR-generated `data/train.extxyz` and `data/test.extxyz` split fixed and
  preserve their mapping to the original structures under `data/subset/`.
- Keep `data/heldout_test.extxyz` and `data/mp-r2scan/test.extxyz` strictly excluded
  from router training, student-model training, hyperparameter tuning, and model selection.
- Split by immutable material or trajectory group whenever group identifiers are available.
- The final deployment uses Top-5 teacher routing with an XGBoost A/R classifier.
- The lightweight post-processing rule rejects low-confidence structures with large
inter-teacher force disagreement.
- Preserve structure IDs, source metadata, and selected-teacher metadata so that every
accepted pseudo-label remains traceable.
- Treat LMDB files as derived caches rather than authoritative split definitions.

## License

The code is released under the license specified in `LICENSE`. Dataset and model
checkpoint licenses may differ and should follow the terms of their original sources.
