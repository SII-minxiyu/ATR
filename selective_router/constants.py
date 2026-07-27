from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REFERENCE_DB = "LeMat-Traj-s18000-260407-MatPESStatic-R2SCAN.db"

DEFAULT_MODEL_FILES = [
    "LeMat-Traj-s18000-260407-7net-mf_0-R2SCAN.db",
    "LeMat-Traj-s18000-260407-7net-omni-i12-matpes_r2scan.db",
    "LeMat-Traj-s18000-260407-7net-omni-i12-mp_r2scan.db",
    "LeMat-Traj-s18000-260407-7net-omni-matpes_r2scan.db",
    "LeMat-Traj-s18000-260407-7net-omni-mp_r2scan.db",
    "LeMat-Traj-s18000-260407-MPALOE_MatPES_combined.db",
    "LeMat-Traj-s18000-260407-chgnet-r2scan.db",
    "LeMat-Traj-s18000-260407-mace-matpes-r2scan-0.db",
    "LeMat-Traj-s18000-260407-mace-mh-1-matpes_r2scan.db",
]

DEFAULT_TEACHER_PRIORITY = [
    "7net-omni-i12-mp_r2scan",
    "7net-omni-i12-matpes_r2scan",
    "7net-omni-mp_r2scan",
    "mace-mh-1-matpes_r2scan",
    "7net-omni-matpes_r2scan",
    "mace-matpes-r2scan-0",
    "MPALOE_MatPES_combined",
    "7net-mf_0-R2SCAN",
    "chgnet-r2scan",
]


TRANSITION_METALS = {
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    57,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    89,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
}
ALKALI_METALS = {3, 11, 19, 37, 55, 87}
ALKALINE_EARTHS = {4, 12, 20, 38, 56, 88}
HALOGENS = {9, 17, 35, 53, 85, 117}
CHALCOGENS = {8, 16, 34, 52, 84, 116}
PNICTOGENS = {7, 15, 33, 51, 83, 115}
LANTHANIDES = set(range(57, 72))
ACTINIDES = set(range(89, 104))
NOBLE_GASES = {2, 10, 18, 36, 54, 86, 118}
METALLOIDS = {5, 14, 32, 33, 51, 52, 84}


@dataclass(frozen=True)
class TierSpec:
    name: str
    energy_pa_max: float
    force_mean_max: float | None = None
    force_p90_max: float | None = None


DEFAULT_TIERS = {
    "A": TierSpec("A", energy_pa_max=0.05, force_mean_max=0.15, force_p90_max=0.30),
    "B": TierSpec("B", energy_pa_max=0.10, force_mean_max=0.25, force_p90_max=0.50),
    "C": TierSpec("C", energy_pa_max=0.10, force_mean_max=None, force_p90_max=None),
}

DEFAULT_PRECISION_TARGETS = {
    "A": 0.90,
    "B": 0.85,
    "C": 0.90,
}


def normalize_model_name(path: str | Path) -> str:
    stem = Path(path).stem
    prefix = "LeMat-Traj-s18000-260407-"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem
