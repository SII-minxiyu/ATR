"""Selective pseudo-label router for multi-teacher DFT surrogate selection."""

from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_XGBOOST = _ROOT / ".vendor" / "xgboost_only"
if _VENDOR_XGBOOST.exists():
    sys.path.insert(0, str(_VENDOR_XGBOOST))
