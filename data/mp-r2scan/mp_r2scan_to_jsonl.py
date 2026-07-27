#!/usr/bin/env python3
"""Convert MP-r2SCAN nested JSON to line-delimited JSON for training pipelines.

Input format (observed):
{
  "mp-xxxx": {
    "config-id": {
      "structure": {...},
      "uncorrected_total_energy": ...,
      "force": [...],
      "stress": [...],
      "magmom": [... or null],
      ...
    },
    ...
  },
  ...
}
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _try_load_ijson() -> Any | None:
    try:
        import ijson  # type: ignore
    except ImportError:  # pragma: no cover - runtime guard
        return None
    return ijson


def normalize_record(mp_id: str, config_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": f"{mp_id}/{config_id}",
        "mp_id": mp_id,
        "config_id": config_id,
        "structure": payload.get("structure"),
        "targets": {
            "energy": payload.get("uncorrected_total_energy"),
            "energy_per_atom": payload.get("energy_per_atom"),
            "forces": payload.get("force"),
            "stress": payload.get("stress"),
            "magmom": payload.get("magmom"),
            "bandgap": payload.get("bandgap"),
        },
        "meta": {
            "source": "MPr2SCAN",
            "raw_mp_id": payload.get("mp_id", mp_id),
        },
    }


def iter_records(input_path: Path):
    ijson = _try_load_ijson()
    if ijson is not None:
        with input_path.open("rb") as fp:
            for mp_id, mp_group in ijson.kvitems(fp, ""):
                if not isinstance(mp_group, dict):
                    continue
                for config_id, payload in mp_group.items():
                    if not isinstance(payload, dict):
                        continue
                    yield normalize_record(str(mp_id), str(config_id), payload)
        return

    # Fallback: use jq streaming when ijson is unavailable.
    jq_bin = "jq"
    jq_filter = (
        "to_entries[] as $mp | "
        "$mp.value | to_entries[] | "
        "{sample_id: ($mp.key + \"/\" + .key), "
        "mp_id: $mp.key, "
        "config_id: .key, "
        "structure: .value.structure, "
        "targets: {"
        "energy: .value.uncorrected_total_energy, "
        "energy_per_atom: .value.energy_per_atom, "
        "forces: .value.force, "
        "stress: .value.stress, "
        "magmom: .value.magmom, "
        "bandgap: .value.bandgap"
        "}, "
        "meta: {source: \"MPr2SCAN\", raw_mp_id: (.value.mp_id // $mp.key)}"
        "}"
    )

    try:
        proc = subprocess.Popen(
            [jq_bin, "-c", jq_filter, str(input_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "No streaming backend available. Install `ijson` or ensure `jq` is in PATH."
        ) from exc

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)

    stderr = ""
    if proc.stderr is not None:
        stderr = proc.stderr.read().strip()
    ret = proc.wait()
    if ret != 0:
        raise SystemExit(f"jq failed with exit code {ret}: {stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to MPr2SCAN.json")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL path (one normalized sample per line)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max samples to export")
    parser.add_argument(
        "--skip-missing-force",
        action="store_true",
        help="Skip samples without force labels",
    )
    parser.add_argument(
        "--skip-missing-energy",
        action="store_true",
        help="Skip samples without total energy labels",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as out:
        for sample in iter_records(args.input):
            targets = sample["targets"]
            if args.skip_missing_force and targets.get("forces") is None:
                skipped += 1
                continue
            if args.skip_missing_energy and targets.get("energy") is None:
                skipped += 1
                continue

            out.write(json.dumps(sample, ensure_ascii=False, default=float) + "\n")
            written += 1

            if args.limit and written >= args.limit:
                break

    print(
        json.dumps(
            {
                "output": str(args.output),
                "written": written,
                "skipped": skipped,
                "limit": args.limit,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
