#!/usr/bin/env python3
"""Split a normalized JSONL file into train/val/test JSONL files by split.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_id_sets(split_json_path: Path) -> dict[str, set[str]]:
    with split_json_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    ids = payload.get("ids", {})
    return {
        "train": set(ids.get("train", [])),
        "val": set(ids.get("val", [])),
        "test": set(ids.get("test", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Normalized JSONL input")
    parser.add_argument("--split-json", required=True, type=Path, help="split.json path")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    id_sets = load_id_sets(args.split_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_files = {
        name: (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ("train", "val", "test")
    }
    counts = {"train": 0, "val": 0, "test": 0, "unmatched": 0}

    try:
        with args.input.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = row["sample_id"]

                matched = False
                for split_name in ("train", "val", "test"):
                    if sample_id in id_sets[split_name]:
                        output_files[split_name].write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[split_name] += 1
                        matched = True
                        break

                if not matched:
                    counts["unmatched"] += 1
    finally:
        for fp in output_files.values():
            fp.close()

    print(json.dumps({"out_dir": str(args.out_dir), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
