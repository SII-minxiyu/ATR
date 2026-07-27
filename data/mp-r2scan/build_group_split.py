#!/usr/bin/env python3
"""Build group-based train/val/test split from normalized JSONL.

Grouping key defaults to mp_id to reduce structural leakage across splits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_groups(jsonl_path: Path, group_key: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            key = row.get(group_key)
            if key is None:
                # Fallback for group key inside meta.
                key = row.get("meta", {}).get(group_key, sample_id)
            groups[str(key)].append(sample_id)
    return groups


def split_groups(
    group_keys: list[str], train_ratio: float, val_ratio: float, seed: int
) -> tuple[set[str], set[str], set[str]]:
    shuffled = list(group_keys)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_groups = set(shuffled[:n_train])
    val_groups = set(shuffled[n_train : n_train + n_val])
    test_groups = set(shuffled[n_train + n_val :])

    return train_groups, val_groups, test_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Normalized JSONL path")
    parser.add_argument("--out-dir", required=True, type=Path, help="Split output directory")
    parser.add_argument("--group-key", default="mp_id", help="Grouping key for leak-safe split")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("Invalid ratios: require 0 < train_ratio, 0 <= val_ratio, train+val < 1")

    groups = load_groups(args.input, args.group_key)
    group_keys = list(groups.keys())
    if not group_keys:
        raise SystemExit("No samples loaded from input JSONL")

    train_groups, val_groups, test_groups = split_groups(
        group_keys,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_ids = {"train": [], "val": [], "test": []}
    for key, sample_ids in groups.items():
        if key in train_groups:
            split_ids["train"].extend(sample_ids)
        elif key in val_groups:
            split_ids["val"].extend(sample_ids)
        elif key in test_groups:
            split_ids["test"].extend(sample_ids)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_json = {
        "group_key": args.group_key,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "counts": {k: len(v) for k, v in split_ids.items()},
        "ids": split_ids,
    }

    with (args.out_dir / "split.json").open("w", encoding="utf-8") as fp:
        json.dump(split_json, fp, ensure_ascii=False, indent=2)

    for split_name, ids in split_ids.items():
        with (args.out_dir / f"{split_name}_ids.txt").open("w", encoding="utf-8") as fp:
            fp.write("\n".join(ids))
            fp.write("\n")

    print(json.dumps({"out_dir": str(args.out_dir), "counts": split_json["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
