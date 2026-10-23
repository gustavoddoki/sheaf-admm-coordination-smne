"""Build disjoint P7/P8 Sudoku families with resumable per-family checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_coordination_paired_clues as builder  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=int, default=136)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "datasets/sudoku_coordination_final_20261008",
    )
    args = parser.parse_args()
    if args.families != 136:
        parser.error("the frozen split requires exactly 136 families: 8 development, 64 P7, 64 P8")

    out = args.out.resolve()
    config = {
        "families": args.families,
        "development_ids": list(range(0, 8)),
        "p7_holdout_ids": list(range(8, 72)),
        "p8_holdout_ids": list(range(72, 136)),
        "base_clues": 32,
        "nested_clue_levels": [32, 40, 48],
        "distributions": ["balanced", "concentrated"],
        "common_hidden_cells": 20,
        "seed": args.seed,
        "builder_sha256": sha256(Path(__file__)),
        "family_generator_sha256": sha256(Path(builder.__file__)),
    }
    config_path = out / "frozen_build_config.json"
    out.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"holdout output already belongs to another configuration: {out}")
    else:
        builder.atomic_json(config_path, config)

    builder.OUT = out
    builder.FAMILY_DIR = out / "families"
    builder.BASE_COUNT = args.families
    builder.BASE_CLUES = 32
    builder.LEVELS = [32, 40, 48]
    builder.SEED = args.seed
    families = builder.load_or_generate_families(args.families)
    manifest = builder.build_dataset(families)
    manifest["frozen_build_config_sha256"] = sha256(config_path)
    builder.atomic_json(out / "paired_clue_manifest.json", manifest)

    hashes = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in {"artifact_hashes.json"} and not path.name.endswith(
            (".sha256", ".tmp")
        ):
            hashes[str(path.relative_to(ROOT))] = sha256(path)
    builder.atomic_json(out / "artifact_hashes.json", hashes)
    print(json.dumps({
        "status": "complete",
        "families": len(families),
        "rows": manifest["rows"],
        "output": str(out.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "manifest_sha256": sha256(out / "paired_clue_manifest.json"),
        "artifact_count": len(hashes),
    }, indent=2))


if __name__ == "__main__":
    main()
