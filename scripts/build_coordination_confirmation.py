"""Build or resume the frozen P5 holdout without touching the exploratory dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

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


def seed_solution_exclusion_cache(output_dir: Path) -> bool:
    """Reuse the audited P1 index when its source inventory is unchanged."""
    p1_dir = ROOT / "datasets/sudoku_coordination_pilot_20261007"
    old_inventory_path = p1_dir / "previous_solution_keys_inventory.json"
    old_cache_path = p1_dir / "previous_solution_keys.npz"
    if not old_inventory_path.is_file() or not old_cache_path.is_file():
        return False

    # The P1 cache deliberately excluded P1 itself. Verify its recorded source
    # inventory against the current datasets before reusing any cached keys.
    paths = [
        path for path in sorted((ROOT / "datasets").rglob("all__labels.npy"))
        if p1_dir not in path.parents and output_dir not in path.parents
    ]
    current_inventory = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]
    old_inventory = json.loads(old_inventory_path.read_text(encoding="utf-8"))
    if old_inventory != current_inventory:
        return False

    with np.load(old_cache_path, allow_pickle=False) as cached:
        keys = {bytes(value) for value in cached["keys"]}
    family_dir = p1_dir / "families"
    family_paths = sorted(family_dir.glob("family_*.json"))
    if len(family_paths) != 64:
        return False
    for path in family_paths:
        family = json.loads(path.read_text(encoding="utf-8"))
        solution = np.asarray(family["solution"], dtype=np.uint8).reshape(9, 9)
        keys.add(np.ascontiguousarray(solution).tobytes())

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "previous_solution_keys.npz"
    cache_tmp = cache_path.with_suffix(".npz.tmp")
    with cache_tmp.open("wb") as stream:
        np.savez_compressed(stream, keys=np.asarray(list(keys), dtype="S81"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(cache_tmp, cache_path)
    builder.atomic_json(output_dir / "previous_solution_keys_inventory.json", [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted((ROOT / "datasets").rglob("all__labels.npy"))
        if output_dir not in path.parents
    ])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument(
        "--out-dir",
        default="datasets/sudoku_coordination_confirmation_20261007",
        help="output directory relative to the repository root; must be under datasets/",
    )
    args = parser.parse_args()
    if not 1 <= args.families <= 64:
        parser.error("families must be in 1..64 for the frozen P5 protocol")

    output_dir = (ROOT / args.out_dir).resolve()
    datasets_dir = (ROOT / "datasets").resolve()
    if not output_dir.is_relative_to(datasets_dir) or output_dir == datasets_dir:
        parser.error("--out-dir must name a new child directory under datasets/")

    # Parameterize the audited P1 generator in memory. Its default destination
    # and constants remain unchanged for existing exploratory workflows.
    builder.OUT = output_dir
    builder.FAMILY_DIR = output_dir / "families"
    builder.BASE_COUNT = args.families
    builder.SEED = args.seed
    output_dir.mkdir(parents=True, exist_ok=True)
    reused_index = seed_solution_exclusion_cache(output_dir)

    families = builder.load_or_generate_families(args.families)
    payload = builder.build_dataset(families)
    payload.update({
        "status": "confirmation_holdout_generated",
        "phase": "P5_confirmatory_holdout",
        "protocol": "experiments/sheaf_admm_reasoning/P5_PROTOCOLO_CONFIRMATORIO_2026-10-07.md",
        "protocol_sha256": sha256(
            ROOT / "experiments/sheaf_admm_reasoning/P5_PROTOCOLO_CONFIRMATORIO_2026-10-07.md"
        ),
        "generator_script_sha256": sha256(Path(__file__)),
        "reused_audited_solution_index": reused_index,
    })
    builder.atomic_json(output_dir / "paired_clue_manifest.json", payload)
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json" and not path.name.endswith(
            (".sha256", ".tmp")
        ):
            files[str(path.relative_to(ROOT))] = sha256(path)
    builder.atomic_json(output_dir / "artifact_hashes.json", files)
    print(json.dumps({key: value for key, value in payload.items() if key != "families"}, indent=2))
    print(f"Distinct confirmation families: {len(families)}; dataset rows: {payload['rows']}")


if __name__ == "__main__":
    main()
