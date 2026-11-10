"""Build/resume nested-clue Sudoku families for the coordination pilot."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sheaf_admm.data.build_sudoku import SudokuConfig, _build_split_from_puzzles  # noqa: E402
from sheaf_admm.data.sudoku_generation import (  # noqa: E402
    count_intercalates,
    generate_unique_puzzles,
    puzzle_digest,
)
from sheaf_admm.data.sudoku_solver import count_solutions  # noqa: E402

OUT = ROOT / "datasets/sudoku_coordination_pilot_20261007"
FAMILY_DIR = OUT / "families"
BASE_COUNT = 64
BASE_CLUES = 32
LEVELS = [32, 40, 48, 56]
TARGET_COUNT = 20
SEED = 20261007


def atomic_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    digest = hashlib.sha256(data).hexdigest()
    checksum = path.with_suffix(path.suffix + ".sha256")
    checksum_tmp = checksum.with_suffix(checksum.suffix + ".tmp")
    with checksum_tmp.open("w", encoding="ascii") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(checksum_tmp, checksum)
    return digest


def read_checkpoint(path: Path) -> dict:
    data = path.read_bytes()
    expected = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"checkpoint checksum mismatch: {path}")
    return json.loads(data)


def unit_indices(pos: int) -> tuple[int, int, int]:
    row, col = divmod(pos, 9)
    return row, 9 + col, 18 + (row // 3) * 3 + col // 3


def unit_clue_counts(puzzle: np.ndarray) -> list[int]:
    grid = np.asarray(puzzle).reshape(9, 9) > 0
    rows = grid.sum(axis=1)
    cols = grid.sum(axis=0)
    boxes = np.array(
        [grid[r : r + 3, c : c + 3].sum() for r in (0, 3, 6) for c in (0, 3, 6)]
    )
    return [int(value) for value in np.concatenate([rows, cols, boxes])]


def clue_order(base: np.ndarray, policy: str, seed: int, common: list[int]) -> list[int]:
    rng = np.random.default_rng(seed)
    counts = np.zeros(27, dtype=np.int32)
    for pos in np.flatnonzero(base.reshape(-1)):
        counts[list(unit_indices(int(pos)))] += 1
    reserved = set(common)
    choices = set(int(x) for x in np.flatnonzero(base.reshape(-1) == 0)) - reserved
    order: list[int] = []
    additions_needed = max(LEVELS) - BASE_CLUES
    while choices and len(order) < additions_needed:
        scored = []
        for pos in choices:
            units = list(unit_indices(pos))
            gain = int(np.sum(2 * counts[units] + 1))
            scored.append((gain, pos))
        best = min(score for score, _ in scored) if policy == "balanced" else max(
            score for score, _ in scored
        )
        tied = [pos for score, pos in scored if score == best]
        pos = int(rng.choice(tied))
        order.append(pos)
        counts[list(unit_indices(pos))] += 1
        choices.remove(pos)
    if len(order) < additions_needed:
        raise RuntimeError("not enough non-reserved cells for the clue schedule")
    return order


def solution_key(solution: np.ndarray) -> bytes:
    """Canonical byte key for a completed 9x9 grid (digits are not remapped)."""
    return np.ascontiguousarray(solution, dtype=np.uint8).reshape(9, 9).tobytes()


def previously_used_solution_keys() -> set[bytes]:
    """Collect exact completed-grid hashes from existing 9x9 dataset labels."""
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        path
        for path in sorted((ROOT / "datasets").rglob("all__labels.npy"))
        if OUT not in path.parents
    ]
    inventory = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]
    cache_path = OUT / "previous_solution_keys.npz"
    inventory_path = OUT / "previous_solution_keys_inventory.json"
    if cache_path.exists() and inventory_path.exists():
        cached_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if cached_inventory == inventory:
            with np.load(cache_path, allow_pickle=False) as cached:
                return {bytes(value) for value in cached["keys"]}

    used: set[bytes] = set()
    for path in paths:
        labels = np.load(path, mmap_mode="r")
        if labels.ndim != 2 or labels.shape[1] != 81:
            continue
        # Most evaluation datasets repeat labels across clue masks or
        # transformations. Deduplicate in NumPy before crossing into Python.
        for row in np.unique(labels, axis=0):
            grid = np.asarray(row, dtype=np.uint8).reshape(9, 9)
            if np.all((grid >= 1) & (grid <= 9)):
                used.add(solution_key(grid))
    cache_tmp = cache_path.with_suffix(".npz.tmp")
    with cache_tmp.open("wb") as stream:
        np.savez_compressed(stream, keys=np.asarray(list(used), dtype="S81"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(cache_tmp, cache_path)
    atomic_json(inventory_path, inventory)
    return used


def generate_family(family_id: int, forbidden_solutions: set[bytes]) -> dict:
    for attempt in range(4096):
        seed = SEED + family_id * 100_003 + attempt
        generated = generate_unique_puzzles(
            grid_size=9,
            clue_count=BASE_CLUES,
            count=1,
            seed=seed,
            solution_generator="backtracking",
            clue_strategy="greedy",
            require_noncanonical_intercalates=True,
            solution_filter=lambda solution: solution_key(solution) not in forbidden_solutions,
        )[0]
        solution = generated.solution.astype(np.uint8)
        puzzle = generated.puzzle.astype(np.uint8)
        digest = puzzle_digest(solution)
        if solution_key(solution) in forbidden_solutions:
            continue
        uniqueness = count_solutions(puzzle, max_solutions=2)
        if uniqueness.solution_count != 1:
            raise RuntimeError(f"base family {family_id} is not unique")
        return {
            "family_id": family_id,
            "seed": seed,
            "solution": solution.reshape(-1).tolist(),
            "base_puzzle": puzzle.reshape(-1).tolist(),
            "solution_digest": digest,
            "base_puzzle_digest": puzzle_digest(puzzle),
            "base_clues": int(np.count_nonzero(puzzle)),
            "uniqueness_count": int(uniqueness.solution_count),
            "uniqueness_nodes": int(uniqueness.nodes),
            "uniqueness_backtracks": int(uniqueness.backtracks),
            "intercalates": int(count_intercalates(solution)),
        }
    raise RuntimeError(f"could not generate family {family_id} after 4096 deterministic seeds")


def load_or_generate_families(count: int) -> list[dict]:
    FAMILY_DIR.mkdir(parents=True, exist_ok=True)
    forbidden = previously_used_solution_keys()
    families = []
    for family_id in range(count):
        checkpoint = FAMILY_DIR / f"family_{family_id:04d}.json"
        if checkpoint.exists():
            family = read_checkpoint(checkpoint)
            if (
                family["family_id"] != family_id
                or solution_key(np.asarray(family["solution"], dtype=np.uint8)) in forbidden
                or family["solution_digest"] in {item["solution_digest"] for item in families}
            ):
                raise RuntimeError(f"invalid or duplicate saved family: {checkpoint}")
            families.append(family)
            forbidden.add(solution_key(np.asarray(family["solution"], dtype=np.uint8)))
            continue
        family = generate_family(family_id, forbidden)
        if family["solution_digest"] in {item["solution_digest"] for item in families}:
            raise RuntimeError("duplicate completed grid in pilot families")
        digest = atomic_json(checkpoint, family)
        family["checkpoint_sha256"] = digest
        families.append(family)
        forbidden.add(solution_key(np.asarray(family["solution"], dtype=np.uint8)))
        print(json.dumps({
            "checkpointed_family": family_id,
            "completed": len(families),
            "total": count,
            "solution_digest": family["solution_digest"],
        }), flush=True)
    return families


def build_dataset(families: list[dict]) -> dict:
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    manifest_families = []
    for family in families:
        family_id = int(family["family_id"])
        base = np.asarray(family["base_puzzle"], dtype=np.uint8).reshape(9, 9)
        solution = np.asarray(family["solution"], dtype=np.uint8).reshape(9, 9)
        if int(np.count_nonzero(base)) != BASE_CLUES:
            raise RuntimeError(f"family {family_id} has wrong base clue count")
        common = sorted(int(x) for x in np.random.default_rng(SEED + 97 * family_id + 13).choice(
            np.flatnonzero(base.reshape(-1) == 0), size=TARGET_COUNT, replace=False
        ))
        schedules = {
            policy: clue_order(
                base,
                policy,
                SEED + 97 * family_id + (0 if policy == "balanced" else 1),
                common,
            )
            for policy in ("balanced", "concentrated")
        }
        for policy in ("balanced", "concentrated"):
            order = schedules[policy]
            for level in LEVELS:
                flat = base.reshape(-1).copy()
                added = order[: level - BASE_CLUES]
                flat[added] = solution.reshape(-1)[added]
                puzzle = flat.reshape(9, 9)
                if np.any(puzzle.reshape(-1)[common] != 0):
                    raise RuntimeError("common evaluation cells were exposed in a clue variant")
                if count_solutions(puzzle, max_solutions=2).solution_count != 1:
                    raise RuntimeError(f"family {family_id} variant {policy}/{level} is not unique")
                rows.append((flat, solution.reshape(-1).copy()))
                counts = unit_clue_counts(puzzle)
                manifest_families.append({
                    "family_id": family_id,
                    "policy": policy,
                    "clues": level,
                    "base_clues": BASE_CLUES,
                    "added_cells": added,
                    "common_evaluation_cells": common,
                    "solution_digest": family["solution_digest"],
                    "row_index": len(rows) - 1,
                    "solution_count_audited": 1,
                    "unit_clue_counts_rows_cols_boxes": counts,
                    "unit_clue_sum_squares": int(sum(count * count for count in counts)),
                })
    cfg = SudokuConfig(
        output_dir=OUT,
        train_size=0,
        test_size=len(rows),
        seed=SEED,
        train_augment=False,
        test_augment=False,
    )
    _build_split_from_puzzles("test_hard", rows, cfg, apply_augmentation=False)
    (OUT / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    payload = {
        "status": "pilot_generated",
        "seed": SEED,
        "base_families": len(families),
        "base_clues": BASE_CLUES,
        "clue_levels": LEVELS,
        "policies": ["balanced", "concentrated"],
        "rows": len(rows),
        "common_cells_per_family": TARGET_COUNT,
        "uniqueness": "each base and every nested variant was counted to exactly one solution",
        "solution_generator": "randomized backtracking with noncanonical-intercalate filter",
        "excluded_existing_exact_solution_digests": True,
        "previous_unique_solution_count": len(previously_used_solution_keys()),
        "families": manifest_families,
    }
    atomic_json(OUT / "paired_clue_manifest.json", payload)
    files = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json" and not path.name.endswith(
            (".sha256", ".tmp")
        ):
            files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(OUT / "artifact_hashes.json", files)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        type=int,
        default=BASE_COUNT,
        help=f"number of solution families to prepare (default: {BASE_COUNT})",
    )
    args = parser.parse_args()
    if args.families < 1 or args.families > BASE_COUNT:
        parser.error(f"--families must be in 1..{BASE_COUNT}")
    OUT.mkdir(parents=True, exist_ok=True)
    families = load_or_generate_families(args.families)
    payload = build_dataset(families)
    print(json.dumps({key: value for key, value in payload.items() if key != "families"}, indent=2))
    print(f"Distinct families: {len(families)}; dataset rows: {payload['rows']}")


if __name__ == "__main__":
    main()
