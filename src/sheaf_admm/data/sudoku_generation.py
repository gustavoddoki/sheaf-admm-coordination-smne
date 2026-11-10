"""Unique, deduplicated parametric Sudoku generation utilities."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .sudoku_solver import SudokuSearchBudgetExceeded, count_solutions, solve_sudoku

SolutionGenerator = Literal["symmetry", "backtracking"]
ClueStrategy = Literal["greedy", "random_mask"]


@dataclass(frozen=True)
class GeneratedSudoku:
    """A puzzle/solution pair with oracle search telemetry."""

    puzzle: np.ndarray
    solution: np.ndarray
    clue_count: int
    uniqueness_nodes: int
    uniqueness_backtracks: int


def generate_complete_grid(rng: np.random.Generator, grid_size: int) -> np.ndarray:
    """Generate a randomized valid completed grid using Sudoku symmetries."""
    box_size = math.isqrt(grid_size)
    if grid_size <= 0 or box_size * box_size != grid_size:
        raise ValueError(f"grid_size must be a positive perfect square, got {grid_size}")
    base = np.fromfunction(
        lambda row, col: (row * box_size + row // box_size + col) % grid_size + 1,
        (grid_size, grid_size),
        dtype=int,
    )
    digits = rng.permutation(np.arange(1, grid_size + 1))
    grid = digits[base.astype(np.int64) - 1]
    band_order = rng.permutation(box_size)
    row_order = np.concatenate([band * box_size + rng.permutation(box_size) for band in band_order])
    stack_order = rng.permutation(box_size)
    col_order = np.concatenate(
        [stack * box_size + rng.permutation(box_size) for stack in stack_order]
    )
    return grid[row_order][:, col_order].astype(np.uint8)


def generate_complete_grid_backtracking(
    rng: np.random.Generator,
    grid_size: int,
    *,
    max_search_nodes: int | None = None,
    max_grid_attempts: int = 32,
) -> np.ndarray:
    """Complete randomized diagonal boxes with the Sudoku search oracle.

    Unlike :func:`generate_complete_grid`, this does not start from the fixed
    cyclic solution template. The disjoint diagonal boxes can be populated
    independently without conflicting row, column, or box constraints.
    """
    box_size = math.isqrt(grid_size)
    if grid_size <= 0 or box_size * box_size != grid_size:
        raise ValueError(f"grid_size must be a positive perfect square, got {grid_size}")
    if max_grid_attempts < 1:
        raise ValueError("max_grid_attempts must be positive")
    for _ in range(max_grid_attempts):
        partial = np.zeros((grid_size, grid_size), dtype=np.uint8)
        for box in range(box_size):
            start = box * box_size
            partial[start : start + box_size, start : start + box_size] = rng.permutation(
                np.arange(1, grid_size + 1, dtype=np.uint8)
            ).reshape(box_size, box_size)
        try:
            result = solve_sudoku(
                partial,
                strategy="mrv_lcv",
                max_solutions=1,
                max_nodes=max_search_nodes,
            )
        except SudokuSearchBudgetExceeded:
            continue
        if result.solution is None:
            raise RuntimeError("random diagonal boxes unexpectedly had no completion")
        return result.solution.astype(np.uint8)
    raise RuntimeError("no complete grid found within the search budget")


def count_intercalates(solution: np.ndarray) -> int:
    """Count 2x2 patterns ``a,b / b,a`` in a completed Latin square.

    This count is invariant to row, column, and symbol permutations and to
    transposition. A different count proves that two Sudoku solutions are not
    related by those transformations; the converse need not hold.
    """
    grid = np.asarray(solution)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError("solution must be a square board")
    size = grid.shape[0]
    total = 0
    for row_a in range(size):
        for row_b in range(row_a + 1, size):
            for col_a in range(size):
                for col_b in range(col_a + 1, size):
                    total += bool(
                        grid[row_a, col_a] == grid[row_b, col_b]
                        and grid[row_a, col_b] == grid[row_b, col_a]
                        and grid[row_a, col_a] != grid[row_a, col_b]
                    )
    return total


def puzzle_digest(puzzle: np.ndarray) -> str:
    """Stable content key used to prevent train/test puzzle overlap."""
    grid = np.asarray(puzzle)
    payload = f"{grid.shape}:{grid.dtype.str}:".encode() + grid.tobytes()
    return hashlib.sha256(payload).hexdigest()


def assign_solution_split(solution: np.ndarray, split_fractions: dict[str, float]) -> str:
    """Deterministically assign a completed grid to one split by content hash.

    Puzzle-level deduplication does not stop two splits from sharing a solution
    grid with different clue masks. Small boards make this unavoidable: the
    symmetry generator reaches only 96 distinct 4x4 grids. Routing each grid to
    exactly one split keeps splits solution-disjoint independent of generation
    order. Fractions need not sum to one; they are normalized.
    """
    if not split_fractions or any(value < 0 for value in split_fractions.values()):
        raise ValueError("split fractions must be non-empty and non-negative")
    total = float(sum(split_fractions.values()))
    if total <= 0:
        raise ValueError("split fractions must have a positive sum")
    grid = np.asarray(solution, dtype=np.uint8)
    position = int(puzzle_digest(grid)[:16], 16) / float(1 << 64)
    cumulative = 0.0
    last_positive = None
    for name, fraction in split_fractions.items():
        if fraction <= 0:
            continue
        last_positive = name
        cumulative += fraction / total
        if position < cumulative:
            return name
    assert last_positive is not None
    return last_positive


def remove_clues_uniquely(
    solution: np.ndarray,
    clue_count: int,
    rng: np.random.Generator,
    *,
    max_orders: int = 32,
) -> GeneratedSudoku | None:
    """Greedily remove clues while retaining exactly one solution.

    Several randomized removal orders are tried because a greedy pass can get
    stuck above the requested clue count.  ``None`` indicates that no order
    reached the exact target within the configured budget.
    """
    solution = np.asarray(solution, dtype=np.uint8)
    if solution.ndim != 2 or solution.shape[0] != solution.shape[1]:
        raise ValueError("solution must be a square board")
    num_cells = int(solution.size)
    if not 1 <= clue_count <= num_cells:
        raise ValueError(f"clue_count must lie in 1..{num_cells}")

    for _ in range(max_orders):
        puzzle = solution.copy()
        last_result = count_solutions(puzzle, max_solutions=2)
        for flat_index in rng.permutation(num_cells):
            if np.count_nonzero(puzzle) <= clue_count:
                break
            old_value = int(puzzle.reshape(-1)[flat_index])
            puzzle.reshape(-1)[flat_index] = 0
            candidate_result = count_solutions(puzzle, max_solutions=2)
            if candidate_result.solution_count == 1:
                last_result = candidate_result
            else:
                puzzle.reshape(-1)[flat_index] = old_value
        if np.count_nonzero(puzzle) == clue_count:
            return GeneratedSudoku(
                puzzle=puzzle,
                solution=solution.copy(),
                clue_count=clue_count,
                uniqueness_nodes=last_result.nodes,
                uniqueness_backtracks=last_result.backtracks,
            )
    return None


def sample_clues_uniquely(
    solution: np.ndarray,
    clue_count: int,
    rng: np.random.Generator,
) -> GeneratedSudoku | None:
    """Try one uniform exact-clue mask, accepting it only if unique.

    Rejection sampling is often cheaper than sequential greedy removal on
    larger boards. The returned oracle telemetry refers to the accepted mask.
    """
    solution = np.asarray(solution, dtype=np.uint8)
    if solution.ndim != 2 or solution.shape[0] != solution.shape[1]:
        raise ValueError("solution must be a square board")
    num_cells = int(solution.size)
    if not 1 <= clue_count <= num_cells:
        raise ValueError(f"clue_count must lie in 1..{num_cells}")
    puzzle = solution.copy()
    puzzle.reshape(-1)[rng.permutation(num_cells)[: num_cells - clue_count]] = 0
    result = count_solutions(puzzle, max_solutions=2)
    if result.solution_count != 1:
        return None
    return GeneratedSudoku(
        puzzle=puzzle,
        solution=solution.copy(),
        clue_count=clue_count,
        uniqueness_nodes=result.nodes,
        uniqueness_backtracks=result.backtracks,
    )


def generate_unique_puzzles(
    *,
    grid_size: int,
    clue_count: int,
    count: int,
    seed: int,
    seen_digests: set[str] | None = None,
    max_attempts_per_puzzle: int = 256,
    solution_generator: SolutionGenerator = "symmetry",
    clue_strategy: ClueStrategy = "greedy",
    completion_max_nodes: int | None = None,
    on_accept: Callable[[int, int], None] | None = None,
    require_noncanonical_intercalates: bool = False,
    solution_filter: Callable[[np.ndarray], bool] | None = None,
) -> list[GeneratedSudoku]:
    """Generate exact-clue, unique puzzles without duplicate input boards.

    ``solution_filter`` rejects completed grids before any clue removal, for
    example to keep train, validation, and test solution-disjoint (see
    :func:`assign_solution_split`). Leaving it unset reproduces older datasets.
    """
    if count < 1:
        raise ValueError("count must be at least one")
    if solution_generator not in ("symmetry", "backtracking"):
        raise ValueError(f"unknown solution_generator={solution_generator!r}")
    if clue_strategy not in ("greedy", "random_mask"):
        raise ValueError(f"unknown clue_strategy={clue_strategy!r}")
    if require_noncanonical_intercalates and solution_generator != "backtracking":
        raise ValueError("noncanonical intercalate filter requires backtracking")
    rng = np.random.default_rng(seed)
    canonical_intercalates = None
    if require_noncanonical_intercalates:
        canonical_intercalates = count_intercalates(
            generate_complete_grid(np.random.default_rng(0), grid_size)
        )
    seen = seen_digests if seen_digests is not None else set()
    generated: list[GeneratedSudoku] = []
    attempts = 0
    max_attempts = count * max_attempts_per_puzzle
    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        solution = (
            generate_complete_grid(rng, grid_size)
            if solution_generator == "symmetry"
            else generate_complete_grid_backtracking(
                rng, grid_size, max_search_nodes=completion_max_nodes
            )
        )
        if (
            canonical_intercalates is not None
            and count_intercalates(solution) == canonical_intercalates
        ):
            continue
        if solution_filter is not None and not solution_filter(solution):
            continue
        candidate = (
            remove_clues_uniquely(solution, clue_count, rng)
            if clue_strategy == "greedy"
            else sample_clues_uniquely(solution, clue_count, rng)
        )
        if candidate is None:
            continue
        digest = puzzle_digest(candidate.puzzle)
        if digest in seen:
            continue
        seen.add(digest)
        generated.append(candidate)
        if on_accept is not None:
            on_accept(len(generated), attempts)
    if len(generated) != count:
        raise RuntimeError(
            f"generated {len(generated)}/{count} unique {grid_size}x{grid_size} puzzles "
            f"with {clue_count} clues after {attempts} attempts"
        )
    return generated
