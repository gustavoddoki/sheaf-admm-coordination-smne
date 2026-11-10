"""Unique partial Latin-square generation for transfer experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .latin_solver import count_latin_solutions
from .sudoku_generation import puzzle_digest


@dataclass(frozen=True)
class GeneratedLatinSquare:
    """A unique partial Latin square and its completion."""

    puzzle: np.ndarray
    solution: np.ndarray
    clue_count: int
    uniqueness_nodes: int
    uniqueness_backtracks: int


def _intercalates(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    size = int(grid.shape[0])
    candidates = []
    for row_a in range(size):
        for row_b in range(row_a + 1, size):
            for col_a in range(size):
                for col_b in range(col_a + 1, size):
                    if (
                        grid[row_a, col_a] == grid[row_b, col_b]
                        and grid[row_a, col_b] == grid[row_b, col_a]
                        and grid[row_a, col_a] != grid[row_a, col_b]
                    ):
                        candidates.append((row_a, row_b, col_a, col_b))
    return candidates


def generate_complete_latin_square(
    rng: np.random.Generator, grid_size: int, *, trades: int = 8
) -> np.ndarray:
    """Generate a randomized Latin square, with optional intercalate trades."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    if trades < 0:
        raise ValueError("trades must be non-negative")
    base = np.fromfunction(
        lambda row, col: (row + col) % grid_size + 1,
        (grid_size, grid_size),
        dtype=int,
    ).astype(np.uint8)
    digits = rng.permutation(np.arange(1, grid_size + 1, dtype=np.uint8))
    grid = digits[base - 1]
    grid = grid[rng.permutation(grid_size)][:, rng.permutation(grid_size)]
    if bool(rng.integers(2)):
        grid = grid.T.copy()
    for _ in range(trades):
        candidates = _intercalates(grid)
        if not candidates:
            break
        row_a, row_b, col_a, col_b = candidates[int(rng.integers(len(candidates)))]
        first = grid[row_a, col_a]
        second = grid[row_a, col_b]
        grid[row_a, col_a] = second
        grid[row_a, col_b] = first
        grid[row_b, col_a] = first
        grid[row_b, col_b] = second
    return grid.astype(np.uint8)


def sample_latin_clues_uniquely(
    solution: np.ndarray,
    clue_count: int,
    rng: np.random.Generator,
) -> GeneratedLatinSquare | None:
    """Sample one exact-clue mask and retain it only when uniquely completable."""
    solution = np.asarray(solution, dtype=np.uint8)
    if solution.ndim != 2 or solution.shape[0] != solution.shape[1]:
        raise ValueError("solution must be a square board")
    if not 1 <= clue_count <= solution.size:
        raise ValueError(f"clue_count must lie in 1..{solution.size}")
    puzzle = solution.copy()
    puzzle.reshape(-1)[rng.permutation(solution.size)[: solution.size - clue_count]] = 0
    result = count_latin_solutions(puzzle, max_solutions=2)
    if result.solution_count != 1:
        return None
    return GeneratedLatinSquare(
        puzzle,
        solution.copy(),
        clue_count,
        result.nodes,
        result.backtracks,
    )


def generate_unique_latin_puzzles(
    *,
    grid_size: int,
    clue_count: int,
    count: int,
    seed: int,
    trades: int = 8,
    seen_digests: set[str] | None = None,
    max_attempts_per_puzzle: int = 512,
    on_accept: Callable[[int, int], None] | None = None,
) -> list[GeneratedLatinSquare]:
    """Generate unique-solution, deduplicated Latin-square puzzles."""
    if count < 1:
        raise ValueError("count must be at least one")
    rng = np.random.default_rng(seed)
    seen = seen_digests if seen_digests is not None else set()
    generated = []
    attempts = 0
    max_attempts = count * max_attempts_per_puzzle
    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        solution = generate_complete_latin_square(rng, grid_size, trades=trades)
        candidate = sample_latin_clues_uniquely(solution, clue_count, rng)
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
            f"generated {len(generated)}/{count} unique {grid_size}x{grid_size} "
            f"Latin puzzles with {clue_count} clues after {attempts} attempts"
        )
    return generated
