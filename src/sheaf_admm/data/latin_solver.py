"""Deterministic Latin-square solver and uniqueness oracle."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


class LatinSearchBudgetExceeded(RuntimeError):
    """Raised when an explicitly bounded search exceeds its node budget."""


@dataclass(frozen=True)
class LatinSearchResult:
    """Solution count and effort telemetry for a Latin-square search."""

    solution: np.ndarray | None
    solution_count: int
    nodes: int
    backtracks: int
    elapsed_seconds: float


def _peer_indices(size: int) -> tuple[tuple[int, ...], ...]:
    peers = []
    for row in range(size):
        for col in range(size):
            indices = {row * size + other for other in range(size) if other != col}
            indices.update(other * size + col for other in range(size) if other != row)
            peers.append(tuple(sorted(indices)))
    return tuple(peers)


def solve_latin_square(
    board: np.ndarray,
    *,
    max_solutions: int = 1,
    max_nodes: int | None = None,
) -> LatinSearchResult:
    """Solve a partial Latin square with MRV+LCV search."""
    grid = np.asarray(board, dtype=np.int64).copy()
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1] or grid.shape[0] == 0:
        raise ValueError(f"Latin board must be non-empty and square, got {grid.shape}")
    size = int(grid.shape[0])
    if np.any(grid < 0) or np.any(grid > size):
        raise ValueError(f"Latin symbols must lie in 0..{size}")
    if max_solutions < 1:
        raise ValueError("max_solutions must be at least one")
    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be positive when provided")

    started = time.perf_counter()
    full_mask = (1 << size) - 1
    row_masks = [0] * size
    col_masks = [0] * size
    empty_indices: list[int] = []
    for row in range(size):
        for col in range(size):
            value = int(grid[row, col])
            if value == 0:
                empty_indices.append(row * size + col)
                continue
            bit = 1 << (value - 1)
            if row_masks[row] & bit or col_masks[col] & bit:
                return LatinSearchResult(None, 0, 0, 0, time.perf_counter() - started)
            row_masks[row] |= bit
            col_masks[col] |= bit

    peers = _peer_indices(size)
    solution_count = 0
    first_solution: np.ndarray | None = None
    nodes = 0
    backtracks = 0

    def candidates(flat_index: int) -> int:
        row, col = divmod(flat_index, size)
        return full_mask & ~(row_masks[row] | col_masks[col])

    def select_cell() -> tuple[int, int] | None:
        best_index = -1
        best_candidates = 0
        best_count = size + 1
        for flat_index in empty_indices:
            row, col = divmod(flat_index, size)
            if grid[row, col] != 0:
                continue
            candidate_mask = candidates(flat_index)
            candidate_count = candidate_mask.bit_count()
            if candidate_count < best_count:
                best_index = flat_index
                best_candidates = candidate_mask
                best_count = candidate_count
                if candidate_count <= 1:
                    break
        return None if best_index < 0 else (best_index, best_candidates)

    def ordered_bits(flat_index: int, candidate_mask: int) -> list[int]:
        bits = []
        remaining = candidate_mask
        while remaining:
            bit = remaining & -remaining
            bits.append(bit)
            remaining ^= bit

        def score(bit: int) -> int:
            return sum(
                grid[row, col] == 0 and bool(candidates(peer) & bit)
                for peer in peers[flat_index]
                for row, col in [divmod(peer, size)]
            )

        return sorted(bits, key=lambda bit: (score(bit), bit))

    def search() -> None:
        nonlocal solution_count, first_solution, nodes, backtracks
        if solution_count >= max_solutions:
            return
        selected = select_cell()
        if selected is None:
            solution_count += 1
            if first_solution is None:
                first_solution = grid.copy()
            return
        flat_index, candidate_mask = selected
        if candidate_mask == 0:
            backtracks += 1
            return
        row, col = divmod(flat_index, size)
        count_before = solution_count
        for bit in ordered_bits(flat_index, candidate_mask):
            if max_nodes is not None and nodes >= max_nodes:
                raise LatinSearchBudgetExceeded(f"search exceeded {max_nodes} nodes")
            nodes += 1
            grid[row, col] = bit.bit_length()
            row_masks[row] |= bit
            col_masks[col] |= bit
            search()
            grid[row, col] = 0
            row_masks[row] ^= bit
            col_masks[col] ^= bit
            if solution_count >= max_solutions:
                break
        if solution_count == count_before:
            backtracks += 1

    search()
    return LatinSearchResult(
        first_solution,
        solution_count,
        nodes,
        backtracks,
        time.perf_counter() - started,
    )


def count_latin_solutions(board: np.ndarray, *, max_solutions: int = 2) -> LatinSearchResult:
    """Count Latin-square completions up to a cap."""
    return solve_latin_square(board, max_solutions=max_solutions)
