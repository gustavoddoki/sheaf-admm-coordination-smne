"""Deterministic parametric Sudoku solvers used as dataset oracles.

The solvers support ``n x n`` boards when ``n`` is a perfect square.  Empty
cells are encoded as zero and symbols as ``1..n``.  Search can stop after a
fixed number of solutions, which makes ``max_solutions=2`` an inexpensive
uniqueness test.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

SearchStrategy = Literal["backtracking", "mrv_lcv"]


class SudokuSearchBudgetExceeded(RuntimeError):
    """Raised only for explicitly budgeted searches; never implies uniqueness."""


@dataclass(frozen=True)
class SudokuSearchResult:
    """Result and search-effort telemetry from a Sudoku solve.

    ``nodes`` counts every attempted assignment, including forced cells with a
    single candidate. Without branching it therefore equals the number of empty
    cells and mostly re-encodes the clue count. ``guesses`` counts only
    assignments tried at branching cells (two or more candidates), and
    ``backtracks`` counts failed subtrees; use those as search-effort proxies.
    """

    solution: np.ndarray | None
    solution_count: int
    nodes: int
    backtracks: int
    elapsed_seconds: float
    guesses: int = 0


def _board_geometry(board: np.ndarray) -> tuple[np.ndarray, int, int]:
    grid = np.asarray(board, dtype=np.int64)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError(f"Sudoku board must be square, got shape {grid.shape}")
    size = int(grid.shape[0])
    box_size = math.isqrt(size)
    if size <= 0 or box_size * box_size != size:
        raise ValueError(f"Sudoku size must be a positive perfect square, got {size}")
    if np.any(grid < 0) or np.any(grid > size):
        raise ValueError(f"Sudoku symbols must lie in 0..{size}")
    return grid.copy(), size, box_size


def _peer_indices(size: int, box_size: int) -> tuple[tuple[int, ...], ...]:
    peers: list[tuple[int, ...]] = []
    for row in range(size):
        for col in range(size):
            indices = {row * size + other for other in range(size) if other != col}
            indices.update(other * size + col for other in range(size) if other != row)
            box_row = (row // box_size) * box_size
            box_col = (col // box_size) * box_size
            indices.update(
                other_row * size + other_col
                for other_row in range(box_row, box_row + box_size)
                for other_col in range(box_col, box_col + box_size)
                if other_row != row or other_col != col
            )
            peers.append(tuple(sorted(indices)))
    return tuple(peers)


def solve_sudoku(
    board: np.ndarray,
    *,
    strategy: SearchStrategy = "mrv_lcv",
    max_solutions: int = 1,
    max_nodes: int | None = None,
) -> SudokuSearchResult:
    """Solve a Sudoku and return at most ``max_solutions`` solutions.

    ``backtracking`` selects the first empty cell and tries symbols in numeric
    order. ``mrv_lcv`` selects the cell with the minimum remaining values and
    tries the least-constraining symbol first.  ``nodes`` counts attempted
    assignments, including forced ones; ``guesses`` counts attempts at cells
    with at least two candidates. Prefer ``guesses`` or ``backtracks`` over
    ``nodes`` (or sub-millisecond wall-clock timings) as a difficulty measure.
    """
    if strategy not in ("backtracking", "mrv_lcv"):
        raise ValueError(f"unknown search strategy: {strategy}")
    if max_solutions < 1:
        raise ValueError("max_solutions must be at least one")
    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be positive when provided")

    started = time.perf_counter()
    grid, size, box_size = _board_geometry(board)
    full_mask = (1 << size) - 1
    row_masks = [0] * size
    col_masks = [0] * size
    box_masks = [0] * size
    empty_indices: list[int] = []

    def box_index(row: int, col: int) -> int:
        return (row // box_size) * box_size + col // box_size

    consistent = True
    for row in range(size):
        for col in range(size):
            value = int(grid[row, col])
            if value == 0:
                empty_indices.append(row * size + col)
                continue
            bit = 1 << (value - 1)
            box = box_index(row, col)
            if row_masks[row] & bit or col_masks[col] & bit or box_masks[box] & bit:
                consistent = False
                break
            row_masks[row] |= bit
            col_masks[col] |= bit
            box_masks[box] |= bit
        if not consistent:
            break

    if not consistent:
        return SudokuSearchResult(
            solution=None,
            solution_count=0,
            nodes=0,
            backtracks=0,
            elapsed_seconds=time.perf_counter() - started,
        )

    peers = _peer_indices(size, box_size) if strategy == "mrv_lcv" else ()
    solution_count = 0
    first_solution: np.ndarray | None = None
    nodes = 0
    backtracks = 0
    guesses = 0

    def candidates(flat_index: int) -> int:
        row, col = divmod(flat_index, size)
        used = row_masks[row] | col_masks[col] | box_masks[box_index(row, col)]
        return full_mask & ~used

    def select_cell() -> tuple[int, int] | None:
        if strategy == "backtracking":
            for flat_index in empty_indices:
                row, col = divmod(flat_index, size)
                if grid[row, col] == 0:
                    return flat_index, candidates(flat_index)
            return None

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
        if best_index < 0:
            return None
        return best_index, best_candidates

    def ordered_bits(flat_index: int, candidate_mask: int) -> list[int]:
        bits = []
        remaining = candidate_mask
        while remaining:
            bit = remaining & -remaining
            bits.append(bit)
            remaining ^= bit
        if strategy == "backtracking" or len(bits) <= 1:
            return bits

        def lcv_score(bit: int) -> int:
            score = 0
            for peer_index in peers[flat_index]:
                peer_row, peer_col = divmod(peer_index, size)
                if grid[peer_row, peer_col] == 0 and candidates(peer_index) & bit:
                    score += 1
            return score

        return sorted(bits, key=lambda bit: (lcv_score(bit), bit))

    def search() -> None:
        nonlocal solution_count, first_solution, nodes, backtracks, guesses
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
        box = box_index(row, col)
        count_before = solution_count
        branching = candidate_mask.bit_count() > 1
        for bit in ordered_bits(flat_index, candidate_mask):
            value = bit.bit_length()
            if max_nodes is not None and nodes >= max_nodes:
                raise SudokuSearchBudgetExceeded(f"search exceeded {max_nodes} nodes")
            nodes += 1
            guesses += int(branching)
            grid[row, col] = value
            row_masks[row] |= bit
            col_masks[col] |= bit
            box_masks[box] |= bit
            search()
            grid[row, col] = 0
            row_masks[row] ^= bit
            col_masks[col] ^= bit
            box_masks[box] ^= bit
            if solution_count >= max_solutions:
                break
        if solution_count == count_before:
            backtracks += 1

    search()
    return SudokuSearchResult(
        solution=first_solution,
        solution_count=solution_count,
        nodes=nodes,
        backtracks=backtracks,
        elapsed_seconds=time.perf_counter() - started,
        guesses=guesses,
    )


def count_solutions(board: np.ndarray, *, max_solutions: int = 2) -> SudokuSearchResult:
    """Count solutions up to a cap using MRV+LCV."""
    return solve_sudoku(board, strategy="mrv_lcv", max_solutions=max_solutions)
