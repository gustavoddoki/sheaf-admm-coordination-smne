"""Constraint-aware metrics for completed Sudoku predictions."""

from __future__ import annotations

import math

import numpy as np


def _unit_deficit(values: np.ndarray, grid_size: int) -> int:
    valid = values[(values >= 1) & (values <= grid_size)]
    return grid_size - int(np.unique(valid).size)


def analyze_sudoku_predictions(
    inputs: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    use_boxes: bool = True,
) -> tuple[dict[str, float | int], list[dict[str, float | int | bool]]]:
    """Measure task errors and constraint violations per puzzle.

    A row/column/box is valid only when it contains every symbol ``1..n`` once.
    ``unit_deficit`` counts missing distinct valid symbols, so it captures both
    duplicate symbols and invalid zero/out-of-range predictions.
    """
    inputs = np.asarray(inputs, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if inputs.shape != labels.shape or inputs.shape != predictions.shape:
        raise ValueError(
            f"inputs, labels, and predictions must share shape; got "
            f"{inputs.shape}, {labels.shape}, {predictions.shape}"
        )
    if inputs.ndim != 3 or inputs.shape[1] != inputs.shape[2]:
        raise ValueError(f"expected [batch, n, n] square boards, got {inputs.shape}")
    batch_size, grid_size, _ = inputs.shape
    box_size = math.isqrt(grid_size)
    if use_boxes and box_size * box_size != grid_size:
        raise ValueError(f"grid size must be a perfect square, got {grid_size}")

    rows: list[dict[str, float | int | bool]] = []
    for puzzle_index in range(batch_size):
        puzzle_input = inputs[puzzle_index]
        label = labels[puzzle_index]
        prediction = predictions[puzzle_index]
        hidden = puzzle_input == 0
        wrong = prediction != label
        clue_mismatches = int(np.count_nonzero((~hidden) & (prediction != puzzle_input)))
        invalid_symbols = int(np.count_nonzero((prediction < 1) | (prediction > grid_size)))
        row_deficits = [
            _unit_deficit(prediction[row_index], grid_size) for row_index in range(grid_size)
        ]
        col_deficits = [
            _unit_deficit(prediction[:, col_index], grid_size) for col_index in range(grid_size)
        ]
        box_deficits = []
        if use_boxes:
            for box_row in range(box_size):
                for box_col in range(box_size):
                    box = prediction[
                        box_row * box_size : (box_row + 1) * box_size,
                        box_col * box_size : (box_col + 1) * box_size,
                    ]
                    box_deficits.append(_unit_deficit(box.reshape(-1), grid_size))

        exact_solution = bool(not np.any(wrong))
        valid_grid = bool(
            clue_mismatches == 0
            and invalid_symbols == 0
            and not any(row_deficits)
            and not any(col_deficits)
            and not any(box_deficits)
        )
        hidden_cells = int(np.count_nonzero(hidden))
        wrong_hidden_cells = int(np.count_nonzero(wrong & hidden))
        rows.append(
            {
                "puzzle_index": puzzle_index,
                "clue_count": int(prediction.size - hidden_cells),
                "hidden_cells": hidden_cells,
                "wrong_cells": int(np.count_nonzero(wrong)),
                "wrong_hidden_cells": wrong_hidden_cells,
                "hidden_accuracy": (
                    1.0 - wrong_hidden_cells / hidden_cells if hidden_cells else 1.0
                ),
                "clue_mismatches": clue_mismatches,
                "invalid_symbol_cells": invalid_symbols,
                "invalid_rows": int(np.count_nonzero(row_deficits)),
                "invalid_columns": int(np.count_nonzero(col_deficits)),
                "invalid_boxes": int(np.count_nonzero(box_deficits)),
                "row_deficit": int(sum(row_deficits)),
                "column_deficit": int(sum(col_deficits)),
                "box_deficit": int(sum(box_deficits)),
                "total_unit_deficit": int(
                    sum(row_deficits) + sum(col_deficits) + sum(box_deficits)
                ),
                "clues_consistent": clue_mismatches == 0,
                "valid_grid": valid_grid,
                "exact_solution": exact_solution,
            }
        )

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0

    wrong_hidden = np.asarray([int(row["wrong_hidden_cells"]) for row in rows])
    summary: dict[str, float | int] = {
        "puzzles": batch_size,
        "grid_size": grid_size,
        "constraint_family": "sudoku" if use_boxes else "latin",
        "cell_accuracy": float(np.mean(predictions == labels)),
        "hidden_accuracy": mean("hidden_accuracy"),
        "exact_solution_rate": mean("exact_solution"),
        "valid_grid_rate": mean("valid_grid"),
        "valid_but_not_exact_rate": float(
            np.mean([bool(row["valid_grid"]) and not bool(row["exact_solution"]) for row in rows])
        ),
        "clue_consistency_rate": mean("clues_consistent"),
        "mean_wrong_hidden_cells": mean("wrong_hidden_cells"),
        "median_wrong_hidden_cells": (float(np.median(wrong_hidden)) if wrong_hidden.size else 0.0),
        "max_wrong_hidden_cells": int(wrong_hidden.max()) if wrong_hidden.size else 0,
        "fraction_with_at_most_1_wrong_hidden": float(np.mean(wrong_hidden <= 1)),
        "fraction_with_at_most_2_wrong_hidden": float(np.mean(wrong_hidden <= 2)),
        "fraction_with_at_most_5_wrong_hidden": float(np.mean(wrong_hidden <= 5)),
        "mean_clue_mismatches": mean("clue_mismatches"),
        "mean_invalid_symbol_cells": mean("invalid_symbol_cells"),
        "mean_invalid_rows": mean("invalid_rows"),
        "mean_invalid_columns": mean("invalid_columns"),
        "mean_invalid_boxes": mean("invalid_boxes"),
        "mean_total_unit_deficit": mean("total_unit_deficit"),
    }
    return summary, rows
