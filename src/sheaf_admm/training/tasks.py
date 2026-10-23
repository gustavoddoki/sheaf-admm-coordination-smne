"""Task hooks: decomposition, loss, and evaluation per benchmark.

A :class:`Task` turns a raw batch into three pieces (``prepare``):

* ``fwd``    — model inputs (all arrays): ``patches``, ``edge_indices``, and
  ``model_kwargs`` (graph extras the model needs: ``node_positions`` for grids,
  ``map_u``/``map_v``/``cell_ids`` for Sudoku). Safe to pass through ``jit``.
* ``targets`` — label arrays for the loss / metrics.
* ``aux``    — non-jittable extras (agent centers, image size) used only for the
  eager eval-time reassembly.

``loss(logits_window, targets)`` scores the windowed per-agent logits;
``evaluate(logits_final, targets, aux)`` reassembles the global prediction and
returns task metrics. Adding a benchmark = adding a Task.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax

from sheaf_admm.data import views as V

PATH_TOKEN = 5  # maze: label token marking a path cell


def _ce(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels)


def _windowed(loss_one, logits_window):
    """Average a per-iterate loss over the ADMM loss window (axis 0)."""
    return jnp.mean(jax.vmap(loss_one)(logits_window))


def sudoku_all_different_loss(logits):
    """Soft penalty requiring each Sudoku unit to contain every digit once.

    ``logits`` ends in ``[cells, classes]`` and may have arbitrary leading
    dimensions (agents and batches). Class zero is the empty token. Summing
    digit probabilities over the cells of a row/column/box should produce one
    unit of mass for every digit.
    """
    digit_probabilities = jax.nn.softmax(logits, axis=-1)[..., 1:]
    digit_counts = jnp.sum(digit_probabilities, axis=-2)
    return jnp.mean(jnp.square(digit_counts - 1.0))


def sudoku_constraint_metrics(pred, inputs, grid_size: int, *, use_boxes: bool = True):
    """Return label-independent Sudoku or Latin-square feasibility metrics.

    ``valid`` accepts any completion satisfying the clues and Sudoku rules,
    which matters when a puzzle admits more than the canonical labelled
    solution. Unit violation rates are fractions of invalid rows, columns, or
    boxes; clue and symbol violation rates are fractions of affected cells.
    Every returned value is averaged per puzzle so evaluation can weight batches
    by their number of examples.
    """
    n = int(grid_size)
    if n <= 0:
        raise ValueError(f"grid_size must be positive, got {n}")
    box = math.isqrt(n)
    if use_boxes and box * box != n:
        raise ValueError(f"grid_size must be a perfect square, got {n}")
    expected_shape = (pred.shape[0], n, n)
    if pred.shape != expected_shape or inputs.shape != expected_shape:
        raise ValueError(
            f"pred and inputs must both have shape [B,{n},{n}]; "
            f"got pred={pred.shape}, inputs={inputs.shape}"
        )

    symbols = jnp.arange(1, n + 1, dtype=pred.dtype)
    row_valid = jnp.all(jnp.sort(pred, axis=-1) == symbols, axis=-1)
    col_valid = jnp.all(jnp.sort(jnp.swapaxes(pred, -1, -2), axis=-1) == symbols, axis=-1)
    if use_boxes:
        boxes = pred.reshape(pred.shape[0], box, box, box, box)
        boxes = jnp.transpose(boxes, (0, 1, 3, 2, 4)).reshape(pred.shape[0], n, n)
        box_valid = jnp.all(jnp.sort(boxes, axis=-1) == symbols, axis=-1)
        all_units = jnp.concatenate([~row_valid, ~col_valid, ~box_valid], axis=-1)
        all_boxes_valid = jnp.all(box_valid, axis=-1)
    else:
        box_valid = jnp.ones((pred.shape[0], 0), dtype=jnp.bool_)
        all_units = jnp.concatenate([~row_valid, ~col_valid], axis=-1)
        all_boxes_valid = jnp.ones((pred.shape[0],), dtype=jnp.bool_)

    symbol_valid = (pred >= 1) & (pred <= n)
    clue_mask = inputs > 0
    clue_wrong = clue_mask & (pred != inputs)
    clue_count = jnp.sum(clue_mask, axis=(1, 2))
    clue_violation = jnp.sum(clue_wrong, axis=(1, 2)) / jnp.maximum(clue_count, 1)
    rule_valid = (
        jnp.all(symbol_valid, axis=(1, 2))
        & jnp.all(row_valid, axis=-1)
        & jnp.all(col_valid, axis=-1)
        & all_boxes_valid
    )
    clue_valid = jnp.all(~clue_wrong, axis=(1, 2))

    return {
        "valid": float(jnp.mean(rule_valid & clue_valid)),
        "rule_valid": float(jnp.mean(rule_valid)),
        "row_violation_rate": float(jnp.mean(~row_valid)),
        "column_violation_rate": float(jnp.mean(~col_valid)),
        "box_violation_rate": float(jnp.mean(~box_valid)) if use_boxes else 0.0,
        "unit_violation_rate": float(jnp.mean(all_units)),
        "clue_violation_rate": float(jnp.mean(clue_violation)),
        "symbol_violation_rate": float(jnp.mean(~symbol_valid)),
    }


def _grid_graph(image_hw, stride, patch_size, connectivity):
    centers = V.grid_agent_centers(image_hw, stride=stride, patch_size=patch_size)
    edge_indices = jnp.asarray(V.build_grid_edge_indices(centers, stride, connectivity))
    return centers, edge_indices, V.node_positions(centers)


class MazeTask:
    name = "maze"

    def __init__(self, *, patch_size=3, stride=2, connectivity=8, num_classes=6):
        self.patch_size, self.stride, self.connectivity = patch_size, stride, connectivity
        self.num_classes = num_classes

    def prepare(self, batch):
        inputs = jnp.asarray(batch["inputs"])
        labels = jnp.asarray(batch["labels"])
        B = inputs.shape[0]
        H, W = V.infer_image_hw(inputs.shape[1], batch.get("height"), batch.get("width"))
        inp_img, lab_img = inputs.reshape(B, H, W), labels.reshape(B, H, W)
        centers, edges, npos = _grid_graph((H, W), self.stride, self.patch_size, self.connectivity)
        patches = V.prepare_maze_patches(inp_img, centers, self.patch_size, self.num_classes)
        label_patches, masks = V.get_agent_label_patches_jax(lab_img, centers, self.patch_size)
        fwd = {"patches": patches, "edge_indices": edges, "model_kwargs": {"node_positions": npos}}
        targets = {"label_patches": label_patches, "masks": masks, "labels_img": lab_img}
        aux = {"centers": centers, "image_hw": (H, W)}
        return fwd, targets, aux

    def loss(self, logits_window, targets):
        lp, masks = targets["label_patches"], targets["masks"]

        def one(logits):
            ce = _ce(logits, lp) * masks
            return jnp.sum(ce) / jnp.maximum(jnp.sum(masks), 1.0)

        return _windowed(one, logits_window)

    def evaluate(self, logits_final, targets, aux):
        recon = jnp.asarray(
            V.reassemble_logits(
                logits_final, aux["centers"], aux["image_hw"], self.num_classes, mode="mean"
            )
        )
        pred = jnp.argmax(recon, axis=-1)
        labels = targets["labels_img"]
        solved = jnp.all((pred == PATH_TOKEN) == (labels == PATH_TOKEN), axis=(1, 2))
        return {"solved": float(jnp.mean(solved)), "cell_acc": float(jnp.mean(pred == labels))}


class MNISTTask:
    name = "mnist"

    def __init__(self, *, patch_size=3, stride=3, connectivity=8, num_classes=10):
        self.patch_size, self.stride, self.connectivity, self.num_classes = (
            patch_size,
            stride,
            connectivity,
            num_classes,
        )

    def prepare(self, batch):
        images = jnp.asarray(batch["images"])
        labels = jnp.asarray(batch["labels"])
        H, W = images.shape[1], images.shape[2]
        centers, edges, npos = _grid_graph((H, W), self.stride, self.patch_size, self.connectivity)
        patches = V.patchify_batch_jax(images, centers, self.patch_size)
        fwd = {"patches": patches, "edge_indices": edges, "model_kwargs": {"node_positions": npos}}
        return fwd, {"labels": labels}, {}

    def loss(self, logits_window, targets):
        labels = targets["labels"]

        def one(logits):
            return jnp.mean(_ce(logits, jnp.broadcast_to(labels, logits.shape[:-1])))

        return _windowed(one, logits_window)

    def loss_graph(self, logits, targets):  # MPNN graph readout [B,C]
        return jnp.mean(_ce(logits, targets["labels"]))

    def evaluate(self, logits_final, targets, aux):
        agg = jnp.mean(jax.nn.softmax(logits_final, axis=-1), axis=0)  # [B,C]
        return {"acc": float(jnp.mean(jnp.argmax(agg, -1) == targets["labels"]))}

    def evaluate_graph(self, logits, targets, aux):
        return {"acc": float(jnp.mean(jnp.argmax(logits, -1) == targets["labels"]))}


class SudokuTask:
    name = "sudoku"

    def __init__(
        self,
        *,
        grid_size: int = 9,
        num_classes: int | None = None,
        symbol_capacity: int | None = None,
        constraint_weight: float = 0.0,
        use_boxes: bool = True,
    ):
        self.grid_size = int(grid_size)
        self.use_boxes = bool(use_boxes)
        self.constraint_weight = float(constraint_weight)
        if self.constraint_weight < 0:
            raise ValueError("constraint_weight must be non-negative")
        # The constraint graph depends only on the board size and cover, so build it
        # once instead of re-running the Python edge construction on every batch.
        self._cell_ids = V.build_sudoku_cell_indices(self.grid_size, use_boxes=self.use_boxes)
        self._graph = V.get_cached_sudoku_graph(self.grid_size, self.use_boxes)
        expected_classes = self.grid_size + 1  # zero is the empty-cell token
        self.num_classes = expected_classes if num_classes is None else int(num_classes)
        if self.num_classes != expected_classes:
            raise ValueError(
                f"Sudoku {self.grid_size}x{self.grid_size} requires num_classes="
                f"{expected_classes}, got {self.num_classes}"
            )
        self.symbol_capacity = self.grid_size if symbol_capacity is None else int(symbol_capacity)
        if self.symbol_capacity < self.grid_size:
            raise ValueError(
                f"symbol_capacity={self.symbol_capacity} cannot represent "
                f"a {self.grid_size}x{self.grid_size} Sudoku"
            )

    def prepare(self, batch):
        inputs = jnp.asarray(batch["inputs"])
        labels = jnp.asarray(batch["labels"])
        B = inputs.shape[0]
        expected_cells = self.grid_size * self.grid_size
        if inputs.shape[1] != expected_cells or labels.shape[1] != expected_cells:
            raise ValueError(
                f"Sudoku {self.grid_size}x{self.grid_size} expects flattened sequences of "
                f"length {expected_cells}; got inputs={inputs.shape}, labels={labels.shape}"
            )
        inp = inputs.reshape(B, self.grid_size, self.grid_size)
        lab = labels.reshape(B, self.grid_size, self.grid_size)
        edges, map_u, map_v = self._graph
        cell_ids = self._cell_ids
        patches = jnp.transpose(
            V.sudoku_slice_batch_jax(
                jax.nn.one_hot(inp, self.symbol_capacity + 1),
                use_boxes=self.use_boxes,
            ),
            (1, 0, 2, 3),
        )
        slice_labels = jnp.transpose(
            V.sudoku_slice_batch_jax(lab[..., None], use_boxes=self.use_boxes)[..., 0],
            (1, 0, 2),
        ).astype(jnp.int32)
        fwd = {
            "patches": patches,
            "edge_indices": edges,
            "model_kwargs": {"map_u": map_u, "map_v": map_v, "cell_ids": cell_ids},
        }
        targets = {
            "slice_labels": slice_labels,
            "labels_grid": lab.astype(jnp.int32),
            "inputs_grid": inp.astype(jnp.int32),
        }
        return fwd, targets, {}

    def loss(self, logits_window, targets):
        labels = targets["slice_labels"]

        def one(logits):
            cross_entropy = jnp.mean(_ce(logits, labels))
            if self.constraint_weight == 0.0:
                return cross_entropy
            return cross_entropy + self.constraint_weight * sudoku_all_different_loss(logits)

        return _windowed(one, logits_window)

    def evaluate(self, logits_final, targets, aux):
        recon = V.reassemble_sudoku_logits(
            jnp.transpose(logits_final, (1, 0, 2, 3)), use_boxes=self.use_boxes
        )
        pred = jnp.argmax(recon, axis=-1)
        labels = targets["labels_grid"]
        empty = targets["inputs_grid"] == 0
        metrics = {
            "cell_acc": float(jnp.mean(pred == labels)),
            "solved": float(jnp.mean(jnp.all(pred == labels, axis=(1, 2)))),
            "completion": float(jnp.sum((pred == labels) * empty) / jnp.maximum(jnp.sum(empty), 1)),
        }
        metrics.update(
            sudoku_constraint_metrics(
                pred,
                targets["inputs_grid"],
                self.grid_size,
                use_boxes=self.use_boxes,
            )
        )
        return metrics


def make_task(name: str, **kwargs):
    tasks = {"maze": MazeTask, "mnist": MNISTTask, "sudoku": SudokuTask}
    if name not in tasks:
        raise KeyError(f"unknown task {name!r}; available: {sorted(tasks)}")
    return tasks[name](**kwargs)
