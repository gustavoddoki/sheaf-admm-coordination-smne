"""Restartable training checkpoints.

The inference checkpoint written by :mod:`scripts.train` intentionally stays
small.  This module stores the additional optimizer and random-number state
needed to continue a training run without silently changing its trajectory.
"""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .optim import TrainState

SCHEMA_VERSION = 1
_RUNTIME_TRAINING_KEYS = {
    "checkpoint_dir",
    "checkpoint_interval",
    "epochs",
    "resume_from",
    "stop_after_epoch",
}


def checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the trajectory-defining part of a resolved training config."""
    normalized = copy.deepcopy(config)
    training = normalized.get("training")
    if isinstance(training, dict):
        for key in _RUNTIME_TRAINING_KEYS:
            training.pop(key, None)
    # Logging metadata does not affect optimization and may change on resume.
    normalized.pop("wandb", None)
    return normalized


def _host_tree(tree: Any) -> Any:
    return jax.device_get(tree)


def _device_tree(tree: Any) -> Any:
    def put(value: Any) -> Any:
        if isinstance(value, (np.ndarray, np.generic)):
            return jax.device_put(value)
        return value

    return jax.tree_util.tree_map(put, tree)


def _atomic_pickle(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def save_inference_checkpoint(
    path: str | Path,
    *,
    state: TrainState,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Atomically save model parameters for evaluation or deployment."""
    _atomic_pickle(
        path,
        {
            "params": _host_tree(state.params),
            "ema_params": _host_tree(state.ema_params),
            "config": copy.deepcopy(config),
            "metadata": copy.deepcopy(metadata or {}),
        },
    )


def save_training_checkpoint(
    path: str | Path,
    *,
    state: TrainState,
    rng: jax.Array,
    rng_np: np.random.Generator,
    next_epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    best: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Atomically save everything required to resume at an epoch boundary."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": {
            "params": _host_tree(state.params),
            "opt_state": _host_tree(state.opt_state),
            "step": _host_tree(state.step),
            "ema_params": _host_tree(state.ema_params),
        },
        "rng": _host_tree(rng),
        "rng_np_state": copy.deepcopy(rng_np.bit_generator.state),
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "history": copy.deepcopy(history),
        "best": copy.deepcopy(best),
        "config": copy.deepcopy(config),
    }
    _atomic_pickle(path, payload)


def load_training_checkpoint(
    path: str | Path,
    *,
    state: TrainState,
    config: dict[str, Any],
) -> tuple[TrainState, jax.Array, np.random.Generator, dict[str, Any]]:
    """Restore a checkpoint into a freshly initialized compatible state."""
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported training checkpoint schema {version!r}")
    if checkpoint_config(payload["config"]) != checkpoint_config(config):
        raise ValueError(
            "resume config differs from the checkpoint's trajectory-defining config"
        )

    saved_state = payload["state"]
    restored = state.replace(
        params=_device_tree(saved_state["params"]),
        opt_state=_device_tree(saved_state["opt_state"]),
        step=_device_tree(saved_state["step"]),
        ema_params=_device_tree(saved_state["ema_params"]),
    )
    rng = _device_tree(payload["rng"])
    rng_np = np.random.default_rng()
    rng_np.bit_generator.state = payload["rng_np_state"]
    metadata = {
        "next_epoch": int(payload["next_epoch"]),
        "global_step": int(payload["global_step"]),
        "history": payload["history"],
        "best": payload["best"],
    }
    return restored, rng, rng_np, metadata
