"""Structured local-state projections for generalized Sudoku agents."""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Short Sinkhorn loops (the internal projection uses 4-12 rounds) stay unrolled,
# which keeps their graph and speed unchanged. Longer loops (the 160-round readout)
# are rolled with this many rounds per trip, so tracing and compilation no longer
# grow with ``num_iters``.
SINKHORN_MAX_UNROLL = 16


def repeat_sinkhorn_rounds(body, num_iters: int, value: jnp.ndarray) -> jnp.ndarray:
    """Apply ``body`` ``num_iters`` times, unrolling only short loops."""
    if num_iters <= SINKHORN_MAX_UNROLL:
        for _ in range(num_iters):
            value = body(value)
        return value
    return jax.lax.fori_loop(
        0, num_iters, lambda _, current: body(current), value, unroll=SINKHORN_MAX_UNROLL
    )


def _validate_layout(x: jnp.ndarray, clue_tokens: jnp.ndarray, grid_size: int) -> int:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if x.shape[-1] % grid_size != 0:
        raise ValueError(f"state width {x.shape[-1]} is not divisible by grid_size={grid_size}")
    cell_dim = x.shape[-1] // grid_size
    if cell_dim < grid_size:
        raise ValueError(
            "each cell block must reserve at least grid_size assignment coordinates; "
            f"got cell_dim={cell_dim}, grid_size={grid_size}"
        )
    expected_prefix = (*x.shape[:-1], grid_size)
    if clue_tokens.shape[:-1] != expected_prefix or clue_tokens.shape[-1] < grid_size + 1:
        raise ValueError(
            "clue_tokens must have shape [*x.shape[:-1], grid_size, C] with "
            f"C >= grid_size + 1; got {clue_tokens.shape}"
        )
    return cell_dim


def clue_lock_project_sudoku_state(
    x: jnp.ndarray,
    clue_tokens: jnp.ndarray,
    *,
    grid_size: int,
) -> jnp.ndarray:
    """Fix observed cells in each local assignment state, leaving free cells untouched.

    This is the clue-preservation control for the Sinkhorn projector. It does
    not normalize rows/columns or suppress a clue's symbol in other cells.
    """
    cell_dim = _validate_layout(x, clue_tokens, grid_size)
    cells = x.reshape((*x.shape[:-1], grid_size, cell_dim))
    token_ids = jnp.argmax(clue_tokens, axis=-1)
    is_clue = token_ids > 0
    clue_symbols = jax.nn.one_hot(
        jnp.clip(token_ids - 1, 0, grid_size - 1), grid_size, dtype=x.dtype
    )
    assignment = jnp.where(is_clue[..., None], clue_symbols, cells[..., :grid_size])
    projected_cells = jnp.concatenate([assignment, cells[..., grid_size:]], axis=-1)
    return projected_cells.reshape(x.shape)


def sinkhorn_project_sudoku_state(
    x: jnp.ndarray,
    clue_tokens: jnp.ndarray,
    *,
    grid_size: int,
    num_iters: int = 8,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Project the assignment subspace of each local Sudoku unit.

    Each of the ``grid_size`` cell blocks reserves its first ``grid_size``
    coordinates for a cell-by-symbol assignment matrix. Sinkhorn normalization
    makes the unfixed part approximately doubly stochastic. Non-empty clue rows
    are fixed to their observed symbol and the corresponding symbol columns are
    removed from the free subproblem. Remaining coordinates are copied exactly.

    The input units must contain non-conflicting clues. This function is an
    inexact differentiable projection; it does not discretize free assignments.
    """
    if num_iters < 1:
        raise ValueError("num_iters must be at least 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    cell_dim = _validate_layout(x, clue_tokens, grid_size)

    cells = x.reshape((*x.shape[:-1], grid_size, cell_dim))
    scores = cells[..., :grid_size] / temperature

    token_ids = jnp.argmax(clue_tokens, axis=-1)
    is_clue = token_ids > 0
    clue_symbols = jax.nn.one_hot(
        jnp.clip(token_ids - 1, 0, grid_size - 1), grid_size, dtype=x.dtype
    )
    fixed = clue_symbols * is_clue[..., None]
    used_symbols = jnp.any(fixed > 0, axis=-2)
    free_pairs = (~is_clue)[..., :, None] & (~used_symbols)[..., None, :]

    masked_value = jnp.asarray(-1.0e9, dtype=x.dtype)

    def normalize(log_assignment):
        row_lse = jax.nn.logsumexp(log_assignment, axis=-1, keepdims=True)
        row_lse = jnp.where(is_clue[..., :, None], 0.0, row_lse)
        log_assignment = jnp.where(free_pairs, log_assignment - row_lse, masked_value)

        col_lse = jax.nn.logsumexp(log_assignment, axis=-2, keepdims=True)
        col_lse = jnp.where(used_symbols[..., None, :], 0.0, col_lse)
        return jnp.where(free_pairs, log_assignment - col_lse, masked_value)

    log_assignment = repeat_sinkhorn_rounds(
        normalize, num_iters, jnp.where(free_pairs, scores, masked_value)
    )

    assignment = fixed + jnp.where(free_pairs, jnp.exp(log_assignment), 0.0)
    projected_cells = jnp.concatenate([assignment, cells[..., grid_size:]], axis=-1)
    return projected_cells.reshape(x.shape)


def sudoku_assignment_logits_from_state(
    x: jnp.ndarray,
    *,
    grid_size: int,
    epsilon: float = 1.0e-8,
    positive_transform: str = "clamp",
) -> jnp.ndarray:
    """Read assignment coordinates as logits with a configurable positive map.

    ``clamp`` preserves the historical behavior. ``softplus`` is an initial
    smooth alternative, but its epsilon floor can erase differences among
    strongly negative scores. ``log_softplus`` computes the log mass directly
    with a stable negative tail and keeps those differences trainable.
    """
    if x.shape[-1] % grid_size != 0:
        raise ValueError(f"state width {x.shape[-1]} is not divisible by grid_size={grid_size}")
    cell_dim = x.shape[-1] // grid_size
    if cell_dim < grid_size:
        raise ValueError(f"cell_dim={cell_dim} cannot store {grid_size} assignment coordinates")
    cells = x.reshape((*x.shape[:-1], grid_size, cell_dim))
    scores = cells[..., :grid_size]
    if positive_transform == "clamp":
        digit_logits = jnp.log(jnp.maximum(scores, epsilon))
    elif positive_transform == "softplus":
        digit_logits = jnp.log(jax.nn.softplus(20.0 * scores) / 20.0 + epsilon)
    elif positive_transform == "log_softplus":
        scaled = 20.0 * scores
        stable_tail = jnp.log(jax.nn.softplus(jnp.maximum(scaled, -20.0)))
        digit_logits = jnp.where(scaled < -20.0, scaled, stable_tail) - jnp.log(20.0)
    else:
        raise ValueError(
            f"unknown assignment positive_transform={positive_transform!r} "
            "(clamp|softplus|log_softplus)"
        )
    empty = jnp.full((*digit_logits.shape[:-1], 1), -30.0, dtype=x.dtype)
    return jnp.concatenate([empty, digit_logits], axis=-1)
