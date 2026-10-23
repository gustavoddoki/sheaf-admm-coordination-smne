"""The unrolled ADMM loop — the coordination core.

Each agent carries three states: the local proposal ``x``, the consensus iterate
``z``, and the scaled dual accumulator ``y`` (``u = lambda / rho``). One ADMM
iteration is::

    z_prev = z
    x       = prox_f(z - y; rho)                       # local x-update (x-solver)
    x_relax = alpha * x + (1 - alpha) * z_prev          # optional over-relaxation
    z       = consensus(x_relax + y; geometry, rho)     # sheaf z-update (z-solver)
    y       = y + (x_relax - z)                          # dual ascent

We unroll ``K`` iterations and backpropagate through the whole trajectory. The
loss is computed from the final ``loss_window`` x-iterates (decoded and
averaged), so the loop returns those. Memory options:

* full unroll (``grad_window=None``): gradients flow through all ``K`` steps.
* truncated BPTT (``grad_window=g``): the first ``K-g`` steps run detached
  (``stop_gradient``); gradients flow only through the last ``g``. Memory is
  ``O(g)`` instead of ``O(K)``.

Either way only the last ``loss_window`` x-iterates are materialized.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from sheaf_admm.geometry.base import SheafGeometry
from sheaf_admm.solvers.base import (
    XSolver,
    XSolverParams,
    ZSolver,
    ZSolverParams,
)

LEARNED_GATE_FEATURES = 5


@struct.dataclass
class ADMMState:
    """Per-agent ADMM variables, each ``[N, B, d_v]``."""

    x: jnp.ndarray  # local proposal (primal)
    z: jnp.ndarray  # consensus iterate
    y: jnp.ndarray  # scaled dual accumulator (u = lambda / rho)


def inverse_softplus(x: float) -> float:
    """Numerically safe inverse of softplus, for initializing softplus-parameterized scalars.

    Guards ``log(expm1(x))`` against underflow (clamp to >=1e-7) and overflow
    (identity for large x), so any positive init value maps to a finite raw param.
    """
    x = max(x, 1e-7)
    return x if x > 20.0 else math.log(math.expm1(x))


def run_admm(
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: jnp.ndarray,
    z_init: jnp.ndarray,  # [N, B, d_v] initial consensus state (encoder h, or zeros)
    num_iters: int,
    *,
    relaxation_alpha: float = 1.0,
    loss_window: int = 1,
    grad_window: int | None = None,
    x_projector: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[ADMMState, jnp.ndarray]:
    """Run ``num_iters`` ADMM steps.

    Returns the final :class:`ADMMState` and the stacked last ``loss_window``
    local proposals ``x`` of shape ``[W, N, B, d_v]`` (oldest first).
    """
    alpha = relaxation_alpha

    def step(state: ADMMState) -> ADMMState:
        z_prev = state.z
        x = x_solver.solve(state.z, state.y, rho, encoder_output, x_params)
        if x_projector is not None:
            x = x_projector(x)
        x_relaxed = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
        z_target = x_relaxed + state.y
        z = z_solver.solve(z_target, z_prev, geometry, z_params, rho)
        y = state.y + (x_relaxed - z)
        return ADMMState(x=x, z=z, y=y)

    state = ADMMState(x=z_init, z=z_init, y=jnp.zeros_like(z_init))

    K = num_iters
    n_detached = 0 if grad_window is None else max(0, K - grad_window)
    if n_detached > 0:
        state = jax.lax.fori_loop(0, n_detached, lambda _i, s: step(s), state)
        state = jax.tree_util.tree_map(jax.lax.stop_gradient, state)

    n_grad = K - n_detached
    window = min(loss_window, n_grad)
    n_pre = n_grad - window
    if n_pre > 0:
        state, _ = jax.lax.scan(lambda s, _: (step(s), None), state, None, length=n_pre)

    def collect(s, _):
        ns = step(s)
        return ns, ns.x

    state, x_window = jax.lax.scan(collect, state, None, length=window)
    return state, x_window


@struct.dataclass
class ADMMHistory:
    """Per-iteration ADMM trajectory, for visualization / convergence diagnostics.

    All arrays stack the ``K`` iterations on a leading axis (oldest first). The
    per-agent state arrays are ``[K, N, B, d_v]``; the residual arrays are the
    standard ADMM stopping diagnostics:

    * ``primal_res = ||x_relaxed - z||`` per agent (``[K, N, B]``) — how far the
      local proposal sits from the consensus it agreed to.
    * ``dual_res = rho * ||z - z_prev||`` per agent (``[K, N, B]``) — the change
      in the consensus iterate, scaled by the penalty (the dual feasibility
      residual of scaled-dual ADMM).
    * ``consistency_rms`` (``[K, B]``) — the geometry's sheaf-disagreement RMS,
      ``sqrt(mean over edges and stalk dims of r^2 + eps)``, the quantity the
      model is trained to drive to zero.
    * ``edge_disagreement`` (``[K, E, B]``) — the Euclidean norm of the exact
      shared-stalk residual on every edge,
      ``||F_{u->e} z_u - F_{v->e} z_v||_2``.  This is diagnostic-only: recording
      it does not mask an edge or otherwise change the baseline iteration.
    * ``message_novelty`` (``[K, E, B]``) — the larger of the two endpoint
      projection changes since the preceding iterate.  At ``k=1`` it is measured
      relative to ``z_init``.  This estimates redundant retransmission while
      remaining independent of endpoint agreement.
    """

    x: jnp.ndarray  # [K, N, B, d_v] local proposals
    z: jnp.ndarray  # [K, N, B, d_v] consensus iterates
    y: jnp.ndarray  # [K, N, B, d_v] scaled duals
    primal_res: jnp.ndarray  # [K, N, B] ||x_relaxed - z|| per agent
    dual_res: jnp.ndarray  # [K, N, B] rho * ||z - z_prev|| per agent
    consistency_rms: jnp.ndarray  # [K, B] sheaf disagreement RMS
    edge_disagreement: jnp.ndarray  # [K, E, B] ||F_u z_u - F_v z_v||_2
    message_novelty: jnp.ndarray  # [K, E, B] max endpoint projected-state change


@struct.dataclass
class TriggerHistory:
    """Event-trigger decisions aligned with an :class:`ADMMHistory` trajectory."""

    active_edges: jnp.ndarray  # [K,E,B] bool; both endpoint messages sent when true
    gate_novelty: jnp.ndarray  # [K,E,B] novelty relative to last transmitted messages
    gate_score: jnp.ndarray  # [K,E,B] absolute or cache-normalized score used by gate


@struct.dataclass
class TriggerSummary:
    """Memory-light per-iteration reductions for event-trigger evaluation."""

    active_edge_count: jnp.ndarray  # [K] active edge/example slots
    communication_probability_mean: jnp.ndarray  # [K] differentiable gate load
    gate_score_mean: jnp.ndarray  # [K]
    gate_score_max: jnp.ndarray  # [K]
    edge_disagreement_mean: jnp.ndarray  # [K]
    edge_disagreement_max: jnp.ndarray  # [K]
    effective_threshold_mean: jnp.ndarray  # [K]
    effective_threshold_min: jnp.ndarray  # [K]
    effective_threshold_max: jnp.ndarray  # [K]
    stale_residual_mean: jnp.ndarray  # [K]
    stale_residual_max: jnp.ndarray  # [K]


def run_admm_history(
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: jnp.ndarray,
    z_init: jnp.ndarray,
    num_iters: int,
    *,
    relaxation_alpha: float = 1.0,
    x_projector: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[ADMMState, ADMMHistory]:
    """Run ``num_iters`` ADMM steps, recording the full per-iteration trajectory.

    Same fixed-point iteration as :func:`run_admm`, but every iterate and its
    primal/dual residuals are stacked and returned in an :class:`ADMMHistory`
    (no gradient window — this is a forward-only diagnostic path). Use it to
    visualize coordination dynamics and x-vs-z trajectories on a single example.
    """
    alpha = relaxation_alpha
    rho_s = jnp.asarray(rho)

    def step(state: ADMMState, _):
        z_prev = state.z
        x = x_solver.solve(state.z, state.y, rho, encoder_output, x_params)
        if x_projector is not None:
            x = x_projector(x)
        x_relaxed = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
        z_target = x_relaxed + state.y
        z = z_solver.solve(z_target, z_prev, geometry, z_params, rho)
        y = state.y + (x_relaxed - z)
        new = ADMMState(x=x, z=z, y=y)
        primal_res = jnp.linalg.norm(x_relaxed - z, axis=-1)  # [N, B]
        dual_res = rho_s * jnp.linalg.norm(z - z_prev, axis=-1)  # [N, B]
        crms = geometry.consistency_rms(z)  # [B]
        edge_disagreement = jnp.linalg.norm(geometry.edge_residuals(z), axis=-1)  # [E, B]
        message_delta = geometry.endpoint_values(z) - geometry.endpoint_values(z_prev)
        message_novelty = jnp.max(jnp.linalg.norm(message_delta, axis=-1), axis=1)  # [E, B]
        return new, (new, primal_res, dual_res, crms, edge_disagreement, message_novelty)

    state = ADMMState(x=z_init, z=z_init, y=jnp.zeros_like(z_init))
    state, outputs = jax.lax.scan(step, state, None, length=num_iters)
    states, primal_res, dual_res, crms, edge_disagreement, message_novelty = outputs
    history = ADMMHistory(
        x=states.x,
        z=states.z,
        y=states.y,
        primal_res=primal_res,
        dual_res=dual_res,
        consistency_rms=crms,
        edge_disagreement=edge_disagreement,
        message_novelty=message_novelty,
    )
    return state, history


def run_admm_triggered_history(
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: jnp.ndarray,
    z_init: jnp.ndarray,
    num_iters: int,
    *,
    threshold: float,
    gate_mode: str = "absolute",
    gate_eps: float = 1e-6,
    relaxation_alpha: float = 1.0,
    x_projector: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[ADMMState, ADMMHistory, TriggerHistory]:
    """Forward-only sample-and-hold evaluation with an edge-level novelty gate.

    At the start of outer iteration ``k``, both endpoint projections on edge
    ``e`` are transmitted when their maximum change from the edge's cached
    projections exceeds ``threshold``. With ``gate_mode='relative'``, each
    endpoint change is divided by the cached projection norm plus ``gate_eps``.
    The first iteration always communicates.
    Active edges use the ordinary coupled sheaf term in the z-update; inactive
    endpoints use the neighbor projection held in the cache.  Training and the
    ordinary :func:`run_admm` path are unaffected.

    ``threshold <= 0`` forces every edge active and must reproduce
    :func:`run_admm_history` numerically.
    """
    if not hasattr(z_solver, "solve_sample_hold"):
        raise ValueError("event-trigger evaluation requires a sample-hold-capable z-solver")
    if gate_mode not in ("absolute", "relative"):
        raise ValueError(f"unknown gate_mode={gate_mode!r} (absolute|relative)")

    alpha = relaxation_alpha
    rho_s = jnp.asarray(rho)
    threshold_s = jnp.asarray(threshold, dtype=z_init.dtype)
    gate_eps_s = jnp.asarray(gate_eps, dtype=z_init.dtype)
    initial_state = ADMMState(x=z_init, z=z_init, y=jnp.zeros_like(z_init))
    initial_cache = geometry.endpoint_values(z_init)

    def step(carry, iteration):
        state, cache = carry
        z_prev = state.z
        messages = geometry.endpoint_values(z_prev)
        gate_delta = messages - cache
        endpoint_novelty = jnp.linalg.norm(gate_delta, axis=-1)  # [E,2,B]
        gate_novelty = jnp.max(endpoint_novelty, axis=1)  # [E,B]
        if gate_mode == "relative":
            cache_norm = jnp.linalg.norm(cache, axis=-1)
            gate_score = jnp.max(endpoint_novelty / (cache_norm + gate_eps_s), axis=1)
        else:
            gate_score = gate_novelty
        force_all = jnp.logical_or(iteration == 0, threshold_s <= 0)
        active = jnp.where(
            force_all, jnp.ones_like(gate_score, dtype=bool), gate_score > threshold_s
        )
        cache = jnp.where(active[:, None, :, None], messages, cache)

        x = x_solver.solve(state.z, state.y, rho, encoder_output, x_params)
        if x_projector is not None:
            x = x_projector(x)
        x_relaxed = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
        z_target = x_relaxed + state.y
        # Use the exact baseline code path for the zero-threshold control.  An
        # all-ones masked geometry is algebraically identical but can accumulate
        # different float32 rounding over many outer/CG iterations on a GPU.
        z = jax.lax.cond(
            threshold_s <= 0,
            lambda _: z_solver.solve(z_target, z_prev, geometry, z_params, rho),
            lambda _: z_solver.solve_sample_hold(
                z_target,
                z_prev,
                geometry,
                z_params,
                rho,
                active,
                cache,
            ),
            operand=None,
        )
        y = state.y + (x_relaxed - z)
        new = ADMMState(x=x, z=z, y=y)

        primal_res = jnp.linalg.norm(x_relaxed - z, axis=-1)
        dual_res = rho_s * jnp.linalg.norm(z - z_prev, axis=-1)
        crms = geometry.consistency_rms(z)
        edge_disagreement = jnp.linalg.norm(geometry.edge_residuals(z), axis=-1)
        message_delta = geometry.endpoint_values(z) - messages
        message_novelty = jnp.max(jnp.linalg.norm(message_delta, axis=-1), axis=1)
        outputs = (
            new,
            primal_res,
            dual_res,
            crms,
            edge_disagreement,
            message_novelty,
            active,
            gate_novelty,
            gate_score,
        )
        return (new, cache), outputs

    iterations = jnp.arange(num_iters)
    (state, _cache), outputs = jax.lax.scan(step, (initial_state, initial_cache), iterations)
    states, primal_res, dual_res, crms, edge_disagreement, message_novelty, active, gate, score = (
        outputs
    )
    history = ADMMHistory(
        x=states.x,
        z=states.z,
        y=states.y,
        primal_res=primal_res,
        dual_res=dual_res,
        consistency_rms=crms,
        edge_disagreement=edge_disagreement,
        message_novelty=message_novelty,
    )
    trigger_history = TriggerHistory(active_edges=active, gate_novelty=gate, gate_score=score)
    return state, history, trigger_history


def run_admm_triggered(
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: jnp.ndarray,
    z_init: jnp.ndarray,
    num_iters: int,
    *,
    threshold: float,
    trigger_mode: str = "fixed",
    threshold_end: float | None = None,
    schedule_power: float = 1.0,
    threshold_schedule: jnp.ndarray | None = None,
    residual_alpha: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0e6,
    edge_ema_decay: float = 0.9,
    edge_ema_scale: float = 1.0,
    prediction_decay: float = 1.0,
    early_stop_patience: int = 2,
    early_stop_min_iters: int = 0,
    coordinator_residual_weight: float = 1.0,
    coordinator_age_weight: float = 0.25,
    coordinator_max_silence: int = 8,
    coordinator_warmup_iters: int = 0,
    coordinator_final_sync_iters: int = 0,
    safe_residual_threshold: float = 0.25,
    safe_residual_power: float = 1.0,
    safe_max_silence: int = 8,
    safe_warmup_iters: int = 0,
    safe_final_sync_iters: int = 0,
    learned_gate_params: dict[str, jnp.ndarray] | None = None,
    learned_temperature: float = 1.0,
    learned_decision_threshold: float = 0.5,
    learned_max_silence: int = 8,
    learned_residual_guard: float = 1.0e6,
    gate_mode: str = "absolute",
    gate_eps: float = 1e-6,
    mask_policy: str = "trigger",
    active_count_schedule: jnp.ndarray | None = None,
    mask_seed: int = 0,
    relaxation_alpha: float = 1.0,
    x_projector: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[ADMMState, TriggerSummary]:
    """Run sample-and-hold ADMM while retaining only reduced diagnostics.

    This follows :func:`run_admm_triggered_history` exactly, but does not
    materialize the ``[K,N,B,d_v]`` state trajectories or ``[K,E,B]`` gate
    tensors.  It is intended for dataset-scale forward-only evaluation.
    Budgeted mask policies replace the novelty decision with a communication
    schedule. ``coordinator`` ranks edges separately for each batch example
    using novelty, stale residual, and time since the last transmission.
    """
    if not hasattr(z_solver, "solve_sample_hold"):
        raise ValueError("event-trigger evaluation requires a sample-hold-capable z-solver")
    if gate_mode not in ("absolute", "relative"):
        raise ValueError(f"unknown gate_mode={gate_mode!r} (absolute|relative)")
    if trigger_mode not in (
        "fixed",
        "iteration",
        "residual",
        "residual_proportional",
        "residual_hybrid",
        "edge_ema",
        "hysteresis",
        "predictive",
        "early_stop",
        "safe_residual",
        "learned",
    ):
        raise ValueError(
            f"unknown trigger_mode={trigger_mode!r} "
            "(fixed|iteration|residual|residual_proportional|"
            "residual_hybrid|edge_ema|hysteresis|predictive|early_stop|"
            "safe_residual|learned)"
        )
    if schedule_power <= 0:
        raise ValueError("schedule_power must be positive")
    if threshold_schedule is not None and threshold_schedule.shape != (num_iters,):
        raise ValueError(
            "threshold_schedule must have shape "
            f"({num_iters},), got {threshold_schedule.shape}"
        )
    if residual_alpha < 0:
        raise ValueError("residual_alpha must be non-negative")
    if threshold_min < 0 or threshold_max < threshold_min:
        raise ValueError("threshold bounds must satisfy 0 <= min <= max")
    if not 0 <= edge_ema_decay < 1:
        raise ValueError("edge_ema_decay must be in [0, 1)")
    if edge_ema_scale <= 0:
        raise ValueError("edge_ema_scale must be positive")
    if not 0 <= prediction_decay <= 1:
        raise ValueError("prediction_decay must be in [0, 1]")
    if early_stop_patience <= 0:
        raise ValueError("early_stop_patience must be positive")
    if not 0 <= early_stop_min_iters < num_iters:
        raise ValueError("early_stop_min_iters must be in [0, num_iters)")
    if coordinator_residual_weight < 0 or coordinator_age_weight < 0:
        raise ValueError("coordinator weights must be non-negative")
    if coordinator_max_silence <= 0:
        raise ValueError("coordinator_max_silence must be positive")
    if not 0 <= coordinator_warmup_iters <= num_iters:
        raise ValueError("coordinator_warmup_iters must be in [0, num_iters]")
    if not 0 <= coordinator_final_sync_iters <= num_iters:
        raise ValueError("coordinator_final_sync_iters must be in [0, num_iters]")
    if coordinator_warmup_iters + coordinator_final_sync_iters > num_iters:
        raise ValueError("coordinator warmup and final sync exceed num_iters")
    if safe_residual_threshold < 0 or safe_residual_power <= 0:
        raise ValueError("safe residual threshold must be non-negative and power positive")
    if safe_max_silence <= 0:
        raise ValueError("safe_max_silence must be positive")
    if not 0 <= safe_warmup_iters <= num_iters:
        raise ValueError("safe_warmup_iters must be in [0, num_iters]")
    if not 0 <= safe_final_sync_iters <= num_iters:
        raise ValueError("safe_final_sync_iters must be in [0, num_iters]")
    if safe_warmup_iters + safe_final_sync_iters > num_iters:
        raise ValueError("safe residual warmup and final sync exceed num_iters")
    if learned_temperature <= 0:
        raise ValueError("learned_temperature must be positive")
    if not 0 <= learned_decision_threshold <= 1:
        raise ValueError("learned_decision_threshold must be in [0, 1]")
    if learned_max_silence <= 0:
        raise ValueError("learned_max_silence must be positive")
    if learned_residual_guard < 0:
        raise ValueError("learned_residual_guard must be non-negative")
    if trigger_mode == "learned":
        if learned_gate_params is None:
            raise ValueError("learned trigger requires learned_gate_params")
        required = {"w1", "b1", "w2", "b2"}
        if set(learned_gate_params) != required:
            raise ValueError(
                "learned_gate_params must contain exactly w1, b1, w2, and b2"
            )
        hidden_dim = learned_gate_params["w1"].shape[1]
        expected_shapes = {
            "w1": (LEARNED_GATE_FEATURES, hidden_dim),
            "b1": (hidden_dim,),
            "w2": (hidden_dim, 1),
            "b2": (1,),
        }
        for name, expected in expected_shapes.items():
            if learned_gate_params[name].shape != expected:
                raise ValueError(
                    f"learned_gate_params[{name!r}] must have shape "
                    f"{expected}, got {learned_gate_params[name].shape}"
                )
    hysteresis_on = threshold if threshold_end is None else threshold_end
    if trigger_mode == "hysteresis" and hysteresis_on < threshold:
        raise ValueError("hysteresis requires threshold_end >= threshold")
    if mask_policy not in ("trigger", "random", "periodic", "topk", "coordinator", "none"):
        raise ValueError(
            f"unknown mask_policy={mask_policy!r} "
            "(trigger|random|periodic|topk|coordinator|none)"
        )
    if mask_policy not in ("trigger", "none") and active_count_schedule is None:
        raise ValueError(f"mask_policy={mask_policy!r} requires active_count_schedule")
    if mask_policy == "none" and not hasattr(z_solver, "solve_sample_hold_local"):
        raise ValueError("mask_policy='none' requires node-local sample-hold CG")
    if trigger_mode == "learned" and mask_policy != "trigger":
        raise ValueError("learned trigger requires mask_policy='trigger'")
    if active_count_schedule is not None and active_count_schedule.shape != (num_iters,):
        raise ValueError(
            "active_count_schedule must have shape "
            f"({num_iters},), got {active_count_schedule.shape}"
        )

    alpha = relaxation_alpha
    threshold_s = jnp.asarray(threshold, dtype=z_init.dtype)
    threshold_end_s = jnp.asarray(
        threshold if threshold_end is None else threshold_end, dtype=z_init.dtype
    )
    schedule_power_s = jnp.asarray(schedule_power, dtype=z_init.dtype)
    threshold_schedule_s = (
        None
        if threshold_schedule is None
        else jnp.asarray(threshold_schedule, dtype=z_init.dtype)
    )
    residual_alpha_s = jnp.asarray(residual_alpha, dtype=z_init.dtype)
    threshold_min_s = jnp.asarray(threshold_min, dtype=z_init.dtype)
    threshold_max_s = jnp.asarray(threshold_max, dtype=z_init.dtype)
    edge_ema_decay_s = jnp.asarray(edge_ema_decay, dtype=z_init.dtype)
    edge_ema_scale_s = jnp.asarray(edge_ema_scale, dtype=z_init.dtype)
    prediction_decay_s = jnp.asarray(prediction_decay, dtype=z_init.dtype)
    coordinator_residual_weight_s = jnp.asarray(
        coordinator_residual_weight, dtype=z_init.dtype
    )
    coordinator_age_weight_s = jnp.asarray(
        coordinator_age_weight, dtype=z_init.dtype
    )
    safe_residual_threshold_s = jnp.asarray(
        safe_residual_threshold, dtype=z_init.dtype
    )
    safe_residual_power_s = jnp.asarray(safe_residual_power, dtype=z_init.dtype)
    learned_temperature_s = jnp.asarray(learned_temperature, dtype=z_init.dtype)
    learned_decision_threshold_s = jnp.asarray(
        learned_decision_threshold, dtype=z_init.dtype
    )
    learned_residual_guard_s = jnp.asarray(
        learned_residual_guard, dtype=z_init.dtype
    )
    gate_eps_s = jnp.asarray(gate_eps, dtype=z_init.dtype)
    initial_state = ADMMState(x=z_init, z=z_init, y=jnp.zeros_like(z_init))
    # A true no-exchange control has no bootstrap messages in its cache. Other
    # policies retain the historical shared initial synchronization.
    initial_cache = (
        jnp.zeros_like(geometry.endpoint_values(z_init))
        if mask_policy == "none"
        else geometry.endpoint_values(z_init)
    )
    initial_edge_ema = jnp.full(
        (initial_cache.shape[0], initial_cache.shape[2]),
        threshold_s,
        dtype=z_init.dtype,
    )
    initial_enabled = jnp.ones(
        (initial_cache.shape[0], initial_cache.shape[2]), dtype=bool
    )
    initial_prediction_delta = jnp.zeros_like(initial_cache)
    initial_prediction_age = jnp.zeros(
        (initial_cache.shape[0], initial_cache.shape[2]), dtype=z_init.dtype
    )
    initial_stable_count = jnp.zeros((initial_cache.shape[2],), dtype=jnp.int32)
    initial_finished = jnp.zeros((initial_cache.shape[2],), dtype=bool)
    edge_count = initial_cache.shape[0]
    slot_count = edge_count * initial_cache.shape[2]
    budget_limit = 0 if mask_policy == "none" else (
        edge_count if mask_policy == "coordinator" else slot_count
    )
    if mask_policy == "none":
        count_schedule = jnp.zeros((num_iters,), dtype=jnp.int32)
    elif active_count_schedule is None:
        count_schedule = jnp.full((num_iters,), budget_limit, dtype=jnp.int32)
    else:
        count_schedule = jnp.clip(
            jnp.asarray(active_count_schedule, dtype=jnp.int32), 0, budget_limit
        )
    periodic_offsets = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(count_schedule[:-1], dtype=jnp.int32),
        )
    )
    slot_indices = jnp.arange(slot_count, dtype=jnp.int32)
    random_key = jax.random.PRNGKey(mask_seed)

    def step(carry, iteration):
        (
            state,
            cache,
            edge_ema,
            enabled,
            prediction_delta,
            prediction_age,
            stable_count,
            finished,
        ) = carry
        z_prev = state.z
        messages = geometry.endpoint_values(z_prev)
        if trigger_mode == "predictive":
            predicted_cache = cache + (
                prediction_decay_s
                * prediction_age[:, None, :, None]
                * prediction_delta
            )
        else:
            predicted_cache = cache
        endpoint_novelty = jnp.linalg.norm(
            messages - predicted_cache, axis=-1
        )  # [E,2,B]
        if gate_mode == "relative":
            cache_norm = jnp.linalg.norm(predicted_cache, axis=-1)
            endpoint_gate_score = endpoint_novelty / (cache_norm + gate_eps_s)
        else:
            cache_norm = jnp.linalg.norm(predicted_cache, axis=-1)
            endpoint_gate_score = endpoint_novelty
        gate_score = jnp.max(endpoint_gate_score, axis=1)
        opposite_cache = predicted_cache[:, ::-1]
        stale_endpoint_residual = jnp.linalg.norm(messages - opposite_cache, axis=-1)
        if gate_mode == "relative":
            opposite_norm = jnp.linalg.norm(opposite_cache, axis=-1)
            endpoint_stale_score = stale_endpoint_residual / (
                opposite_norm + gate_eps_s
            )
        else:
            endpoint_stale_score = stale_endpoint_residual
        stale_residual = jnp.max(endpoint_stale_score, axis=1)

        # Relative scores can spike when a cached projection is close to zero.
        # Winsorize each observation against its own running scale so one spike
        # cannot pin every edge to the global threshold ceiling.
        bounded_gate_score = jnp.clip(
            gate_score,
            0.5 * edge_ema,
            2.0 * edge_ema,
        )
        updated_edge_ema = jnp.where(
            iteration == 0,
            edge_ema,
            edge_ema_decay_s * edge_ema
            + (1.0 - edge_ema_decay_s) * bounded_gate_score,
        )
        if trigger_mode == "iteration":
            if threshold_schedule_s is None:
                denominator = jnp.asarray(max(num_iters - 1, 1), dtype=z_init.dtype)
                progress = jnp.asarray(iteration, dtype=z_init.dtype) / denominator
                effective_threshold = threshold_s + (
                    threshold_end_s - threshold_s
                ) * progress**schedule_power_s
            else:
                effective_threshold = threshold_schedule_s[iteration]
            effective_threshold = jnp.broadcast_to(effective_threshold, gate_score.shape)
        elif trigger_mode == "residual":
            effective_threshold = threshold_s / (
                1.0 + residual_alpha_s * stale_residual
            )
        elif trigger_mode == "residual_proportional":
            effective_threshold = threshold_s * stale_residual
        elif trigger_mode == "safe_residual":
            effective_threshold = threshold_s * (
                stale_residual + gate_eps_s
            ) ** safe_residual_power_s
        elif trigger_mode == "residual_hybrid":
            denominator = jnp.asarray(max(num_iters - 1, 1), dtype=z_init.dtype)
            progress = jnp.asarray(iteration, dtype=z_init.dtype) / denominator
            scheduled_threshold = threshold_s + (
                threshold_end_s - threshold_s
            ) * progress**schedule_power_s
            effective_threshold = scheduled_threshold / (
                1.0 + residual_alpha_s * stale_residual
            )
        elif trigger_mode == "edge_ema":
            effective_threshold = edge_ema_scale_s * updated_edge_ema
        elif trigger_mode == "learned":
            effective_threshold = jnp.full_like(
                gate_score, learned_decision_threshold_s
            )
        else:
            effective_threshold = jnp.broadcast_to(threshold_s, gate_score.shape)
        if trigger_mode not in (
            "residual_proportional",
            "residual_hybrid",
            "safe_residual",
        ):
            effective_threshold = jnp.clip(
                effective_threshold, threshold_min_s, threshold_max_s
            )
        updated_enabled = enabled
        updated_stable_count = stable_count
        updated_finished = finished
        if mask_policy == "trigger" and trigger_mode == "early_stop":
            sample_residual = jnp.mean(stale_residual, axis=0)
            eligible = iteration >= early_stop_min_iters
            below = jnp.logical_and(eligible, sample_residual <= threshold_s)
            updated_stable_count = jnp.where(
                jnp.logical_and(below, jnp.logical_not(finished)),
                stable_count + 1,
                jnp.where(finished, stable_count, 0),
            )
            updated_finished = jnp.logical_or(
                finished, updated_stable_count >= early_stop_patience
            )
            active = jnp.broadcast_to(
                jnp.logical_not(updated_finished)[None, :], gate_score.shape
            )
            active = jnp.where(
                iteration == 0, jnp.ones_like(active, dtype=bool), active
            )
        elif mask_policy == "trigger" and trigger_mode == "safe_residual":
            force_full = jnp.logical_or(
                iteration < safe_warmup_iters,
                iteration >= num_iters - safe_final_sync_iters,
            )
            safety_due = prediction_age >= safe_max_silence
            active = jnp.logical_or(
                stale_residual > safe_residual_threshold_s,
                gate_score > effective_threshold,
            )
            active = jnp.logical_or(active, safety_due)
            active = jnp.where(
                jnp.logical_or(iteration == 0, force_full),
                jnp.ones_like(active),
                active,
            )
        elif mask_policy == "trigger" and trigger_mode == "hysteresis":
            updated_enabled = jnp.where(
                enabled,
                gate_score > threshold_s,
                gate_score > threshold_end_s,
            )
            active = jnp.where(
                iteration == 0,
                jnp.ones_like(gate_score, dtype=bool),
                updated_enabled,
            )
        elif mask_policy == "trigger" and trigger_mode == "learned":
            assert learned_gate_params is not None
            dual_norm = jnp.linalg.norm(
                geometry.endpoint_values(state.y), axis=-1
            )
            normalized_age = prediction_age / jnp.asarray(
                learned_max_silence, dtype=z_init.dtype
            )
            progress = jnp.asarray(iteration, dtype=z_init.dtype) / jnp.asarray(
                max(num_iters - 1, 1), dtype=z_init.dtype
            )
            features = jnp.stack(
                (
                    jnp.log1p(endpoint_gate_score),
                    jnp.log1p(endpoint_stale_score),
                    jnp.log1p(dual_norm),
                    jnp.broadcast_to(
                        normalized_age[:, None, :], endpoint_gate_score.shape
                    ),
                    jnp.full_like(endpoint_gate_score, progress),
                ),
                axis=-1,
            )
            w1 = jnp.asarray(learned_gate_params["w1"], dtype=z_init.dtype)
            b1 = jnp.asarray(learned_gate_params["b1"], dtype=z_init.dtype)
            w2 = jnp.asarray(learned_gate_params["w2"], dtype=z_init.dtype)
            b2 = jnp.asarray(learned_gate_params["b2"], dtype=z_init.dtype)
            hidden = jnp.tanh(jnp.einsum("eobf,fh->eobh", features, w1) + b1)
            endpoint_logits = (
                jnp.einsum("eobh,hq->eobq", hidden, w2) + b2
            )[..., 0]
            endpoint_probability = jax.nn.sigmoid(
                endpoint_logits / learned_temperature_s
            )
            edge_probability = 1.0 - jnp.prod(
                1.0 - endpoint_probability, axis=1
            )
            learned_hard = edge_probability >= learned_decision_threshold_s
            learned_force = jnp.logical_or(
                jnp.logical_or(
                    iteration == 0, prediction_age >= learned_max_silence
                ),
                stale_residual > learned_residual_guard_s,
            )
            active = jnp.logical_or(learned_hard, learned_force)
            straight_through = edge_probability + jax.lax.stop_gradient(
                active.astype(z_init.dtype) - edge_probability
            )
            learned_active_weight = jnp.where(
                learned_force, jnp.ones_like(straight_through), straight_through
            )
            communication_probability = jnp.where(
                learned_force, jnp.ones_like(edge_probability), edge_probability
            )
        elif mask_policy == "trigger":
            current_threshold = (
                threshold_s
                if threshold_schedule_s is None
                else threshold_schedule_s[iteration]
            )
            force_all = jnp.logical_or(iteration == 0, current_threshold <= 0)
            active = jnp.where(
                force_all,
                jnp.ones_like(gate_score, dtype=bool),
                gate_score > effective_threshold,
            )
        elif mask_policy == "none":
            active = jnp.zeros_like(gate_score, dtype=bool)
            communication_probability = jnp.zeros_like(gate_score)
        elif mask_policy == "random":
            target_count = count_schedule[iteration]
            ranks = jax.random.permutation(
                jax.random.fold_in(random_key, iteration), slot_count
            )
            active = (ranks < target_count).reshape(gate_score.shape)
        elif mask_policy == "periodic":
            target_count = count_schedule[iteration]
            offset = periodic_offsets[iteration] % slot_count
            active = (((slot_indices - offset) % slot_count) < target_count).reshape(
                gate_score.shape
            )
        elif mask_policy == "topk":
            target_count = count_schedule[iteration]
            flat_score = gate_score.reshape((slot_count,))
            ranks = jnp.argsort(jnp.argsort(-flat_score))
            active = (ranks < target_count).reshape(gate_score.shape)
        else:
            target_count = count_schedule[iteration]
            normalized_age = jnp.minimum(
                prediction_age
                / jnp.asarray(coordinator_max_silence, dtype=z_init.dtype),
                1.0,
            )
            coordinator_score = (
                gate_score
                + coordinator_residual_weight_s * stale_residual
                + coordinator_age_weight_s * normalized_age
            )
            overdue_bonus = jnp.where(
                prediction_age >= coordinator_max_silence,
                jnp.asarray(1.0e6, dtype=z_init.dtype),
                jnp.asarray(0.0, dtype=z_init.dtype),
            )
            ranks = jnp.argsort(
                jnp.argsort(-(coordinator_score + overdue_bonus), axis=0), axis=0
            )
            active = ranks < target_count
            force_full = jnp.logical_or(
                iteration < coordinator_warmup_iters,
                iteration >= num_iters - coordinator_final_sync_iters,
            )
            active = jnp.where(force_full, jnp.ones_like(active), active)

        if trigger_mode == "learned":
            active_weight = learned_active_weight
            active_expanded = active_weight[:, None, :, None]
            solver_cache = (
                active_expanded * messages
                + (1.0 - active_expanded) * predicted_cache
            )
        else:
            active_weight = active.astype(z_init.dtype)
            communication_probability = active_weight
            active_expanded = active[:, None, :, None]
            solver_cache = jnp.where(active_expanded, messages, predicted_cache)
        if trigger_mode == "predictive":
            updated_prediction_delta = jnp.where(
                active_expanded, messages - cache, prediction_delta
            )
            updated_cache = jnp.where(active_expanded, messages, cache)
        else:
            updated_prediction_delta = prediction_delta
            updated_cache = solver_cache
        updated_prediction_age = jnp.where(
            active, jnp.zeros_like(prediction_age), prediction_age + 1.0
        )

        x = x_solver.solve(state.z, state.y, rho, encoder_output, x_params)
        if x_projector is not None:
            x = x_projector(x)
        x_relaxed = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
        z_target = x_relaxed + state.y
        if mask_policy == "none":
            z = z_solver.solve_sample_hold_local(
                z_target,
                z_prev,
                geometry,
                z_params,
                rho,
                solver_cache,
            )
        elif mask_policy == "trigger" and trigger_mode != "learned":
            z = jax.lax.cond(
                threshold_s <= 0,
                lambda _: z_solver.solve(z_target, z_prev, geometry, z_params, rho),
                lambda _: z_solver.solve_sample_hold(
                    z_target,
                    z_prev,
                    geometry,
                    z_params,
                    rho,
                    active,
                    solver_cache,
                ),
                operand=None,
            )
        else:
            z = z_solver.solve_sample_hold(
                z_target,
                z_prev,
                geometry,
                z_params,
                rho,
                active_weight,
                solver_cache,
            )
        y = state.y + (x_relaxed - z)
        new = ADMMState(x=x, z=z, y=y)
        edge_disagreement = jnp.linalg.norm(geometry.edge_residuals(z), axis=-1)
        reduced = (
            jnp.sum(active_weight),
            jnp.mean(communication_probability),
            jnp.mean(gate_score),
            jnp.max(gate_score),
            jnp.mean(edge_disagreement),
            jnp.max(edge_disagreement),
            jnp.mean(effective_threshold),
            jnp.min(effective_threshold),
            jnp.max(effective_threshold),
            jnp.mean(stale_residual),
            jnp.max(stale_residual),
        )
        return (
            new,
            updated_cache,
            updated_edge_ema,
            updated_enabled,
            updated_prediction_delta,
            updated_prediction_age,
            updated_stable_count,
            updated_finished,
        ), reduced

    iterations = jnp.arange(num_iters)
    (
        state,
        _cache,
        _edge_ema,
        _enabled,
        _prediction_delta,
        _prediction_age,
        _stable_count,
        _finished,
    ), reduced = jax.lax.scan(
        step,
        (
            initial_state,
            initial_cache,
            initial_edge_ema,
            initial_enabled,
            initial_prediction_delta,
            initial_prediction_age,
            initial_stable_count,
            initial_finished,
        ),
        iterations,
    )
    (
        active_count,
        communication_probability_mean,
        score_mean,
        score_max,
        disagreement_mean,
        disagreement_max,
        effective_threshold_mean,
        effective_threshold_min,
        effective_threshold_max,
        stale_residual_mean,
        stale_residual_max,
    ) = reduced
    summary = TriggerSummary(
        active_edge_count=active_count,
        communication_probability_mean=communication_probability_mean,
        gate_score_mean=score_mean,
        gate_score_max=score_max,
        edge_disagreement_mean=disagreement_mean,
        edge_disagreement_max=disagreement_max,
        effective_threshold_mean=effective_threshold_mean,
        effective_threshold_min=effective_threshold_min,
        effective_threshold_max=effective_threshold_max,
        stale_residual_mean=stale_residual_mean,
        stale_residual_max=stale_residual_max,
    )
    return state, summary
