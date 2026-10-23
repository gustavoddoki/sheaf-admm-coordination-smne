"""Unrolled conjugate-gradient z-solver (the paper's default consensus step).

The z-update solves a PSD linear system in the sheaf Laplacian. We run a *fixed*
number ``T`` of CG iterations and let gradients flow through every step via
autodiff (explicit unroll, not implicit differentiation): implicit diff would
require CG to have converged to give correct gradients, whereas the unrolled
gradients capture the actual (under-solved) optimization trajectory the model is
trained with — essential for meta-learning the restriction maps.

Two modes:

* ``project`` (hard consensus, ``F z = 0``): solve ``(L + eps I) w = L z_target``
  and return ``z_target - w``. The tiny Tikhonov ``eps`` makes the singular ``L``
  invertible (its kernel is the consensus subspace).
* ``prox`` (soft consensus): solve ``(gamma L + rho I) z = rho z_target``.

CG runs batched: each batch element is an independent system, so the inner
products reduce over agents and stalk dims but keep the batch axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from sheaf_admm.geometry.base import SheafGeometry
from sheaf_admm.solvers.base import ZSolverParams, rho_as_vec

# CG denominator guard. 1e-8 (not 1e-12) because pTAp / rTr get small near
# convergence and 1e-12 sits below float32 roundoff there, risking 0/0 -> NaN
# in both the forward pass and the unrolled gradient. Distinct from the
# project-mode Tikhonov ``tikhonov_eps`` below.
_CG_DENOM_EPS = 1e-8


@struct.dataclass
class UnrolledCGParams(ZSolverParams):
    num_iters: int = struct.field(pytree_node=False, default=5)  # T
    tikhonov_eps: float = struct.field(pytree_node=False, default=1e-5)  # project-mode L+eps*I
    prox_init: str = struct.field(pytree_node=False, default="legacy")  # "legacy" | "warm"
    scan_unroll: int = struct.field(pytree_node=False, default=5)  # compile-time unroll, not T


def _batched_cg(matvec, b, x0, num_iters, scan_unroll):
    """Batched CG for ``A x = b`` over node states ``[N, B, d]``.

    Inner products reduce over agents (axis 0) and stalk dim (axis 2), keeping
    the batch axis -> per-batch scalars. ``num_iters`` is fixed (no early stop)
    for consistent gradients.
    """

    def bdot(a, c):
        return jnp.sum(a * c, axis=(0, 2))  # [B]

    def bscale(s, v):
        return s[None, :, None] * v  # [B] -> [1, B, 1]

    r0 = b - matvec(x0)
    p0 = r0
    rTr0 = bdot(r0, r0)

    def step(carry, _):
        x, r, p, rTr = carry
        Ap = matvec(p)
        pTAp = bdot(p, Ap)
        alpha = rTr / (pTAp + _CG_DENOM_EPS)
        x = x + bscale(alpha, p)
        r = r - bscale(alpha, Ap)
        rTr_new = bdot(r, r)
        beta = rTr_new / (rTr + _CG_DENOM_EPS)
        p = r + bscale(beta, p)
        return (x, r, p, rTr_new), None

    (x, *_), _ = jax.lax.scan(step, (x0, r0, p0, rTr0), None, length=num_iters, unroll=scan_unroll)
    return x


def _nodewise_cg(matvec, b, x0, num_iters, scan_unroll):
    """CG with independent coefficients for each node and batch example.

    Use only when the operator is block diagonal over agents. Unlike
    :func:`_batched_cg`, this does not reduce scalar products across the node
    axis, so one agent's residual cannot affect another agent's step sizes.
    """

    def bdot(a, c):
        return jnp.sum(a * c, axis=2)  # [N, B]

    def bscale(s, v):
        return s[:, :, None] * v  # [N, B] -> [N, B, 1]

    r0 = b - matvec(x0)
    p0 = r0
    rTr0 = bdot(r0, r0)

    def step(carry, _):
        x, r, p, rTr = carry
        Ap = matvec(p)
        pTAp = bdot(p, Ap)
        alpha = rTr / (pTAp + _CG_DENOM_EPS)
        x = x + bscale(alpha, p)
        r = r - bscale(alpha, Ap)
        rTr_new = bdot(r, r)
        beta = rTr_new / (rTr + _CG_DENOM_EPS)
        p = r + bscale(beta, p)
        return (x, r, p, rTr_new), None

    (x, *_), _ = jax.lax.scan(
        step, (x0, r0, p0, rTr0), None, length=num_iters, unroll=scan_unroll
    )
    return x


class UnrolledCGZSolver:
    @staticmethod
    def solve(z_target, z_prev, geometry: SheafGeometry, params: UnrolledCGParams, rho):
        if params.mode == "project":
            eps = jnp.asarray(params.tikhonov_eps, dtype=z_target.dtype)
            matvec = lambda x: geometry.laplacian_apply(x) + eps * x  # noqa: E731
            b = geometry.laplacian_apply(z_target)
            # Warm-start the correction at the ADMM target delta. This is the
            # paper-run inexact hard-consensus path, not a standalone exact
            # Euclidean projector: kernel components in the warm start are not
            # removed by the RHS. Use zero/range-space initialization if exact
            # projection idempotence is required.
            w0 = z_target - z_prev
            w = _batched_cg(matvec, b, w0, params.num_iters, params.scan_unroll)
            return z_target - w

        if params.mode == "prox":
            gamma = jnp.asarray(params.gamma, dtype=z_target.dtype)
            rho_v = rho_as_vec(rho, z_target)
            matvec = lambda x: gamma * geometry.laplacian_apply(x) + rho_v * x  # noqa: E731
            b = rho_v * z_target
            # Detach the warm-start so it does not unroll the whole ADMM chain.
            z0 = z_target if params.prox_init == "legacy" else jax.lax.stop_gradient(z_prev)
            return _batched_cg(matvec, b, z0, params.num_iters, params.scan_unroll)

        raise ValueError(f"unknown z mode {params.mode!r} (project|prox)")

    @staticmethod
    def solve_sample_hold(
        z_target,
        z_prev,
        geometry: SheafGeometry,
        params: UnrolledCGParams,
        rho,
        active_edges,
        cached_messages,
    ):
        """Prox z-update with active coupled edges and sample-held inactive edges.

        An active edge uses the ordinary sheaf term ``1/2 ||F_u z_u-F_v z_v||²``.
        For an inactive edge, each endpoint uses the other endpoint's last
        transmitted projection as a fixed target.  The two one-sided penalties
        receive weight 1/4 each, matching the ordinary edge energy when both
        cached values equal the current projections.

        The active mask is fixed throughout this CG solve, so the resulting
        Hessian remains linear, symmetric, and positive semidefinite.
        """
        if params.mode != "prox":
            raise ValueError("sample-hold evaluation currently supports z_mode='prox' only")

        gamma = jnp.asarray(params.gamma, dtype=z_target.dtype)
        rho_v = rho_as_vec(rho, z_target)
        active = jnp.asarray(active_edges, dtype=z_target.dtype)  # [E,B]
        inactive = 1.0 - active
        active_geometry = geometry.replace(edge_mask=active)
        u, v = geometry.edge_indices[:, 0], geometry.edge_indices[:, 1]

        def scatter(endpoint_contrib):
            out = jnp.zeros_like(z_target)
            out = out.at[u].add(endpoint_contrib[:, 0])
            out = out.at[v].add(endpoint_contrib[:, 1])
            return out

        def inactive_hessian_apply(z):
            values = geometry.endpoint_values(z) * inactive[:, None, :, None]
            return 0.5 * scatter(geometry.endpoint_adjoint(values))

        def matvec(z):
            hessian_z = active_geometry.laplacian_apply(z) + inactive_hessian_apply(z)
            return gamma * hessian_z + rho_v * z

        # Each inactive endpoint is attracted to the other endpoint's cached projection.
        cached_other = cached_messages[:, ::-1]
        held_values = cached_other * inactive[:, None, :, None]
        cache_rhs = 0.5 * scatter(geometry.endpoint_adjoint(held_values))
        b = rho_v * z_target + gamma * cache_rhs
        z0 = z_target if params.prox_init == "legacy" else jax.lax.stop_gradient(z_prev)
        return _batched_cg(matvec, b, z0, params.num_iters, params.scan_unroll)

    @staticmethod
    def solve_sample_hold_local(
        z_target,
        z_prev,
        geometry: SheafGeometry,
        params: UnrolledCGParams,
        rho,
        cached_messages,
    ):
        """No-edge-message z-update with node-local CG reductions.

        All edge projections are held at their cached values. The resulting
        Hessian is block diagonal by agent; nodewise CG is required to prevent
        the usual global CG coefficients from becoming an implicit channel.
        """
        if params.mode != "prox":
            raise ValueError("local sample-hold evaluation supports z_mode='prox' only")

        gamma = jnp.asarray(params.gamma, dtype=z_target.dtype)
        rho_v = rho_as_vec(rho, z_target)
        inactive = jnp.ones(
            (geometry.edge_indices.shape[0], z_target.shape[1]), dtype=z_target.dtype
        )
        u, v = geometry.edge_indices[:, 0], geometry.edge_indices[:, 1]

        def scatter(endpoint_contrib):
            out = jnp.zeros_like(z_target)
            out = out.at[u].add(endpoint_contrib[:, 0])
            out = out.at[v].add(endpoint_contrib[:, 1])
            return out

        def inactive_hessian_apply(z):
            values = geometry.endpoint_values(z) * inactive[:, None, :, None]
            return 0.5 * scatter(geometry.endpoint_adjoint(values))

        def matvec(z):
            return gamma * inactive_hessian_apply(z) + rho_v * z

        cached_other = cached_messages[:, ::-1]
        held_values = cached_other * inactive[:, None, :, None]
        cache_rhs = 0.5 * scatter(geometry.endpoint_adjoint(held_values))
        b = rho_v * z_target + gamma * cache_rhs
        z0 = z_target if params.prox_init == "legacy" else jax.lax.stop_gradient(z_prev)
        return _nodewise_cg(matvec, b, z0, params.num_iters, params.scan_unroll)
