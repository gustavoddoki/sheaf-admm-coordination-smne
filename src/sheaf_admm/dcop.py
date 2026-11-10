"""Learned consensus ADMM (Sheaf-ADMM with identity restriction maps) for pairwise DCOPs.

Agents are the constraint edges. Edge e = (u_e, w_e) keeps local copies a_e, b_e in the simplex over
the d values of its two variables. Each variable i has a consensus vector z_i shared by the agents
of its incident edges (the sheaf with identity restriction maps on the variable stalks). One
unrolled iteration is one communication round:

  local (learned): a = P(z_u - l - alpha_t * (C b) + delta_theta(...)), then the same for b
  consensus:       z_i = P(mean over incident copies of (copy + dual)); committed variables stay one-hot
  dual:            l += a - z_u, m += b - z_w

delta_theta is a small MLP applied per value, with value-pooled features, so the model is
equivariant to value permutations and runs for any domain size d and any number of agents.
Training is unsupervised: minimize the expected cost sum_e z_u^T C_e z_w over the unrolled
iterations (a window), with a random symmetry-breaking initialization of z.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def proj_simplex(v):
    d = v.shape[-1]
    u = jnp.sort(v, axis=-1)[..., ::-1]
    css = jnp.cumsum(u, axis=-1) - 1.0
    k = jnp.arange(1, d + 1)
    cond = u - css / k > 0
    rho = jnp.sum(cond, axis=-1) - 1  # cond is True on a prefix
    theta = jnp.take_along_axis(css, rho[..., None], -1) / (rho[..., None] + 1.0)
    return jnp.maximum(v - theta, 0.0)


def init_params(key, hidden=32, n_feat=4, n_iters=50):
    k1, k2, k3 = jax.random.split(key, 3)
    fin = 2 * n_feat
    return {
        "w1": jax.random.normal(k1, (fin, hidden)) / np.sqrt(fin),
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(k2, (hidden, hidden)) / np.sqrt(hidden),
        "b2": jnp.zeros(hidden),
        "w3": jax.random.normal(k3, (hidden, 1)) * 0.01,
        "b3": jnp.zeros(1),
        "log_alpha": jnp.full((n_iters,), np.log(1.0)),
        "log_noise": jnp.array(0.0),
    }


def _delta(params, feats):
    """feats [..., d, F] -> per-value correction [..., d]; value-equivariant via pooled features."""
    pooled = jnp.broadcast_to(feats.mean(-2, keepdims=True), feats.shape)
    h = jnp.concatenate([feats, pooled], -1)
    h = jax.nn.gelu(h @ params["w1"] + params["b1"])
    h = jax.nn.gelu(h @ params["w2"] + params["b2"])
    return (h @ params["w3"] + params["b3"])[..., 0]


def _side(params, tgt, g, alpha):
    feats = jnp.stack([tgt, g, tgt - tgt.mean(-1, keepdims=True), g - g.mean(-1, keepdims=True)], -1)
    return proj_simplex(tgt - alpha * g + _delta(params, feats))


def expected_cost(z, u, w, C, emask):
    return jnp.sum(jnp.einsum("ea,eab,eb->e", z[u], C, z[w]) * emask)


def init_state(params, inst, key):
    u, w, deg = inst["u"], inst["w"], inst["deg"]
    N, d = deg.shape[0], inst["C"].shape[-1]
    z = jax.nn.softmax(jnp.exp(params["log_noise"]) * jax.random.normal(key, (N, d)), -1)
    a, b = z[u], z[w]
    return (z, a, b, jnp.zeros_like(a), jnp.zeros_like(b))


def unroll(params, inst, key, n_iters, fixed=None, state=None, t0=0):
    """inst: dict with u, w [E], C [E,d,d] (normalized), emask [E], deg [N].
    Runs n_iters iterations from `state` (or a fresh random one). Returns (final state, z per iteration)."""
    u, w, C, emask, deg = inst["u"], inst["w"], inst["C"], inst["emask"], inst["deg"]
    N, d = deg.shape[0], C.shape[-1]
    if state is None:
        state = init_state(params, inst, key)
    alen = params["log_alpha"].shape[0]
    if fixed is None:
        fixed = -jnp.ones(N, dtype=jnp.int32)
    onehot_fixed = jax.nn.one_hot(jnp.maximum(fixed, 0), d)
    is_fixed = (fixed >= 0)[:, None]

    def step(carry, t):
        z, a, b, lam, mu = carry
        alpha = jnp.exp(params["log_alpha"][jnp.minimum(t, alen - 1)])
        a = _side(params, z[u] - lam, jnp.einsum("eab,eb->ea", C, b), alpha)
        b = _side(params, z[w] - mu, jnp.einsum("eab,ea->eb", C, a), alpha)
        m = emask[:, None]
        acc = jnp.zeros((N, d)).at[u].add((a + lam) * m).at[w].add((b + mu) * m)
        z = proj_simplex(acc / jnp.maximum(deg, 1.0)[:, None])
        z = jnp.where(deg[:, None] > 0, z, 1.0 / d)
        z = jnp.where(is_fixed, onehot_fixed, z)
        lam = lam + (a - z[u]) * m
        mu = mu + (b - z[w]) * m
        return (z, a, b, lam, mu), z

    state, zs = jax.lax.scan(step, state, t0 + jnp.arange(n_iters))
    return state, zs


def loss_fn(params, batch, key, n_iters, window):
    keys = jax.random.split(key, batch["u"].shape[0])

    def one(inst, k):
        _, zs = unroll(params, inst, k, n_iters)
        costs = jax.vmap(lambda z: expected_cost(z, inst["u"], inst["w"], inst["C"], inst["emask"]))(zs[-window:])
        return costs.mean() / jnp.maximum(inst["norm"], 1e-9)

    return jax.vmap(one)(batch, keys).mean()


# ---------------------------------------------------------------------------------------------
# v2: hard-cost training (fix 1) and an escape mechanism (fix 2)
#   fix 1: the loss mixes the expected cost with the straight-through cost of the hard readout
#          (argmax of z), so training targets the discrete solution, not the fractional one;
#   fix 2: each (edge, side, value) keeps a small learned memory, updated every round from the
#          local features (a learned analogue of breakout weights on table entries), and a learned
#          per-round noise is added to the local targets (exploration). Both stay value-equivariant.
# ---------------------------------------------------------------------------------------------
N_FEAT = 4


def _mlp(params, prefix, h):
    h = jax.nn.gelu(h @ params[prefix + "w1"] + params[prefix + "b1"])
    h = jax.nn.gelu(h @ params[prefix + "w2"] + params[prefix + "b2"])
    return h @ params[prefix + "w3"] + params[prefix + "b3"]


def init_params_v2(key, hidden=32, mem=4, n_iters=40):
    keys = jax.random.split(key, 6)

    def layer(k, fin, fout, scale=None):
        return jax.random.normal(k, (fin, fout)) * (scale if scale is not None else 1 / np.sqrt(fin))

    fin = 2 * (N_FEAT + mem)
    return {
        "d_w1": layer(keys[0], fin, hidden), "d_b1": jnp.zeros(hidden),
        "d_w2": layer(keys[1], hidden, hidden), "d_b2": jnp.zeros(hidden),
        "d_w3": layer(keys[2], hidden, 1, 0.01), "d_b3": jnp.zeros(1),
        "m_w1": layer(keys[3], fin, hidden), "m_b1": jnp.zeros(hidden),
        "m_w2": layer(keys[4], hidden, hidden), "m_b2": jnp.zeros(hidden),
        "m_w3": layer(keys[5], hidden, mem, 0.1), "m_b3": jnp.zeros(mem),
        "log_alpha": jnp.full((n_iters,), 0.0),
        "log_sigma": jnp.full((n_iters,), np.log(0.05)),
        "log_noise": jnp.array(0.0),
    }


def _side_v2(params, tgt, g, mem, alpha, sigma, eps):
    feats = jnp.stack([tgt, g, tgt - tgt.mean(-1, keepdims=True), g - g.mean(-1, keepdims=True)], -1)
    h = jnp.concatenate([feats, mem], -1)
    h = jnp.concatenate([h, jnp.broadcast_to(h.mean(-2, keepdims=True), h.shape)], -1)
    delta = _mlp(params, "d_", h)[..., 0]
    new_mem = jnp.tanh(mem + _mlp(params, "m_", h))
    return proj_simplex(tgt - alpha * g + delta + sigma * eps), new_mem


def init_state_v2(params, inst, key):
    z, a, b, lam, mu = init_state(params, inst, key)
    mem = params["m_b3"].shape[0]
    zeros = jnp.zeros(a.shape + (mem,))
    return (z, a, b, lam, mu, zeros, zeros)


def unroll_v2(params, inst, key, n_iters, fixed=None, state=None, t0=0):
    u, w, C, emask, deg = inst["u"], inst["w"], inst["C"], inst["emask"], inst["deg"]
    N, d = deg.shape[0], C.shape[-1]
    k_init, k_noise = jax.random.split(key)
    if state is None:
        state = init_state_v2(params, inst, k_init)
    alen = params["log_alpha"].shape[0]
    if fixed is None:
        fixed = -jnp.ones(N, dtype=jnp.int32)
    onehot_fixed = jax.nn.one_hot(jnp.maximum(fixed, 0), d)
    is_fixed = (fixed >= 0)[:, None]
    m = emask[:, None]

    def step(carry, xs):
        t, k = xs
        z, a, b, lam, mu, ma, mb = carry
        ti = jnp.minimum(t, alen - 1)
        alpha, sigma = jnp.exp(params["log_alpha"][ti]), jnp.exp(params["log_sigma"][ti])
        ka, kb = jax.random.split(k)
        a, ma = _side_v2(params, z[u] - lam, jnp.einsum("eab,eb->ea", C, b), ma, alpha, sigma,
                         jax.random.normal(ka, a.shape))
        b, mb = _side_v2(params, z[w] - mu, jnp.einsum("eab,ea->eb", C, a), mb, alpha, sigma,
                         jax.random.normal(kb, b.shape))
        acc = jnp.zeros((N, d)).at[u].add((a + lam) * m).at[w].add((b + mu) * m)
        z = proj_simplex(acc / jnp.maximum(deg, 1.0)[:, None])
        z = jnp.where(deg[:, None] > 0, z, 1.0 / d)
        z = jnp.where(is_fixed, onehot_fixed, z)
        lam = lam + (a - z[u]) * m
        mu = mu + (b - z[w]) * m
        return (z, a, b, lam, mu, ma, mb), z

    keys = jax.random.split(k_noise, n_iters)
    state, zs = jax.lax.scan(step, state, (t0 + jnp.arange(n_iters), keys))
    return state, zs


def loss_fn_v2(params, batch, key, n_iters, window):
    keys = jax.random.split(key, batch["u"].shape[0])

    def one(inst, k):
        _, zs = unroll_v2(params, inst, k, n_iters)
        zs = zs[-window:]
        hard = jax.nn.one_hot(jnp.argmax(zs, -1), zs.shape[-1])
        z_st = zs + jax.lax.stop_gradient(hard - zs)  # straight-through: value = hard cost
        cost = lambda z: expected_cost(z, inst["u"], inst["w"], inst["C"], inst["emask"])
        soft = jax.vmap(cost)(zs).mean()
        hard_c = jax.vmap(cost)(z_st).mean()
        return (0.5 * soft + 0.5 * hard_c) / jnp.maximum(inst["norm"], 1e-9)

    return jax.vmap(one)(batch, keys).mean()


def pack(n, d, E, C, e_max, n_max):
    """Pad one instance to fixed sizes; C normalized by the median positive entry."""
    scale = float(np.median(C[C > 0])) if np.any(C > 0) else 1.0
    m = len(E)
    if m > e_max or n > n_max:
        raise ValueError(f"instance too large: E={m}>{e_max} or n={n}>{n_max}")
    u = np.zeros(e_max, np.int32)
    w = np.zeros(e_max, np.int32)
    Cp = np.zeros((e_max, d, d), np.float32)
    u[:m], w[:m], Cp[:m] = E[:, 0], E[:, 1], C / scale
    emask = np.zeros(e_max, np.float32)
    emask[:m] = 1
    deg = np.zeros(n_max, np.float32)
    np.add.at(deg, E[:, 0], 1)
    np.add.at(deg, E[:, 1], 1)
    norm = float((C / scale).max(axis=(1, 2)).sum())
    return {"u": u, "w": w, "C": Cp, "emask": emask, "deg": deg, "norm": np.float32(norm), "scale": np.float32(scale)}
