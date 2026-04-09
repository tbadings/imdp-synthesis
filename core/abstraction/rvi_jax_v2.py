"""
Optimized JAX robust value iteration for interval MDPs.
Designed for models with millions of states.

Key optimizations over rvi_jax.py:
- Compact arrays: only the M update-states are stored (M <= N)
- Gauss-Seidel batching: each batch sees V updates from prior batches
- V stays on device: only a scalar (delta) transfers to host per iteration
- Per-batch JIT: small, fast-compiling step function
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import argparse
from tqdm import tqdm
from typing import Optional, Tuple
from jaxtyping import Array, Float32, UInt8

from core.abstraction.imdp import IMDP


def RVI_JAX(
    args: argparse.Namespace,
    imdp: IMDP,
    s0: Optional[int] = None,
    max_iterations: int = 1000,
    epsilon: float = 1e-6,
    RND_SWEEPS: bool = False,
    BATCH_SIZE: int = 2000,
    policy_iteration: bool = False,
    **kwargs,
) -> Tuple[Float32[Array, "nr_states"], UInt8[Array, "nr_states"], Float32[Array, "nr_states p"]]:
    """
    Optimized robust value iteration / policy iteration for interval MDPs.

    Supports Gauss-Seidel random sweeps (batched updates where each batch sees
    the latest V from preceding batches). All computation stays on device;
    only a convergence scalar reads back per iteration.

    :param args: Argument namespace (must have .floatprecision, .rvi_device)
    :param imdp: Instance of IMDP class
    :param s0: Initial state for tracking
    :param max_iterations: Maximum number of iterations
    :param epsilon: Convergence threshold
    :param RND_SWEEPS: Random permutation + batching (Gauss-Seidel)
    :param BATCH_SIZE: Number of states per batch
    :param policy_iteration: Use policy iteration instead of value iteration
    :return: Tuple of (values, policy_labels, policy_inputs)
    """

    dtype = args.floatprecision
    device = args.rvi_device

    # ================================================================
    # 1. Build COMPACT padded arrays for update-states only
    # ================================================================
    t0 = time.time()

    states = imdp.states
    N = len(states)

    na_list = np.array([len(imdp.A_id.get(s, [])) for s in states])
    max_actions = int(np.max(na_list)) if np.any(na_list > 0) else 1
    max_succ = max(
        (imdp.S_id[s].shape[1] + 1 for s in imdp.S_id),
        default=1,
    )

    has_actions = na_list > 0
    update_bool = (
        ~imdp.goal_regions & ~imdp.critical_regions
        & (states != imdp.absorbing_state) & has_actions
    )
    stu = states[update_bool]  # states to update (compact → global mapping)
    M = len(stu)

    # Compact arrays: (M, max_actions, max_succ)
    succ_np  = np.full((M, max_actions, max_succ), imdp.absorbing_state, dtype=np.int32)
    lb_np    = np.zeros((M, max_actions, max_succ), dtype=dtype)
    ub_np    = np.zeros((M, max_actions, max_succ), dtype=dtype)
    amask_np = np.zeros((M, max_actions), dtype=np.bool_)

    for i, s in enumerate(stu):
        na  = len(imdp.A_id[s])
        sid = imdp.S_id[s]
        ns  = sid.shape[1]
        succ_np[i, :na, :ns] = sid
        lb_np[i, :na, :ns]   = imdp.P_full[s][:, :, 0]
        ub_np[i, :na, :ns]   = imdp.P_full[s][:, :, 1]
        succ_np[i, :na, max_succ - 1] = imdp.absorbing_state
        lb_np[i, :na, max_succ - 1]   = imdp.P_absorbing[s][:, 0]
        ub_np[i, :na, max_succ - 1]   = imdp.P_absorbing[s][:, 1]
        amask_np[i, :na] = True

    mode_str = "policy iteration" if policy_iteration else "value iteration"
    print(f"=== Optimized robust {mode_str} (JAX v2) ===")
    print(f"- States: {N:,}  |  Update: {M:,}  |  Actions: {max_actions}  |  Successors: {max_succ}")
    print(f"- Total choices: {int(np.sum(na_list)):,}")
    print(f"- Array build: {time.time() - t0:.2f}s")

    if M == 0:
        print("- No states to update.")
        V = np.zeros(imdp.nr_states, dtype=dtype)
        if np.any(imdp.goal_regions):
            V[:-1][imdp.goal_regions] = 1.0
        pl = np.full(imdp.nr_states, -1, dtype=np.int32)
        return V, pl, imdp.actions_inputs[pl]

    # ================================================================
    # 2. Batch setup (Gauss-Seidel or single-batch Jacobi)
    # ================================================================
    if RND_SWEEPS:
        perm = np.random.permutation(M)
        BS = min(BATCH_SIZE, M)
    else:
        perm = np.arange(M)
        BS = M  # single batch = Jacobi

    n_batches = (M + BS - 1) // BS
    padded_M = n_batches * BS
    pad_size = padded_M - M

    # Pad compact arrays so all batches have identical shape (for single JIT)
    if pad_size > 0:
        succ_full  = np.full((padded_M, max_actions, max_succ), imdp.absorbing_state, dtype=np.int32)
        lb_full    = np.zeros((padded_M, max_actions, max_succ), dtype=dtype)
        ub_full    = np.zeros((padded_M, max_actions, max_succ), dtype=dtype)
        amask_full = np.zeros((padded_M, max_actions), dtype=np.bool_)
        succ_full[:M] = succ_np; lb_full[:M] = lb_np; ub_full[:M] = ub_np; amask_full[:M] = amask_np

        # Padded rows: enable one action with all mass on absorbing state (value=0)
        # so they write V[absorbing]=0 instead of -inf
        amask_full[M:, 0] = True
        lb_full[M:, 0, 0] = 1.0
        ub_full[M:, 0, 0] = 1.0

        stu_full = np.full(padded_M, imdp.absorbing_state, dtype=np.int32)
        stu_full[:M] = stu
        perm_full = np.concatenate([perm, np.arange(M, padded_M)])
    else:
        succ_full = succ_np; lb_full = lb_np; ub_full = ub_np; amask_full = amask_np
        stu_full = stu.astype(np.int32)
        perm_full = perm

    # Permute and reshape into (n_batches, BS, ...)
    idx_b_np   = stu_full[perm_full].reshape(n_batches, BS).astype(np.int32)
    succ_b_np  = succ_full[perm_full].reshape(n_batches, BS, max_actions, max_succ)
    lb_b_np    = lb_full[perm_full].reshape(n_batches, BS, max_actions, max_succ)
    ub_b_np    = ub_full[perm_full].reshape(n_batches, BS, max_actions, max_succ)
    amask_b_np = amask_full[perm_full].reshape(n_batches, BS, max_actions)

    del succ_np, lb_np, ub_np, amask_np

    print(f"- Batches: {n_batches} x {BS}" + (" (random sweep)" if RND_SWEEPS else ""))

    # ================================================================
    # 3. Transfer to device (once)
    # ================================================================
    t0 = time.time()
    d_succ_b  = jax.device_put(succ_b_np, device)
    d_lb_b    = jax.device_put(lb_b_np, device)
    d_ub_b    = jax.device_put(ub_b_np, device)
    d_amask_b = jax.device_put(amask_b_np, device)
    d_idx_b   = jax.device_put(idx_b_np, device)
    del succ_b_np, lb_b_np, ub_b_np, amask_b_np, idx_b_np
    print(f"- Device transfer: {time.time() - t0:.2f}s")

    # ================================================================
    # 4. Initialize value function on device
    # ================================================================
    V_np = np.zeros(imdp.nr_states, dtype=dtype)
    if np.any(imdp.goal_regions):
        V_np[:-1][imdp.goal_regions] = 1.0
    V = jax.device_put(V_np, device)
    del V_np

    # ================================================================
    # 5. JIT-compiled per-batch step functions
    # ================================================================
    _bs_range = jax.device_put(jnp.arange(BS, dtype=jnp.int32), device)

    @jax.jit
    def vi_batch_step(V, succ, lb, ub, amask, idx):
        """Bellman backup for one batch. Returns (V_new, best_actions, delta)."""
        sv = V[succ]                                              # (BS, A, S)

        sort_idx = jnp.argsort(sv, axis=-1)
        s_lb  = jnp.take_along_axis(lb, sort_idx, axis=-1)
        s_ub  = jnp.take_along_axis(ub, sort_idx, axis=-1)
        s_val = jnp.take_along_axis(sv, sort_idx, axis=-1)

        budget = 1.0 - jnp.sum(s_lb, axis=-1, keepdims=True)    # (BS, A, 1)
        cap    = s_ub - s_lb
        extra  = jnp.minimum(cap, budget)
        cum    = jnp.cumsum(extra, axis=-1)
        extra  = jnp.minimum(extra, jnp.maximum(0.0, budget - cum + extra))

        qvals  = jnp.sum((s_lb + extra) * s_val, axis=-1)        # (BS, A)
        masked = jnp.where(amask, qvals, -jnp.inf)

        best_v = jnp.max(masked, axis=1)                          # (BS,)
        best_a = jnp.argmax(masked, axis=1).astype(jnp.int32)

        delta_b = jnp.max(jnp.abs(best_v - V[idx]))
        V_new   = V.at[idx].set(best_v)
        return V_new, best_a, delta_b

    @jax.jit
    def pe_batch_step(V, succ, lb, ub, idx, pol):
        """Policy evaluation for one batch. Returns (V_new, delta)."""
        sp = succ[_bs_range, pol]                                  # (BS, S)
        lp = lb[_bs_range, pol]
        up = ub[_bs_range, pol]
        sv = V[sp]

        sort_idx = jnp.argsort(sv, axis=-1)
        s_lb  = jnp.take_along_axis(lp, sort_idx, axis=-1)
        s_ub  = jnp.take_along_axis(up, sort_idx, axis=-1)
        s_val = jnp.take_along_axis(sv, sort_idx, axis=-1)

        budget = 1.0 - jnp.sum(s_lb, axis=-1, keepdims=True)
        cap    = s_ub - s_lb
        extra  = jnp.minimum(cap, budget)
        cum    = jnp.cumsum(extra, axis=-1)
        extra  = jnp.minimum(extra, jnp.maximum(0.0, budget - cum + extra))

        vals    = jnp.sum((s_lb + extra) * s_val, axis=-1)        # (BS,)
        delta_b = jnp.max(jnp.abs(vals - V[idx]))
        V_new   = V.at[idx].set(vals)
        return V_new, delta_b

    # ================================================================
    # 6. JIT warm-up
    # ================================================================
    t0 = time.time()
    _V, _, _d = vi_batch_step(V, d_succ_b[0], d_lb_b[0], d_ub_b[0], d_amask_b[0], d_idx_b[0])
    _d.block_until_ready()
    print(f"- JIT compile (vi_batch_step): {time.time() - t0:.2f}s")
    del _V, _, _d

    # ================================================================
    # 7. Main iteration loop
    # ================================================================
    t_start = time.time()
    policy_batches = [None] * n_batches

    if not policy_iteration:
        # ---- Value iteration (Gauss-Seidel batched) ----
        pbar = tqdm(range(max_iterations), desc="VI")
        for iteration in pbar:
            max_delta = jnp.float32(0.0)

            for b in range(n_batches):
                V, pol_b, delta_b = vi_batch_step(
                    V, d_succ_b[b], d_lb_b[b], d_ub_b[b], d_amask_b[b], d_idx_b[b])
                policy_batches[b] = pol_b
                max_delta = jnp.maximum(max_delta, delta_b)

            d = float(max_delta)
            postfix = {"delta": f"{d:.2e}"}
            if s0 is not None:
                postfix["v[s0]"] = f"{float(V[s0]):.6f}"
            pbar.set_postfix(postfix)

            if d < epsilon:
                pbar.close()
                print(
                    f"Converged after {iteration + 1} iterations "
                    f"({time.time() - t_start:.2f}s)"
                )
                break
    else:
        # ---- Policy iteration ----
        converged_once = False
        pe_inner_max = 100

        # Initial improvement pass
        for b in range(n_batches):
            V, pol_b, _ = vi_batch_step(
                V, d_succ_b[b], d_lb_b[b], d_ub_b[b], d_amask_b[b], d_idx_b[b])
            policy_batches[b] = pol_b

        pbar = tqdm(range(1, max_iterations), desc="PI")
        for iteration in pbar:
            # Policy evaluation
            for _ in range(pe_inner_max):
                max_pe_delta = jnp.float32(0.0)
                for b in range(n_batches):
                    V, d_pe = pe_batch_step(
                        V, d_succ_b[b], d_lb_b[b], d_ub_b[b],
                        d_idx_b[b], policy_batches[b])
                    max_pe_delta = jnp.maximum(max_pe_delta, d_pe)
                if float(max_pe_delta) < epsilon:
                    break

            # Policy improvement
            old_policies = [jax.device_get(pb) for pb in policy_batches]
            for b in range(n_batches):
                V, pol_b, _ = vi_batch_step(
                    V, d_succ_b[b], d_lb_b[b], d_ub_b[b], d_amask_b[b], d_idx_b[b])
                policy_batches[b] = pol_b

            # Check policy convergence
            same = all(
                np.array_equal(jax.device_get(policy_batches[b]), old_policies[b])
                for b in range(n_batches)
            )

            if s0 is not None:
                pbar.set_postfix({"v[s0]": f"{float(V[s0]):.6f}"})

            if same:
                if converged_once:
                    pbar.close()
                    print(
                        f"Converged after {iteration + 1} iterations "
                        f"({time.time() - t_start:.2f}s)"
                    )
                    break
                else:
                    print("Partial convergence. Refining values...")
                    converged_once = True

    # ================================================================
    # 8. Extract policy labels and inputs
    # ================================================================
    V_out = np.asarray(jax.device_get(V))

    # Reconstruct compact policy in original (unpermuted) order
    compact_policy = np.zeros(padded_M, dtype=np.int32)
    for b in range(n_batches):
        compact_policy[perm_full[b * BS:(b + 1) * BS]] = np.asarray(
            jax.device_get(policy_batches[b]))

    policy_labels = np.full(imdp.nr_states, -1, dtype=np.int32)
    for i in range(M):
        policy_labels[stu[i]] = imdp.A_id[stu[i]][compact_policy[i]]

    policy_inputs = imdp.actions_inputs[policy_labels]

    return V_out, policy_labels, policy_inputs
