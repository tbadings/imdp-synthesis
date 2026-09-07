"""Record, replay, and compare the first PPO update in this checkout.

Run in the project's Python environment, from either Mac:
    python diagnose_ppo.py run --out output/diag-m5 --model Drone4D --seed 0
    python diagnose_ppo.py run --out output/diag-m3 --model Drone4D --seed 0
Copy the entire output directory from the other Mac, then:
    python diagnose_ppo.py compare output/diag-m5 output/diag-m3

To isolate optimization from rollout differences, run on BOTH Macs with the SAME
reference directory, then compare the two replay directories:
    python diagnose_ppo.py replay output/diag-m5 --out output/replay-m5

Normal benchmark/PPO CLI options are accepted by `run`. Only one rollout batch is
trained, regardless of total_timesteps. Production code is not modified. Detailed
probes split compilation into stages, which can itself affect rounding; a separate
call to the original train_ppo saves its final parameters as a production baseline.
The diagnostic PPO equations below mirror core/rl/ppo.py; source hashes in each
record help detect changes to either implementation between runs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
SCHEMA = 1


def command_output(*command):
    try:
        return subprocess.check_output(
            command, cwd=ROOT, stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return str(value)


def runtime_metadata():
    import jax

    sources = [Path(__file__), ROOT / "core/jax_config.py", ROOT / "core/options.py",
               ROOT / "core/abstraction/model.py"]
    sources += sorted((ROOT / "core/rl").glob("*.py"))
    sources += sorted((ROOT / "benchmarks").rglob("*.py"))
    env_names = {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED",
                 "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_brand": command_output("sysctl", "-n", "machdep.cpu.brand_string"),
        "rosetta": command_output("sysctl", "-n", "sysctl.proc_translated"),
        "backend": jax.default_backend(),
        "devices": [{"platform": d.platform, "kind": d.device_kind, "id": d.id}
                    for d in jax.devices()],
        "jax_config": json_value(jax.config.values),
        "environment": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith(("JAX_", "XLA_")) or k in env_names},
        "packages": dict(sorted((d.metadata["Name"], d.version)
                                for d in package_metadata.distributions()
                                if d.metadata["Name"])),
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }


def array_key(prefix, path):
    # JAX DictKey, GetAttrKey (namedtuple), and SequenceKey, without relying on repr.
    parts = []
    for key in path:
        parts.append(str(key.key if hasattr(key, "key") else
                         key.name if hasattr(key, "name") else key.idx))
    return "/".join([prefix, *parts])


def save_tree(arrays, prefix, tree):
    import jax

    leaves, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(tree))
    for path, value in leaves:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Cannot save object array at {prefix}")
        arrays[array_key(prefix, path)] = array


def load_tree(arrays, prefix, template):
    import jax

    def restore(path, expected):
        key = array_key(prefix, path)
        value = arrays[key]
        if value.shape != np.shape(expected) or value.dtype != np.asarray(expected).dtype:
            raise ValueError(f"Replay shape/dtype mismatch for {key}: {value.shape}, {value.dtype}")
        return jax.device_put(value)

    return jax.tree_util.tree_map_with_path(restore, template)


def save_numeric_inputs(arrays, prefix, values):
    """Record model/environment constants as well as resolved settings in metadata."""
    for key, value in sorted(values.items()):
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            save_numeric_inputs(arrays, name, value)
        elif isinstance(value, (list, tuple, int, float, bool, np.ndarray, np.generic)) or hasattr(value, "dtype"):
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                continue
            if array.dtype.kind in "biufc":
                arrays[name] = array


def initialize(network, cfg, seed, obs_dim):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState

    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    params = network.init(init_key, jnp.zeros((1, obs_dim), dtype=jnp.float32))
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(cfg.learning_rate, eps=cfg.adam_eps))
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx), key


def collect_rollout(state, env, cfg, key, arrays):
    import jax
    import jax.numpy as jnp
    from core.rl.env import EnvState, _sample_safe_state, _env_step_jnp
    from core.rl.policy import gaussian_sample, gaussian_log_prob
    from core.rl.ppo import Transition

    key, env_key = jax.random.split(key)
    env_keys = jax.random.split(env_key, cfg.n_envs)
    observations = jax.vmap(lambda k: _sample_safe_state(k, env))(env_keys)
    env_state = EnvState(observations, jnp.zeros((cfg.n_envs,), dtype=jnp.int32),
                         jax.vmap(env.distance_to_goal)(observations))
    save_tree(arrays, "03_initial_env", env_state)
    save_tree(arrays, "03_initial_env/key", key)
    forward = jax.jit(state.apply_fn)(state.params, observations)
    save_tree(arrays, "04_forward_probe", dict(zip(("mean", "log_std", "value"), forward)))

    def step(carry, _):
        t_state, e_state, k = carry
        k, action_key, step_key = jax.random.split(k, 3)
        mean, log_std, value = t_state.apply_fn(t_state.params, e_state.state)
        action = gaussian_sample(action_key, mean, log_std)
        log_prob = gaussian_log_prob(action, mean, log_std)
        _, next_env, reward, done, info = jax.vmap(
            lambda rk, s, a: _env_step_jnp(rk, s, a, env)
        )(jax.random.split(step_key, cfg.n_envs), e_state, action)
        _, _, next_value = t_state.apply_fn(t_state.params, info["next_state"])
        transition = Transition(e_state.state, action, value, reward, done,
                                info["terminated"], next_value, log_prob)
        return (t_state, next_env, k), transition

    # Parameters are dynamic arguments, as in production, rather than JIT constants.
    def rollout(t_state, e_state, k):
        return jax.lax.scan(step, (t_state, e_state, k), None, length=cfg.n_steps)

    (_, final_env, key), trajectory = jax.jit(rollout)(state, env_state, key)
    save_tree(arrays, "05_rollout", trajectory)
    save_tree(arrays, "05_rollout_final_env", final_env)
    return trajectory, key


def prepare_batch(trajectory, cfg, arrays):
    import jax
    import jax.numpy as jnp
    from core.rl.ppo import FlatTransition

    @jax.jit
    def gae(traj):
        def step(advantage, transition):
            done = transition.done.astype(jnp.float32)
            next_value = jnp.where(transition.terminated, 0.0, transition.next_value)
            delta = transition.reward + cfg.gamma * next_value - transition.value
            advantage = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done) * advantage
            return advantage, (advantage, advantage + transition.value)

        return jax.lax.scan(step, jnp.zeros_like(traj.value[0]), traj, reverse=True)[1]

    advantages, targets = gae(trajectory)
    save_tree(arrays, "06_gae", {"advantage": advantages, "target": targets})
    size = cfg.n_envs * cfg.n_steps
    flat = FlatTransition(trajectory.obs.reshape(size, -1), trajectory.action.reshape(size, -1),
                          trajectory.value.reshape(size), trajectory.log_prob.reshape(size),
                          advantages.reshape(size), targets.reshape(size))
    save_tree(arrays, "07_optimizer_input/batch", flat)
    return flat


def optimize(state, flat, key, cfg, arrays):
    """Both record and replay use these same compiled optimizer probes."""
    import jax
    import jax.numpy as jnp
    from core.rl.policy import gaussian_log_prob, gaussian_entropy

    size = cfg.n_envs * cfg.n_steps
    count = max(1, size // cfg.rl_batch_size)
    mb_size = size // count

    def minibatches(batch, k):
        k, perm_key = jax.random.split(k)
        perm = jax.random.permutation(perm_key, size)
        shuffled = jax.tree_util.tree_map(lambda x: x[perm], batch)
        batches = jax.tree_util.tree_map(
            lambda x: x.reshape((count, mb_size) + x.shape[1:]), shuffled)
        return batches, k, perm

    def loss(params, mb):
        mean, log_std, val = state.apply_fn(params, mb.obs)
        lp = gaussian_log_prob(mb.action, mean, log_std)
        entropy = gaussian_entropy(log_std)
        ratio = jnp.exp(lp - mb.log_prob)
        norm_adv = (mb.advantage - jnp.mean(mb.advantage)) / (jnp.std(mb.advantage) + 1e-8)
        actor_loss1 = -norm_adv * ratio
        actor_loss2 = -norm_adv * jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
        actor_loss = jnp.mean(jnp.maximum(actor_loss1, actor_loss2))
        v_clipped = mb.value + jnp.clip(val - mb.value, -cfg.clip_eps, cfg.clip_eps)
        v_loss1 = jnp.square(val - mb.target)
        v_loss2 = jnp.square(v_clipped - mb.target)
        critic_loss = 0.5 * jnp.mean(jnp.maximum(v_loss1, v_loss2))
        mean_entropy = jnp.mean(entropy)
        ent_loss = -cfg.ent_coef * mean_entropy
        total = actor_loss + cfg.vf_coef * critic_loss + ent_loss
        return total, {"total": total, "actor": actor_loss, "critic": critic_loss,
                       "entropy": mean_entropy, "ratio": ratio, "normalized_advantage": norm_adv}

    @jax.jit
    def first_update(t_state, mb):
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(t_state.params, mb)
        return t_state.apply_gradients(grads=grads), metrics, grads

    @jax.jit
    def full_update(t_state, batch, k):
        def epoch(carry, _):
            ts, epoch_key = carry
            batches, epoch_key, _ = minibatches(batch, epoch_key)

            def update(ts, mb):
                grads = jax.grad(lambda p: loss(p, mb)[0])(ts.params)
                return ts.apply_gradients(grads=grads), None

            ts, _ = jax.lax.scan(update, ts, batches)
            return (ts, epoch_key), None

        return jax.lax.scan(epoch, (t_state, k), None, length=cfg.update_epochs)[0]

    save_tree(arrays, "07_optimizer_input/key", key)
    batches, _, perm = jax.jit(minibatches)(flat, key)
    mb = jax.tree_util.tree_map(lambda x: x[0], batches)
    save_tree(arrays, "08_first_minibatch/permutation", perm)
    save_tree(arrays, "08_first_minibatch/data", mb)
    first, metrics, grads = first_update(state, mb)
    save_tree(arrays, "09_first_loss", metrics)
    save_tree(arrays, "10_first_gradients", grads)
    save_tree(arrays, "11_first_update/params", first.params)
    save_tree(arrays, "11_first_update/opt_state", first.opt_state)
    final, final_key = full_update(state, flat, key)
    save_tree(arrays, "12_staged_update/params", final.params)
    save_tree(arrays, "12_staged_update/opt_state", final.opt_state)
    save_tree(arrays, "12_staged_update/key", final_key)
    return final


def configure(project_argv):
    import jax
    from core.jax_config import configure_jax
    from core.options import parse_arguments

    args = parse_arguments(project_argv)
    configure_jax(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)
    args.cwd = str(ROOT)
    args.root_dir = ROOT
    return args


def write_record(out, metadata, arrays):
    nonfinite = [k for k, a in arrays.items() if not np.isfinite(a).all()]
    metadata["nonfinite_arrays"] = nonfinite
    metadata["array_count"] = len(arrays)
    np.savez_compressed(out / "arrays.npz", **arrays)
    (out / "metadata.json").write_text(json.dumps(json_value(metadata), indent=2) + "\n")
    print(f"Saved {len(arrays)} arrays to {out}", flush=True)
    if nonfinite:
        print(f"WARNING: {len(nonfinite)} arrays contain NaN/Inf; see metadata.json")


def read_metadata(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported diagnostic schema in {directory}")
    return metadata


def record_run(options, project_argv):
    import jax
    import jax.numpy as jnp
    import benchmarks
    from core.rl.config import resolve_rl_config
    from core.rl.env import BenchmarkEnv
    from core.rl.policy import ActorCritic
    from core.rl.ppo import train_ppo

    args = configure(project_argv)
    if not args.model:
        raise ValueError("Specify a benchmark, e.g. --model Drone4D")
    model = benchmarks.create_model(args)
    cfg = resolve_rl_config(model, args)
    cfg = replace(cfg, total_timesteps=cfg.n_envs * cfg.n_steps)
    count = max(1, cfg.total_timesteps // cfg.rl_batch_size)
    if cfg.total_timesteps % count:
        raise ValueError("Rollout size is not divisible by the production minibatch count")
    env = BenchmarkEnv(model, cfg)
    network = ActorCritic(action_dim=len(env.u_min), pi_arch=tuple(cfg.pi_arch), vf_arch=tuple(cfg.vf_arch))
    metadata = {"schema": SCHEMA, "mode": "run", "label": options.label,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "project_argv": project_argv,
                "experiment": {"model": args.model, "model_version": args.model_version,
                               "noise_distr": args.noise_distr, "seed": args.seed,
                               "obs_dim": model.n, "action_dim": len(env.u_min),
                               "cfg": asdict(cfg)},
                "runtime": runtime_metadata()}
    options.out.mkdir(parents=True, exist_ok=False)
    arrays = {}
    save_numeric_inputs(arrays, "00_inputs/model", vars(model))
    save_numeric_inputs(arrays, "00_inputs/env", vars(env))
    # Independent probes do not consume the training random stream.
    probe_key = jax.random.PRNGKey(args.seed)
    save_tree(arrays, "01_rng_probe", {
        "bits": jax.random.bits(probe_key, (4096,), dtype=jnp.uint32),
        "uniform": jax.random.uniform(probe_key, (4096,)),
        "normal": jax.random.normal(probe_key, (4096,)),
    })
    state, key = initialize(network, cfg, args.seed, model.n)
    save_tree(arrays, "02_initial/params", state.params)
    save_tree(arrays, "02_initial/opt_state", state.opt_state)
    print(f"Recording {cfg.total_timesteps} environment steps and {count * cfg.update_epochs} Adam updates "
          f"on {jax.default_backend()} ({metadata['runtime']['cpu_brand'] or platform.machine()}).", flush=True)
    trajectory, key = collect_rollout(state, env, cfg, key, arrays)
    flat = prepare_batch(trajectory, cfg, arrays)
    print("Recording first minibatch and full staged optimization...", flush=True)
    staged = optimize(state, flat, key, cfg, arrays)
    print("Running original train_ppo for one rollout batch...", flush=True)
    _, production_params = train_ppo(env, cfg, seed=args.seed)
    save_tree(arrays, "13_production_update/params", production_params)
    deltas = jax.tree_util.tree_map(lambda a, b: np.max(np.abs(np.asarray(a, dtype=np.float64) -
                                                            np.asarray(b, dtype=np.float64))),
                                    staged.params, production_params)
    metadata["staged_vs_production_max_abs"] = float(max(jax.tree_util.tree_leaves(deltas)))
    print(f"Staged vs production max parameter difference: {metadata['staged_vs_production_max_abs']:.3e}")
    write_record(options.out, metadata, arrays)


def replay_run(options):
    import jax
    import jax.numpy as jnp
    from core.rl.config import RLConfig
    from core.rl.policy import ActorCritic
    from core.rl.ppo import FlatTransition

    reference = read_metadata(options.reference)
    if reference["mode"] != "run":
        raise ValueError("Replay requires a directory created by 'run'")
    configure(reference["project_argv"])
    experiment = reference["experiment"]
    cfg = RLConfig(**experiment["cfg"])
    network = ActorCritic(action_dim=experiment["action_dim"], pi_arch=tuple(cfg.pi_arch), vf_arch=tuple(cfg.vf_arch))
    state, _ = initialize(network, cfg, experiment["seed"], experiment["obs_dim"])
    arrays = {}
    with np.load(options.reference / "arrays.npz", allow_pickle=False) as saved:
        state = state.replace(params=load_tree(saved, "02_initial/params", state.params),
                              opt_state=load_tree(saved, "02_initial/opt_state", state.opt_state))
        # Preserve recorded dtypes; detect silent dtype truncation on this runtime.
        def device_array(key):
            array = saved[key]
            result = jnp.asarray(array)
            if result.dtype != array.dtype:
                raise ValueError(f"Replay would change {key} from {array.dtype} to {result.dtype}; match JAX precision settings")
            return result

        flat = FlatTransition(*(device_array(f"07_optimizer_input/batch/{name}")
                                for name in FlatTransition._fields))
        key = device_array("07_optimizer_input/key")
    metadata = {"schema": SCHEMA, "mode": "replay", "label": options.label,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": experiment, "runtime": runtime_metadata(),
                "reference_arrays_sha256": hashlib.sha256((options.reference / "arrays.npz").read_bytes()).hexdigest()}
    options.out.mkdir(parents=True, exist_ok=False)
    save_tree(arrays, "02_initial/params", state.params)
    save_tree(arrays, "02_initial/opt_state", state.opt_state)
    save_tree(arrays, "07_optimizer_input/batch", flat)
    print("Replaying the saved parameters, optimizer state, batch, and shuffle key...", flush=True)
    optimize(state, flat, key, cfg, arrays)
    write_record(options.out, metadata, arrays)


def compare_array(a, b, atol, rtol):
    """Bitwise equality and numeric errors; nonfinite values are always flagged."""
    result = {"shape_a": list(a.shape), "shape_b": list(b.shape),
              "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}
    if a.shape != b.shape or a.dtype != b.dtype:
        return {**result, "compatible": False, "exact": False}
    byte_a = np.ascontiguousarray(a).reshape(-1).view(np.uint8).reshape(a.size, a.itemsize)
    byte_b = np.ascontiguousarray(b).reshape(-1).view(np.uint8).reshape(b.size, b.itemsize)
    different = np.any(byte_a != byte_b, axis=1).reshape(a.shape)
    finite = np.isfinite(a) & np.isfinite(b)
    # Float64 avoids subtracting in the recorded float32 precision or overflowing integers.
    aa, bb = a.astype(np.complex128 if a.dtype.kind == "c" else np.float64), b.astype(np.complex128 if b.dtype.kind == "c" else np.float64)
    if a.dtype.kind in "biu":
        # Preserve single-unit differences in int64/uint64 values above 2**53.
        error = np.fromiter((abs(int(x) - int(y)) for x, y in zip(a.flat, b.flat)),
                            dtype=np.float64, count=a.size)
    else:
        error = np.abs(aa[finite] - bb[finite])
    scale = np.maximum(np.abs(aa[finite]), np.abs(bb[finite]))
    relative = error / np.maximum(scale, np.finfo(np.float64).tiny)
    close = np.isclose(aa, bb, atol=atol, rtol=rtol) & finite if a.dtype.kind in "fc" else ~different
    indices = np.flatnonzero(different)
    first = list(map(int, np.unravel_index(int(indices[0]), a.shape))) if indices.size else None
    max_abs = float(np.max(error, initial=0))
    rms = float(max_abs * np.sqrt(np.mean((error / max_abs) ** 2))) if max_abs and error.size else 0.0
    return {**result, "compatible": True, "exact": not bool(different.any()),
            "different_elements": int(different.sum()), "elements": int(a.size),
            "outside_tolerance": int((~close).sum()), "nonfinite_elements": int((~finite).sum()),
            "max_abs": max_abs, "max_relative": float(np.max(relative, initial=0)),
            "rms": rms, "first_different_index": first}


def metadata_differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(a.keys() | b.keys()):
            path = f"{prefix}/{key}" if prefix else key
            if key not in a or key not in b:
                rows.append({"path": path, "a": a.get(key), "b": b.get(key)})
            else:
                rows.extend(metadata_differences(a[key], b[key], path))
        return rows
    return [] if a == b else [{"path": prefix, "a": a, "b": b}]


def compare_runs(options):
    meta_a, meta_b = read_metadata(options.a), read_metadata(options.b)
    # Labels, timestamps and summary diagnostics are not experimental settings.
    fields = ("mode", "experiment", "runtime", "reference_arrays_sha256")
    differences = metadata_differences({k: meta_a.get(k) for k in fields}, {k: meta_b.get(k) for k in fields})
    print(f"Comparing {options.a} and {options.b}")
    if differences:
        print(f"Metadata differences ({len(differences)}; use --report to save all details):")
        for row in differences[:25]:
            print(f"  {row['path']}: {str(row['a'])[:100]} -> {str(row['b'])[:100]}")
    confounds = [d for d in differences if d["path"].startswith(
        ("mode", "experiment", "runtime/source_sha256", "reference_arrays_sha256",
         "runtime/packages", "runtime/jax_config", "runtime/environment", "runtime/python",
         "runtime/backend", "runtime/platform"))]
    if confounds:
        print("WARNING: software, configuration, or replay reference differ; this is not a controlled hardware comparison.")
    rows = {}
    with np.load(options.a / "arrays.npz", allow_pickle=False) as aa, np.load(options.b / "arrays.npz", allow_pickle=False) as bb:
        for name in sorted(set(aa.files) | set(bb.files)):
            if name not in aa or name not in bb:
                rows[name] = {"compatible": False, "exact": False, "missing_from": "a" if name not in aa else "b"}
            else:
                rows[name] = compare_array(aa[name], bb[name], options.atol, options.rtol)
    changed = {k: v for k, v in rows.items() if not v["exact"] or v.get("nonfinite_elements", 0)}
    print("\nStage                              changed/arrays       max |difference|")
    for stage in sorted({k.split('/')[0] for k in rows}):
        group = {k: v for k, v in rows.items() if k.split('/')[0] == stage}
        count = sum(k in changed for k in group)
        maximum = max((v.get("max_abs", 0) for v in group.values()), default=0)
        print(f"{stage:34} {count:5}/{len(group):<5}        {maximum:.3e}")
    if changed:
        first = next(iter(changed))
        print(f"\nFirst differing/invalid checkpoint in stage order: {first}")
        print("Largest differences (use --report for every array, relative error, RMS, and tolerance counts):")
        for name, row in sorted(changed.items(), key=lambda item: item[1].get("max_abs", float('inf')), reverse=True)[:options.top]:
            print(f"  {name}: max_abs={row.get('max_abs', 'incompatible')}, first_index={row.get('first_different_index')}")
        rollout = {k: v["first_different_index"] for k, v in changed.items()
                   if k.startswith("05_rollout/") and v.get("first_different_index") is not None}
        if rollout:
            print("First differing rollout indices [time step, environment, optional coordinate], zero-based:")
            for name, index in rollout.items():
                print(f"  {name}: {index}")
    else:
        print("\nAll recorded arrays are bitwise identical and finite.")
    print("Stage order includes independent probes; it does not prove which production operation diverged first.")
    report = {"a": str(options.a), "b": str(options.b), "atol": options.atol, "rtol": options.rtol,
              "metadata_differences": differences, "changed_arrays": len(changed), "arrays": rows}
    if options.report:
        # Refuse accidental replacement, just as record/replay do for output directories.
        with options.report.open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        print(f"Report: {options.report}")
    return 1 if changed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Record one PPO batch; accepts the normal benchmark/PPO options", allow_abbrev=False)
    run.add_argument("--out", type=Path, required=True, help="New output directory")
    run.add_argument("--label", default="", help="Optional machine/run label")
    replay = commands.add_parser("replay", help="Optimize the exact saved data from a run", allow_abbrev=False)
    replay.add_argument("reference", type=Path)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--label", default="")
    compare = commands.add_parser("compare", help="Compare saved arrays; needs only NumPy", allow_abbrev=False)
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)
    compare.add_argument("--atol", type=float, default=1e-7)
    compare.add_argument("--rtol", type=float, default=1e-6)
    compare.add_argument("--top", type=int, default=12)
    compare.add_argument("--report", type=Path, help="Write the full comparison to a new JSON file")
    options, project_argv = parser.parse_known_args(argv)
    if options.command != "run" and project_argv:
        parser.error(f"Unrecognized arguments: {' '.join(project_argv)}")
    logging.basicConfig(level=logging.WARNING)
    try:
        if options.command == "run":
            record_run(options, project_argv)
        elif options.command == "replay":
            replay_run(options)
        else:
            if options.atol < 0 or options.rtol < 0 or options.top < 1 or not np.isfinite([options.atol, options.rtol]).all():
                raise ValueError("Tolerances must be finite and nonnegative, and --top must be positive")
            return compare_runs(options)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
