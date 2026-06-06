import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import dysts

BENCHMARK_SYSTEMS = [
    "Lorenz63",
    "Rossler",
    "Chua",
    "Lorenz96",
]


@dataclass
class TrajectoryMetadata:
    system_name: str
    state_dim: int
    seed: int
    num_trajectories: int
    benchmark_split_mode: str
    integration_dt: float
    lyapunov_time: float
    target_pts_per_lyapunov: int
    downsample_stride: int
    effective_dt: float
    warmup_steps_raw: int
    kept_steps_raw: int
    train_points: int
    test_points: int
    ic_sampling_method: str
    ic_sampling_fallback: str | None
    reference_traj_length: int
    reference_traj_transient: int


class NativeLorenz96:
    def __init__(self, dimension, forcing=8.0):
        if int(dimension) < 4:
            raise ValueError(f"Lorenz96 requires dimension >= 4, got {dimension}.")
        self.dimension = int(dimension)
        self.forcing = float(forcing)
        self.ic = self.forcing * np.ones(self.dimension, dtype=np.float64)
        self.ic[self.dimension // 2] += 0.01

    def rhs(self, state, t):
        x = np.asarray(state, dtype=np.float64)
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + self.forcing


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 4-system chaotic benchmark from dysts.")
    parser.add_argument("--save_dir", type=str, default="./data_dysts_benchmark")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_trajectories", type=int, default=20, help="Number of unseen-initial-condition test trajectories.")
    parser.add_argument("--integration_dt", type=float, default=0.001)
    parser.add_argument("--target_pts_per_lyapunov", type=int, default=30)
    parser.add_argument("--warmup_lyapunov_times", type=float, default=5.0)
    parser.add_argument("--kept_lyapunov_times", type=float, default=80.0)
    parser.add_argument("--train_points", type=int, default=30000)
    parser.add_argument("--test_points", type=int, default=None)
    parser.add_argument("--reference_traj_length", type=int, default=4096)
    parser.add_argument("--reference_traj_transient", type=int, default=500)
    parser.add_argument(
        "--lorenz96_lyapunov_time",
        type=float,
        default=None,
        help="Optional manual Lyapunov time override for native Lorenz96 generation. "
        "If omitted, it will be estimated numerically for the requested dimension.",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=BENCHMARK_SYSTEMS,
        help="Subset of systems to generate. Defaults to the full 4-system benchmark.",
    )
    parser.add_argument("--lorenz96_dim", type=int, default=20)
    return parser.parse_args()


def _infer_state_dim(system):
    for attr_name in ("ic", "initial_state"):
        value = getattr(system, attr_name, None)
        if value is not None:
            value = np.asarray(value)
            if value.ndim == 1 and value.size > 0:
                return int(value.size)
    for attr_name in ("dimension", "embedding_dimension"):
        value = getattr(system, attr_name, None)
        if value is not None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    raise AttributeError(
        f"Could not infer the state dimension for system {system.__class__.__name__}. "
        "Please inspect the dysts object and add a system-specific rule."
    )


def _build_system(system_name, lorenz96_dim):
    if system_name == "Lorenz96":
        return NativeLorenz96(dimension=lorenz96_dim)

    import dysts.flows as flows

    return getattr(flows, system_name)()


def _estimate_lorenz96_lyapunov_time(system, integration_dt, warmup_steps=5000, estimate_steps=15000):
    base_ic = _resolve_base_initial_state(system)
    base_traj = _rk4_integrate(system, base_ic, integration_dt, warmup_steps + estimate_steps)
    anchor = base_traj[warmup_steps]

    perturb = np.zeros_like(anchor)
    perturb[0] = 1e-8
    perturbed_anchor = anchor + perturb

    ref = _rk4_integrate(system, anchor, integration_dt, estimate_steps)
    pert = _rk4_integrate(system, perturbed_anchor, integration_dt, estimate_steps)
    distances = np.linalg.norm(pert - ref, axis=1)
    distances = np.maximum(distances, 1e-16)

    valid = np.where((distances > 1e-10) & (distances < 1.0))[0]
    if len(valid) < max(100, estimate_steps // 20):
        raise RuntimeError(
            "Failed to estimate Lorenz96 Lyapunov time: not enough divergence samples in the linear regime."
        )

    log_dist = np.log(distances[valid])
    time_axis = valid.astype(np.float64) * integration_dt
    slope, _ = np.polyfit(time_axis, log_dist, 1)
    if not np.isfinite(slope) or slope <= 0:
        raise RuntimeError(f"Failed to estimate Lorenz96 Lyapunov time: non-positive growth rate {slope}.")
    return 1.0 / float(slope)


def _resolve_lyapunov_time(system, integration_dt=None, lorenz96_override=None):
    if isinstance(system, NativeLorenz96):
        if lorenz96_override is not None:
            return float(lorenz96_override)
        cached = getattr(system, "_estimated_lyapunov_time", None)
        if cached is not None:
            return float(cached)
        if integration_dt is None:
            raise ValueError("integration_dt is required to estimate Lyapunov time for NativeLorenz96.")
        estimated = _estimate_lorenz96_lyapunov_time(system, integration_dt)
        system._estimated_lyapunov_time = estimated
        return float(estimated)

    candidates = [
        "lyapunov_time",
        "lyapunov_t",
        "lyapunov_timescale",
        "tau_lyap",
        "tau",
    ]
    for name in candidates:
        value = getattr(system, name, None)
        if value is not None:
            value = float(np.asarray(value).reshape(-1)[0])
            if np.isfinite(value) and value > 0:
                return value

    mle_candidates = [
        "maximum_lyapunov_estimated",
        "maximum_lyapunov",
        "lyapunov_max",
    ]
    for name in mle_candidates:
        value = getattr(system, name, None)
        if value is not None:
            value = float(np.asarray(value).reshape(-1)[0])
            if np.isfinite(value) and value > 0:
                return 1.0 / value

    raise AttributeError(
        f"Could not determine Lyapunov time for system {system.__class__.__name__}. "
        "Please inspect the dysts object and add the appropriate attribute name."
    )


def _resolve_base_initial_state(system):
    base_ic = getattr(system, "ic", None)
    if base_ic is None:
        base_ic = getattr(system, "initial_state", None)
    if base_ic is None:
        if hasattr(system, "dimension"):
            dim = int(getattr(system, "dimension"))
        elif hasattr(system, "embedding_dimension"):
            dim = int(getattr(system, "embedding_dimension"))
        else:
            raise AttributeError(
                f"Could not infer an initial state template for {system.__class__.__name__}. "
                "Please add a system-specific initialization rule."
            )
        base_ic = np.ones(dim, dtype=np.float64)
    return np.asarray(base_ic, dtype=np.float64).reshape(-1)


def _resolve_initial_state_gaussian(system, rng):
    base_ic = _resolve_base_initial_state(system)
    scale = np.maximum(np.abs(base_ic), 1.0)
    perturbation = rng.normal(loc=0.0, scale=0.05 * scale, size=base_ic.shape)
    return base_ic + perturbation


def _build_attractor_sampler(args, seed):
    from dysts.sampling import OnAttractorInitCondSampler

    return OnAttractorInitCondSampler(
        random_seed=seed,
        reference_traj_length=args.reference_traj_length,
        reference_traj_transient=args.reference_traj_transient,
    )


def _build_reference_state_pool(system, integration_dt, reference_traj_transient, reference_traj_length):
    base_ic = _resolve_base_initial_state(system)
    total_steps = reference_traj_transient + reference_traj_length
    reference_traj = _rk4_integrate(system, base_ic, integration_dt, total_steps)
    state_pool = reference_traj[reference_traj_transient:]
    if state_pool.ndim != 2 or len(state_pool) == 0:
        raise RuntimeError(
            f"Reference trajectory sampling failed for {system.__class__.__name__}; got pool shape {state_pool.shape}."
        )
    if not np.isfinite(state_pool).all():
        raise RuntimeError(
            f"Reference trajectory for {system.__class__.__name__} contains non-finite values; "
            "cannot use it as the attractor state pool."
        )
    return state_pool.astype(np.float64, copy=False)


def _sample_initial_state_from_reference(state_pool, rng):
    sampled_idx = int(rng.integers(0, len(state_pool)))
    return state_pool[sampled_idx].astype(np.float64, copy=True)


def _sample_initial_state_with_sampler(system, sampler, rng):
    base_ic = getattr(system, "ic", None)
    if base_ic is None:
        base_ic = getattr(system, "initial_state", None)
    if base_ic is None:
        sampled_ic = None
    else:
        base_ic = np.asarray(base_ic, dtype=np.float64)
        sampled_ic = sampler(base_ic, system)
    if sampled_ic is None:
        return _resolve_initial_state_gaussian(system, rng), "gaussian_perturbation"
    sampled_ic = np.asarray(sampled_ic, dtype=np.float64)
    if not np.isfinite(sampled_ic).all():
        return _resolve_initial_state_gaussian(system, rng), "gaussian_perturbation"
    return sampled_ic, None


def _rk4_integrate(system, initial_state, dt, num_steps):
    states = np.empty((num_steps, initial_state.shape[0]), dtype=np.float64)
    states[0] = initial_state
    current = initial_state.astype(np.float64, copy=True)

    for step in range(1, num_steps):
        k1 = np.asarray(system.rhs(current, 0.0), dtype=np.float64)
        k2 = np.asarray(system.rhs(current + 0.5 * dt * k1, 0.0), dtype=np.float64)
        k3 = np.asarray(system.rhs(current + 0.5 * dt * k2, 0.0), dtype=np.float64)
        k4 = np.asarray(system.rhs(current + dt * k3, 0.0), dtype=np.float64)
        current = current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states[step] = current

    return states


def _downsample_stride(lyapunov_time, dt, target_pts_per_lyapunov):
    stride = int(round((lyapunov_time / target_pts_per_lyapunov) / dt))
    return max(1, stride)


def _raw_steps_for_downsampled_points(num_points, stride):
    if num_points <= 0:
        raise ValueError(f"Number of downsampled points must be positive, got {num_points}.")
    return (num_points - 1) * stride + 1


def _generate_downsampled_trajectory(system, ic, integration_dt, warmup_steps_raw, kept_steps_raw, stride):
    raw_traj = _rk4_integrate(system, ic, integration_dt, warmup_steps_raw + kept_steps_raw)
    kept = raw_traj[warmup_steps_raw:]
    return kept[::stride].astype(np.float32)


def generate_system(system_name, args):
    rng = np.random.default_rng(args.seed)
    system = _build_system(system_name, args.lorenz96_dim)
    state_dim = _infer_state_dim(system)
    if system_name == "Lorenz96" and state_dim != args.lorenz96_dim:
        raise ValueError(
            f"Lorenz96 was requested with dim={args.lorenz96_dim}, but the constructed system still has "
            f"state_dim={state_dim}."
        )

    reference_state_pool = None
    sampler = None
    if system_name == "Lorenz96":
        reference_state_pool = _build_reference_state_pool(
            system,
            args.integration_dt,
            args.reference_traj_transient,
            args.reference_traj_length,
        )
    else:
        sampler = _build_attractor_sampler(args, int(rng.integers(0, 2**32 - 1)))

    lyapunov_time = _resolve_lyapunov_time(
        system,
        integration_dt=args.integration_dt,
        lorenz96_override=args.lorenz96_lyapunov_time,
    )
    stride = _downsample_stride(lyapunov_time, args.integration_dt, args.target_pts_per_lyapunov)
    effective_dt = args.integration_dt * stride

    warmup_steps_raw = int(round((args.warmup_lyapunov_times * lyapunov_time) / args.integration_dt))
    default_test_points = max(1, int(round(args.kept_lyapunov_times * args.target_pts_per_lyapunov)))
    test_points = default_test_points if args.test_points is None else args.test_points
    train_kept_steps_raw = _raw_steps_for_downsampled_points(args.train_points, stride)
    test_kept_steps_raw = _raw_steps_for_downsampled_points(test_points, stride)

    train_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    if system_name == "Lorenz96":
        train_ic = _sample_initial_state_from_reference(reference_state_pool, train_rng)
        train_fallback_method = None
    else:
        train_ic, train_fallback_method = _sample_initial_state_with_sampler(system, sampler, train_rng)
    train_trajectory = _generate_downsampled_trajectory(
        system,
        train_ic,
        args.integration_dt,
        warmup_steps_raw,
        train_kept_steps_raw,
        stride,
    )[: args.train_points]

    test_trajectories = []
    fallback_used = train_fallback_method is not None
    for _ in range(args.num_trajectories):
        traj_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        if system_name == "Lorenz96":
            ic = _sample_initial_state_from_reference(reference_state_pool, traj_rng)
            fallback_method = None
        else:
            ic, fallback_method = _sample_initial_state_with_sampler(system, sampler, traj_rng)
        fallback_used = fallback_used or (fallback_method is not None)
        downsampled = _generate_downsampled_trajectory(
            system,
            ic,
            args.integration_dt,
            warmup_steps_raw,
            test_kept_steps_raw,
            stride,
        )
        test_trajectories.append(downsampled[:test_points])

    min_len = min(len(traj) for traj in test_trajectories)
    stacked_test = np.stack([traj[:min_len] for traj in test_trajectories], axis=0)
    metadata = TrajectoryMetadata(
        system_name=system_name,
        state_dim=state_dim,
        seed=args.seed,
        num_trajectories=args.num_trajectories,
        benchmark_split_mode="single_train_multi_test",
        integration_dt=args.integration_dt,
        lyapunov_time=lyapunov_time,
        target_pts_per_lyapunov=args.target_pts_per_lyapunov,
        downsample_stride=stride,
        effective_dt=effective_dt,
        warmup_steps_raw=warmup_steps_raw,
        kept_steps_raw=test_kept_steps_raw,
        train_points=len(train_trajectory),
        test_points=min_len,
        ic_sampling_method="reference_trajectory_tail" if system_name == "Lorenz96" else "on_attractor",
        ic_sampling_fallback=None if system_name == "Lorenz96" else ("gaussian_perturbation" if fallback_used else None),
        reference_traj_length=args.reference_traj_length,
        reference_traj_transient=args.reference_traj_transient,
    )
    return train_trajectory, stacked_test, metadata


def save_system(save_dir, system_name, train_trajectory, test_trajectories, metadata):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{system_name}.npz")
    np.savez_compressed(
        output_path,
        train_trajectory=train_trajectory,
        test_trajectories=test_trajectories,
        system_name=metadata.system_name,
        state_dim=metadata.state_dim,
        benchmark_split_mode=metadata.benchmark_split_mode,
        integration_dt=metadata.integration_dt,
        lyapunov_time=metadata.lyapunov_time,
        target_pts_per_lyapunov=metadata.target_pts_per_lyapunov,
        downsample_stride=metadata.downsample_stride,
        effective_dt=metadata.effective_dt,
        warmup_steps_raw=metadata.warmup_steps_raw,
        kept_steps_raw=metadata.kept_steps_raw,
        train_points=metadata.train_points,
        test_points=metadata.test_points,
        ic_sampling_method=metadata.ic_sampling_method,
        ic_sampling_fallback=metadata.ic_sampling_fallback,
        reference_traj_length=metadata.reference_traj_length,
        reference_traj_transient=metadata.reference_traj_transient,
        seed=metadata.seed,
    )
    with open(os.path.join(save_dir, f"{system_name}.json"), "w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2)
    print(
        f"[Saved] {system_name}: train_trajectory={train_trajectory.shape}, "
        f"test_trajectories={test_trajectories.shape}, "
        f"state_dim={metadata.state_dim}, "
        f"lyapunov_time={metadata.lyapunov_time:.6f}, stride={metadata.downsample_stride}, "
        f"effective_dt={metadata.effective_dt:.6f}, "
        f"ic_sampling={metadata.ic_sampling_method}, fallback={metadata.ic_sampling_fallback or 'none'}"
    )


def main():
    args = parse_args()
    for system_name in args.systems:
        train_trajectory, test_trajectories, metadata = generate_system(system_name, args)
        save_system(args.save_dir, system_name, train_trajectory, test_trajectories, metadata)


if __name__ == "__main__":
    main()
