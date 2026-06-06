import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler as SklearnScaler
from torch.utils.data import Dataset


class StandardScaler2D:
    """Scale 2D time-series arrays shaped as [time, features]."""

    def __init__(self):
        self.scaler = SklearnScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        is_tensor = torch.is_tensor(data)
        device = data.device if is_tensor else None
        data_np = data.detach().cpu().numpy() if is_tensor else data
        original_shape = data_np.shape
        feat_dim = original_shape[-1]
        restored = self.scaler.inverse_transform(data_np.reshape(-1, feat_dim)).reshape(original_shape)
        if is_tensor:
            return torch.from_numpy(restored).float().to(device)
        return restored


class BaseChaosDataset(Dataset):
    def __init__(self, data, lookback, horizon, flag="train", data_ratio=1.0, noise=0.0, window_stride=1):
        super().__init__()
        if window_stride < 1:
            raise ValueError(f"window_stride must be >= 1, got {window_stride}.")
        if len(data) < lookback + horizon:
            raise ValueError(
                f"Data length ({len(data)}) is too short for lookback={lookback} and horizon={horizon}."
            )

        num_samples = len(data) - lookback - horizon + 1
        indices = np.arange(0, num_samples, window_stride)
        x = np.array([data[i : i + lookback] for i in indices], dtype=np.float32)
        y = np.array([data[i + lookback : i + lookback + horizon] for i in indices], dtype=np.float32)

        if flag == "train" and data_ratio < 1.0:
            keep = max(1, int(len(x) * data_ratio))
            chosen = np.random.choice(len(x), keep, replace=False)
            x = x[chosen]
            y = y[chosen]

        self.X = torch.from_numpy(x).float()
        self.Y = torch.from_numpy(y).float()

        if flag == "train" and noise > 0.0:
            self.X = self.X + torch.randn_like(self.X) * noise

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


class ChaosTimeSeriesDataset(BaseChaosDataset):
    """Single-step chaotic forecasting dataset with y shaped as [B, 1, F]."""

    def __init__(self, data, lookback, flag="train", data_ratio=1.0, noise=0.0, window_stride=1):
        super().__init__(
            data,
            lookback=lookback,
            horizon=1,
            flag=flag,
            data_ratio=data_ratio,
            noise=noise,
            window_stride=window_stride,
        )


class ChaosTimeSeriesDatasetMultiSteps(BaseChaosDataset):
    """Multi-step chaotic forecasting dataset with y shaped as [B, H, F]."""

    def __init__(self, data, lookback, horizon, flag="train", data_ratio=1.0, noise=0.0, window_stride=1):
        super().__init__(
            data,
            lookback=lookback,
            horizon=horizon,
            flag=flag,
            data_ratio=data_ratio,
            noise=noise,
            window_stride=window_stride,
        )


def _trajectory_split_indices(
    num_trajectories,
    train_trajectories=None,
    val_trajectories=None,
    test_trajectories=None,
    val_ratio=0.1,
    test_ratio=0.2,
):
    if train_trajectories is None and val_trajectories is None and test_trajectories is None:
        if num_trajectories >= 20:
            train_trajectories, val_trajectories, test_trajectories = 12, 4, 4
        else:
            test_trajectories = max(1, int(round(num_trajectories * test_ratio)))
            val_trajectories = max(1, int(round(num_trajectories * val_ratio)))
            train_trajectories = num_trajectories - val_trajectories - test_trajectories
    else:
        val_trajectories = 0 if val_trajectories is None else val_trajectories
        test_trajectories = 0 if test_trajectories is None else test_trajectories
        if train_trajectories is None:
            train_trajectories = num_trajectories - val_trajectories - test_trajectories

    if train_trajectories <= 0 or val_trajectories <= 0 or test_trajectories <= 0:
        raise ValueError("Invalid trajectory split. Train/val/test trajectory counts must all be positive.")
    if train_trajectories + val_trajectories + test_trajectories > num_trajectories:
        raise ValueError("Trajectory split exceeds the number of available trajectories.")

    train_end = train_trajectories
    val_end = train_end + val_trajectories
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, val_end + test_trajectories)


def _fit_scaler_on_trajectories(train_raw):
    feat_dim = train_raw.shape[-1]
    scaler = StandardScaler2D()
    scaler.fit(train_raw.reshape(-1, feat_dim))
    return scaler


def _transform_trajectories(data, scaler):
    num_traj, time_len, feat_dim = data.shape
    transformed = scaler.transform(data.reshape(-1, feat_dim)).reshape(num_traj, time_len, feat_dim)
    return transformed.astype(np.float32)


def _transform_trajectory(data, scaler):
    return scaler.transform(data).astype(np.float32)


def _stack_trajectory_windows(data, lookback, horizon, window_stride=1):
    if window_stride < 1:
        raise ValueError(f"window_stride must be >= 1, got {window_stride}.")
    x_segments = []
    y_segments = []
    for trajectory in data:
        if len(trajectory) < lookback + horizon:
            continue
        num_samples = len(trajectory) - lookback - horizon + 1
        indices = np.arange(0, num_samples, window_stride)
        x_segments.append(np.array([trajectory[i : i + lookback] for i in indices], dtype=np.float32))
        y_segments.append(
            np.array([trajectory[i + lookback : i + lookback + horizon] for i in indices], dtype=np.float32)
        )

    if not x_segments:
        feat_dim = data.shape[-1]
        return (
            np.empty((0, lookback, feat_dim), dtype=np.float32),
            np.empty((0, horizon, feat_dim), dtype=np.float32),
        )
    return np.concatenate(x_segments, axis=0), np.concatenate(y_segments, axis=0)


class MultiTrajectoryChaosDataset(Dataset):
    def __init__(self, data, lookback, horizon, flag="train", data_ratio=1.0, noise=0.0, window_stride=1):
        super().__init__()
        if data.ndim != 3:
            raise ValueError(f"Expected multi-trajectory data with shape [N, T, F], but got {data.shape}")

        x, y = _stack_trajectory_windows(
            data,
            lookback=lookback,
            horizon=horizon,
            window_stride=window_stride,
        )

        if flag == "train" and data_ratio < 1.0 and len(x) > 0:
            keep = max(1, int(len(x) * data_ratio))
            chosen = np.random.choice(len(x), keep, replace=False)
            x = x[chosen]
            y = y[chosen]

        self.X = torch.from_numpy(x).float()
        self.Y = torch.from_numpy(y).float()

        if flag == "train" and noise > 0.0 and len(self.X) > 0:
            self.X = self.X + torch.randn_like(self.X) * noise

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


def resolve_chaos_data_path(args):
    candidates = []
    benchmark_split_mode = getattr(args, "benchmark_split_mode", None)
    if benchmark_split_mode == "single_train_multi_test" and hasattr(args, "system"):
        candidates.extend(
            [
                os.path.join(getattr(args, "save_dir", "."), f"{args.system}.npz"),
                f"{args.system}.npz",
            ]
        )
    if hasattr(args, "data_path"):
        candidates.append(args.data_path)
    if hasattr(args, "save_dir") and hasattr(args, "data_path"):
        candidates.append(os.path.join(args.save_dir, args.data_path))
    if hasattr(args, "save_dir"):
        candidates.append(os.path.join(args.save_dir, "long_traj.npy"))
    candidates.extend(["long_traj.npy", "long_trajlo.npy"])
    if hasattr(args, "system"):
        candidates.extend(
            [
                os.path.join(getattr(args, "save_dir", "."), f"{args.system}.npz"),
                os.path.join(getattr(args, "save_dir", "."), f"{args.system}.npy"),
                f"{args.system}.npz",
                f"{args.system}.npy",
            ]
        )

    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find chaotic trajectory file. Checked: {candidates}")


def load_chaos_array(file_path):
    raw = np.load(file_path, allow_pickle=False)
    if isinstance(raw, np.lib.npyio.NpzFile):
        if "train_trajectory" in raw and "test_trajectories" in raw:
            return {
                "train_trajectory": raw["train_trajectory"],
                "test_trajectories": raw["test_trajectories"],
            }
        if "trajectories" in raw:
            return raw["trajectories"]
        if "data" in raw:
            return raw["data"]
        raise ValueError(f"Unsupported chaotic benchmark archive keys: {list(raw.keys())}")
    return raw


def normalize_chaos_shape(raw):
    if isinstance(raw, dict):
        return {
            "train_trajectory": normalize_chaos_shape(raw["train_trajectory"]),
            "test_trajectories": normalize_chaos_shape(raw["test_trajectories"]),
        }
    raw = np.asarray(raw)
    if raw.ndim == 3:
        if raw.shape[0] > 1 and raw.shape[1] > 1:
            return raw.astype(np.float32)
        if raw.shape[0] == 1:
            raw = raw.squeeze(0)
        elif raw.shape[1] == 1:
            raw = raw.squeeze(1)
        else:
            raise ValueError(f"Unsupported 3D chaotic trajectory shape: {raw.shape}")
    if raw.ndim != 2:
        raise ValueError(f"Expected raw data with shape [time, features], but got {raw.shape}")
    return raw


def build_chaos_splits(args):
    file_path = resolve_chaos_data_path(args)
    raw = normalize_chaos_shape(load_chaos_array(file_path))

    lookback = getattr(args, "lookback", getattr(args, "input_size"))
    val_ratio = getattr(args, "val_ratio", 0.1)
    test_ratio = getattr(args, "test_ratio", 0.2)
    train_ratio = 1.0 - val_ratio - test_ratio
    train_val_ratio = getattr(args, "train_val_ratio", 0.85)

    if isinstance(raw, dict):
        train_trajectory = raw["train_trajectory"]
        test_trajectories = raw["test_trajectories"]
        if train_trajectory.ndim != 2:
            raise ValueError(f"Expected train_trajectory with shape [T, F], got {train_trajectory.shape}.")
        if test_trajectories.ndim != 3:
            raise ValueError(
                f"Expected test_trajectories with shape [N, T, F], got {test_trajectories.shape}."
            )
        if not (0.0 < train_val_ratio < 1.0):
            raise ValueError(f"train_val_ratio must be in (0, 1), got {train_val_ratio}.")

        train_end = int(len(train_trajectory) * train_val_ratio)
        train_raw = train_trajectory[:train_end]
        val_start = max(0, train_end - lookback)
        val_raw = train_trajectory[val_start:]

        scaler = StandardScaler2D()
        scaler.fit(train_raw)

        train_data = _transform_trajectory(train_raw, scaler)
        val_data = _transform_trajectory(val_raw, scaler)
        test_data = _transform_trajectories(test_trajectories, scaler)
        return train_data, val_data, test_data, scaler

    if raw.ndim == 3:
        train_slice, val_slice, test_slice = _trajectory_split_indices(
            raw.shape[0],
            train_trajectories=getattr(args, "train_trajectories", None),
            val_trajectories=getattr(args, "val_trajectories", None),
            test_trajectories=getattr(args, "test_trajectories", None),
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_raw = raw[train_slice]
        val_raw = raw[val_slice]
        test_raw = raw[test_slice]

        scaler = _fit_scaler_on_trajectories(train_raw)
        train_data = _transform_trajectories(train_raw, scaler)
        val_data = _transform_trajectories(val_raw, scaler)
        test_data = _transform_trajectories(test_raw, scaler)
        return train_data, val_data, test_data, scaler

    if train_ratio <= 0:
        raise ValueError("Invalid split ratios: train ratio must be positive.")

    total_len = len(raw)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train_raw = raw[:train_end]
    val_start = max(0, train_end - lookback)
    test_start = max(0, val_end - lookback)
    val_raw = raw[val_start:val_end]
    test_raw = raw[test_start:]

    scaler = StandardScaler2D()
    scaler.fit(train_raw)

    train_data = scaler.transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)
    return train_data, val_data, test_data, scaler

