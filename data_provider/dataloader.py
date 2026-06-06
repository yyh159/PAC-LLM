from torch.utils.data import DataLoader

try:
    from .dataset import (
        ChaosTimeSeriesDataset,
        ChaosTimeSeriesDatasetMultiSteps,
        MultiTrajectoryChaosDataset,
        build_chaos_splits,
    )
except ImportError:
    from dataset import (
        ChaosTimeSeriesDataset,
        ChaosTimeSeriesDatasetMultiSteps,
        MultiTrajectoryChaosDataset,
        build_chaos_splits,
    )


def data_provider_chaos(args, mode="train", task_type="single_step"):
    """
    Reference-style data provider for chaotic forecasting.

    task_type:
    - "single_step": y shape [B, 1, F]
    - "multi_step": y shape [B, H, F]
    """

    train_data, val_data, test_data, scaler = build_chaos_splits(args)
    lookback = getattr(args, "lookback", getattr(args, "input_size"))
    horizon = getattr(args, "horizon", getattr(args, "eval_horizon", getattr(args, "pred_len", 1)))
    data_ratio = getattr(args, "data_ratio", 1.0)
    noise = getattr(args, "noise", 0.0)
    train_window_stride = getattr(args, "train_window_stride", 4)
    eval_window_stride = getattr(args, "eval_window_stride", 1)

    def build_dataset(split_data, flag):
        window_stride = train_window_stride if flag == "train" else eval_window_stride
        target_horizon = 1 if task_type == "single_step" else horizon
        split_data_kwargs = {
            "lookback": lookback,
            "flag": flag,
            "window_stride": window_stride,
        }
        if flag == "train":
            split_data_kwargs.update(data_ratio=data_ratio, noise=noise)

        if split_data.ndim == 3:
            return MultiTrajectoryChaosDataset(
                split_data,
                horizon=target_horizon,
                **split_data_kwargs,
            )
        if split_data.ndim != 2:
            raise ValueError(f"Expected 2D or 3D chaotic split data, got shape {split_data.shape}.")
        if task_type == "single_step":
            return ChaosTimeSeriesDataset(split_data, **split_data_kwargs)
        if task_type == "multi_step":
            return ChaosTimeSeriesDatasetMultiSteps(
                split_data,
                horizon=target_horizon,
                **split_data_kwargs,
            )
        raise ValueError(f"Unsupported task_type: {task_type}")

    train_dataset = build_dataset(train_data, "train")
    val_dataset = build_dataset(val_data, "val")
    test_dataset = build_dataset(test_data, "test")

    mode_to_dataset = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    if mode not in mode_to_dataset:
        raise ValueError(f"Unsupported mode: {mode}. Expected train/val/test.")

    dataset = mode_to_dataset[mode]
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(mode == "train"),
        num_workers=0,
        drop_last=(mode == "train"),
    )
    return dataset, data_loader, scaler


def load_data_single_step(args):
    _, train_loader, scaler = data_provider_chaos(args, mode="train", task_type="single_step")
    _, val_loader, _ = data_provider_chaos(args, mode="val", task_type="single_step")
    _, test_loader, _ = data_provider_chaos(args, mode="test", task_type="single_step")
    return train_loader, val_loader, test_loader, scaler


def load_data_multi_step(args):
    _, train_loader, scaler = data_provider_chaos(args, mode="train", task_type="multi_step")
    _, val_loader, _ = data_provider_chaos(args, mode="val", task_type="multi_step")
    _, test_loader, _ = data_provider_chaos(args, mode="test", task_type="multi_step")
    return train_loader, val_loader, test_loader, scaler


def load_data(args):
    """Compatibility helper: default to multi-step if horizon > 1, else single-step."""
    horizon = getattr(args, "horizon", getattr(args, "eval_horizon", getattr(args, "pred_len", 1)))
    if horizon > 1:
        return load_data_multi_step(args)
    return load_data_single_step(args)

