import os
import random
import sys
import argparse
from copy import copy
import time
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from data_provider.dataloader import load_data as load_data_chaos
from util.metrics import KLD, MAE, MSE, SMAPE, VPT, gpdim_rmse
from model.pac_llm import PACLLM
from model.baseline.nbeats import SpatioTemporalNBeats
from model.baseline.timellm import Model as TimeLLM
tslib_path = os.path.abspath("./Time-Series-Library-main")
if tslib_path not in sys.path:
    sys.path.insert(0, tslib_path)
    sys.path.insert(0, os.path.join(tslib_path, "models"))
from model.baseline.crossformer import Model as Crossformer
from model.baseline.timesnet import Model as TimesNet
from model.baseline.fm import PandaPipelineAdapter
from model.baseline.nvar import SpatioTemporalNVAR
from model.baseline.llmmixer import SpatioTemporalLLMMixer
from model.baseline.esn import ESNAdapter

def fix_random_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Global random seed fixed at {seed} for fair comparison.")


class ForecastLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_diff=0.1):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_diff = lambda_diff
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
        diff_target = target[:, 1:, :] - target[:, :-1, :]
        loss_diff = self.mse(diff_pred, diff_target) if pred.size(1) > 1 else pred.new_tensor(0.0)
        total_loss = self.lambda_mse * loss_mse + self.lambda_diff * loss_diff
        return total_loss, {"mse": loss_mse.item(), "diff": loss_diff.item()}


class Config:
    save_dir = "./data_dysts_benchmark"
    checkpoint_root = "./checkpoints_rollout"
    results_dir = "./benchmark_results"
    viz_cache_dir = "./benchmark_viz_cache"
    export_viz_cache = True
    system = "Lorenz"#VKVS_POD
    benchmark_split_mode = "single_train_multi_test"
    train_val_ratio = 0.85
    num_test_trajectories = 20
    input_size = 30
    label_len = 0
    lookback = 30
    pred_len = 30
    train_horizon = 30
    valid_horizon = 30
    eval_horizon = 300
    horizon = eval_horizon
    rollout_horizons = [30,120,300]
    train_window_stride = 1
    eval_window_stride = 1
    train_data_path = None
    test_data_path = None
    ks_fourier_k = 16
    vkvs_pod_k = 16
    batch_size = 32
    epochs = 150
    learning_rate = 5e-5
    weight_decay = 1e-4
    patience = 20
    warmup_epochs = 8
    grad_clip = 0.3

    task_name = "long_term_forecast"
    llm_model = "GPT2"
    llm_dim = 768
    seq_len = input_size
    d_model = 32
    d_ff = 128
    n_heads = 8
    e_layers = 2
    factor = 3
    dropout = 0.1
    embed = "timeF"
    patch_len = 8
    stride =4
    feat_dim =3
    enc_in = feat_dim
    dec_in = feat_dim
    c_out = feat_dim
    takens_embed_dim = 3
    features = "M"
    output_attention = False
    freq = "h"
    moving_avg = 25
    activation = "gelu"
    top_k = 5
    num_kernels = 6
    num_class = 1
    seg_len = 12
    win_size = 2
    fc_dropout = 0.1
    head_dropout = 0.0
    pred_seg_len = 16
    auto_patch = False
    decoder_layers = 1
    is_gpt = True
    pretrain = True
    freeze = True
    gpt_layers = 2
    patch_size = patch_len
    prompt_domain = False
    content = "A synthetic dataset for training and evaluating time-series forecasting models, generated from the classic Lorenz chaotic dynamical system.​ This dataset contains multiple time series, each consisting of a set of three-dimensional state variables (x, y, z) obtained through numerical integration. It is commonly used to test models' capabilities in short-term prediction and long-term trajectory generalization within complex, nonlinear dynamical systems."
    prompt_dir = os.path.join(PROJECT_ROOT, "prompt")
    dataset_description = None
    prompt_prefix = ""
    lambda_main = 0.9
    pac_aux_beta = 1
    feature_mode = "raw"
    aux_model_type='mlp'
    model_variant = "pac_llm"

    nbeats_hidden_units = 128
    nbeats_nb_blocks_per_stack = 3
    nbeats_num_stacks = 2
    nbeats_theta_dim = 128
    nbeats_share_weights_in_stack = False
    nbeats_backcast_length = input_size
   
    esn_reservoir_size = 500
    esn_spectral_radius = 0.9
    esn_input_scaling = 0.5
    esn_leak_rate = 1.0
    esn_sparsity = 0.9
    esn_ridge = 1e-4
    esn_washout = 5
    esn_bias_scale = 0.2
   
   
    panda_pretrain_path = "GilpinLab/panda"
    panda_device_map = "cuda" if torch.cuda.is_available() else "cpu"
    panda_limit_prediction_length = False
    panda_sliding_context = True

    nvar_k = 30
    nvar_ridge_param = 1
    nvar_include_constant = True

  
    target = "var_4"
    seasonal_patterns = None
    num_workers = 0
    augmentation_ratio = 0

   
    llmmixer_llm_path = "openai-community/gpt2"
    llmmixer_tokenizer_path = "openai-community/gpt2"
    llmmixer_d_model = 32
    llmmixer_d_ff = 128
    llmmixer_n_heads = 8
    llmmixer_e_layers = 2
    llmmixer_factor = 3
    llmmixer_moving_avg = 25
    llmmixer_top_k = 5
    llmmixer_down_sampling_layers = 1
    llmmixer_down_sampling_window = 2
    llmmixer_down_sampling_method = "avg"
    llmmixer_channel_independence = 1
    llmmixer_decomp_method = "moving_avg"
    llmmixer_use_norm = 1
    llmmixer_use_future_temporal_feature = 0
    
    benchmark_efficiency = False
    efficiency_warmup = 5
    efficiency_iters = 20
  
    llm_layers = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_cli_overrides(config):
    parser = argparse.ArgumentParser(description="Train or evaluate PAC-LLM on chaotic benchmark datasets.")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--model_variant", type=str, default=None)
    parser.add_argument("--feature_mode", type=str, default=None)
    parser.add_argument("--aux_model_type", type=str, default=None)
    parser.add_argument("--aux_source", type=str, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--llm_dim", type=int, default=None)
    parser.add_argument("--llm_layers", type=int, default=None)
    parser.add_argument("--takens_embed_dim", type=int, default=None)
    parser.add_argument("--lambda_main", type=float, default=None)
    parser.add_argument("--lambda_mse", type=float, default=None)
    parser.add_argument("--lambda_diff", type=float, default=None)
    parser.add_argument("--feat_dim", type=int, default=None)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--train_horizon", type=int, default=None)
    parser.add_argument("--valid_horizon", type=int, default=None)
    parser.add_argument("--eval_horizon", type=int, default=None)
    parser.add_argument("--rollout_horizons", nargs="+", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--checkpoint_root", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--viz_cache_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--prompt_dir", type=str, default=None)
    parser.add_argument("--dataset_description", type=str, default=None)
    parser.add_argument("--export_viz_cache", choices=["yes", "no"], default=None)
    args, _ = parser.parse_known_args()

    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    if args.input_size is not None:
        config.seq_len = config.input_size
        config.lookback = config.input_size if args.lookback is None else config.lookback
        config.nbeats_backcast_length = config.input_size
    if args.lookback is not None:
        config.input_size = config.lookback
        config.seq_len = config.lookback
        config.nbeats_backcast_length = config.lookback
    if args.train_horizon is not None:
        config.pred_len = config.train_horizon
        config.horizon = config.eval_horizon
    if args.eval_horizon is not None:
        config.horizon = config.eval_horizon
    if args.feat_dim is not None:
        config.enc_in = config.feat_dim
        config.dec_in = config.feat_dim
        config.c_out = config.feat_dim
    if args.patch_len is not None:
        config.patch_size = config.patch_len
    if args.export_viz_cache is not None:
        config.export_viz_cache = args.export_viz_cache == "yes"

    return config


def is_pac_llm_variant(config):
    return config.model_variant == "pac_llm"


def is_timellm_variant(config):
    return config.model_variant == "timellm"
def is_esn_variant(config):
    return config.model_variant == "esn"
def is_nbeats_variant(config):
    return config.model_variant == "nbeats"
def is_panda_variant(config):
    return config.model_variant == "panda"


def uses_predict_interface(config):
    return is_panda_variant(config)


def is_llmmixer_variant(config):
    return config.model_variant == "llmmixer"
def is_nvar_variant(config):
    return config.model_variant == "nvar"

def should_use_amp(config, device):
    if device.type != "cuda":
        return False
    if config.model_variant in {"timesnet"}:
        return False
    return True

def uses_aux_loss(config):
    return is_pac_llm_variant(config)


def uses_similarity_gate(config):
    return is_pac_llm_variant(config)


def format_variant_name(config):
    if is_pac_llm_variant(config):
        return f"pac_llm_{config.feature_mode}_{config.aux_model_type}"
    return config.model_variant

def load_data(config):
    return load_data_chaos(config)


def load_dataset_description(config):
    if getattr(config, "dataset_description", None):
        return config.dataset_description.strip()

    prompt_dir = getattr(config, "prompt_dir", None)
    if not prompt_dir:
        return "Chaotic multivariate trajectory forecasting."

    candidates = [
        os.path.join(prompt_dir, f"{config.system}.txt"),
        os.path.join(prompt_dir, f"{config.system.lower()}.txt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                description = handle.read().strip()
            if description:
                return description

    raise FileNotFoundError(
        f"Dataset description prompt not found for system '{config.system}'. "
        f"Expected one of: {candidates}"
    )


def inverse_scale_numpy(data, scaler):
    if scaler is None:
        return data
    array = np.asarray(data)
    original_shape = array.shape
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim > 2:
        array = array.reshape(-1, original_shape[-1])
    restored = scaler.inverse_transform(array)
    return restored.reshape(original_shape)


def resolve_dataset_metadata(config):
    candidate_paths = []
    for path in (getattr(config, "data_path", None), getattr(config, "train_data_path", None)):
        if not path:
            continue
        candidate_paths.append(path)
        if not os.path.isabs(path):
            candidate_paths.append(os.path.join(config.save_dir, path))

    for candidate in candidate_paths:
        if not (candidate and os.path.exists(candidate) and candidate.endswith(".npz")):
            continue
        with np.load(candidate, allow_pickle=True) as payload:
            metadata = {}
            for key in ("effective_dt", "lyapunov_time", "downsample_stride", "system_name"):
                if key in payload:
                    value = payload[key]
                    if np.isscalar(value) or getattr(value, "shape", ()) == ():
                        metadata[key] = value.item() if hasattr(value, "item") else value
                    else:
                        metadata[key] = value
            if metadata:
                return metadata
    return {}

def summarize_metric(value):
    if value != value:
        return "NA"
    return f"{value:.4f}"


def get_checkpoint_dir(config):
    return os.path.join(config.checkpoint_root, format_variant_name(config))

def format_efficiency_value(value, precision=2):
    if value != value:
        return "NA"
    return f"{value:.{precision}f}"


def benchmark_efficiency(model, sample_batch, device, config):
    model.train()
    #use_amp = device.type == "cuda"
    use_amp = should_use_amp(config, device)

    batch_x, batch_y, prompt_embeddings = unpack_batch(sample_batch, device, config)
    batch_x_mark, batch_y_mark, x_dec = build_decoder_inputs(
        batch_x, batch_y, config.train_horizon, config.label_len, device
    )
    batch_y_target = batch_y[:, : config.train_horizon, :]
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(device=device.type, enabled=use_amp)
    criterion = ForecastLoss(
        lambda_mse=config.lambda_mse,
        lambda_diff=config.lambda_diff,
    ).to(device)

    def run_train_iteration():
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = forward_train_step(
                model,
                batch_x,
                batch_x_mark,
                x_dec,
                batch_y_mark,
                config,
                prompt_embeddings=prompt_embeddings,
            )
            loss, _ = compute_training_loss(outputs, batch_y_target, criterion, config)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    for _ in range(config.efficiency_warmup):
        loss = run_train_iteration()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered during efficiency warmup.")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    timings = []
    for _ in range(config.efficiency_iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = run_train_iteration()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss encountered during efficiency benchmark.")
        timings.append(time.perf_counter() - start)

    optimizer.zero_grad(set_to_none=True)

    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    peak_mem_mb = float("nan")
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "params_m": params_m,
        "peak_mem_mb": peak_mem_mb,
        "avg_forward_time_s": sum(timings) / max(len(timings), 1),
    }

def collect_efficiency_results(config, sample_batch):
    results = []
    benchmark_variants = [
        ("pac_llm", "PAC-LLM"),
        ("crossformer", "Crossformer"),
        ("nbeats", "NBEATS"),
        ("timesnet", "TimesNet"),
        ("timellm", "TimeLLM"),
        ("llmmixer", "LLMMixer"),
    ]

    for variant, label in benchmark_variants:
        bench_config = copy(config)
        bench_config.model_variant = variant
        model = build_model(bench_config, sample_batch=sample_batch).to(bench_config.device)
        metrics = benchmark_efficiency(model, sample_batch, bench_config.device, bench_config)
        results.append({"label": label, **metrics})
        del model
        if bench_config.device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def print_efficiency_table(results):
    print("Efficiency Metrics")
    print("-" * 72)
    print(f"{'Model':<18}{'Param.(M)':>12}{'Mem.(MB)':>14}{'Speed(s/iter)':>18}")
    print("-" * 72)
    for result in results:
        print(
            f"{result['label']:<18}"
            f"{format_efficiency_value(result['params_m']):>12}"
            f"{format_efficiency_value(result['peak_mem_mb']):>14}"
            f"{format_efficiency_value(result['avg_forward_time_s'], precision=4):>18}"
        )
    print("-" * 72)


def build_decoder_inputs(batch_x, batch_y, pred_len, label_len, device):
    batch_size = batch_x.shape[0]
    seq_len = batch_x.shape[1]

    batch_x_mark = torch.zeros(batch_size, seq_len, 4, device=device)
    batch_y_mark = torch.zeros(batch_size, label_len + pred_len, 4, device=device)

    if label_len > 0:
        known_part = batch_y[:, :label_len, :]
    else:
        known_part = torch.empty((batch_size, 0, batch_x.shape[2]), device=device)

    zero_part = torch.zeros(batch_size, pred_len, batch_x.shape[2], device=device)
    x_dec = torch.cat([known_part, zero_part], dim=1)
    return batch_x_mark, batch_y_mark, x_dec


def unpack_batch(batch, device, config):
    batch_x, batch_y = batch
    return batch_x.float().to(device), batch_y.float().to(device), None


def compute_pac_llm_rollout_loss(outputs, target, criterion, lambda_main, use_aux_loss):
    main_loss, main_details = criterion(outputs["main"], target)
    aux_loss, aux_details = criterion(outputs["aux"], target)

    if use_aux_loss:
        total_loss = lambda_main * main_loss + (1.0 - lambda_main) * aux_loss
        mse_value = lambda_main * main_details["mse"] + (1.0 - lambda_main) * aux_details["mse"]
        diff_value = lambda_main * main_details["diff"] + (1.0 - lambda_main) * aux_details["diff"]
    else:
        total_loss = main_loss
        mse_value = main_details["mse"]
        diff_value = main_details["diff"]

    details = {
        "mse": mse_value,
        "diff": diff_value,
        "main_mse": main_details["mse"],
        "aux_mse": aux_details["mse"],
        "alpha_mean": outputs["alpha"].mean().item(),
        "sim_gate_mean": outputs["sim_gate_mean"].mean().item(),
    }
    return total_loss, details


def compute_baseline_loss(outputs, target, criterion):
    loss, details = criterion(outputs, target)
    details = {
        "mse": details["mse"],
        "diff": details["diff"],
        "main_mse": details["mse"],
        "aux_mse": float("nan"),
        "alpha_mean": float("nan"),
        "sim_gate_mean": float("nan"),
    }
    return loss, details

def forward_train_step(model, batch_x, batch_x_mark, x_dec, batch_y_mark, config, prompt_embeddings=None):
    if is_pac_llm_variant(config):
        outputs = model(batch_x, batch_x_mark, x_dec, batch_y_mark, return_aux=True)
        outputs["main"] = outputs["main"][:, : config.train_horizon, :]
        outputs["aux"] = outputs["aux"][:, : config.train_horizon, :]
        return outputs

    outputs = model(batch_x, batch_x_mark, x_dec, batch_y_mark)
    return outputs[:, : config.train_horizon, :]


def compute_training_loss(outputs, target, criterion, config):
    if isinstance(outputs, dict):
        return compute_pac_llm_rollout_loss(
            outputs,
            target,
            criterion,
            lambda_main=config.lambda_main,
            use_aux_loss=uses_aux_loss(config),
        )
    return compute_baseline_loss(outputs, target, criterion)


def build_model(config, sample_batch=None):
    common_kwargs = {
        "h": config.train_horizon,
        "input_size": config.input_size,
        "patch_len": config.patch_len,
        "stride": config.stride,
        "d_llm": config.llm_dim,
        "feat_dim": config.feat_dim,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "n_heads": config.n_heads,
        "dropout": config.dropout,
        "llm_layers": config.llm_layers,
        "embed_dim": config.takens_embed_dim,
    }

    if is_pac_llm_variant(config):
        return PACLLM(
            lambda_main=config.lambda_main,
            use_similarity_gate=uses_similarity_gate(config),
            feature_mode=config.feature_mode,
            aux_model_type=config.aux_model_type,
            prompt_prefix=getattr(config, "prompt_prefix", ""),
            dataset_description=load_dataset_description(config),
            **common_kwargs,
        )
    if is_esn_variant(config):
        return ESNAdapter(
            h=config.train_horizon,
            input_size=config.input_size,
            feat_dim=config.feat_dim,
            reservoir_size=config.esn_reservoir_size,
            spectral_radius=config.esn_spectral_radius,
            input_scaling=config.esn_input_scaling,
            leak_rate=config.esn_leak_rate,
            sparsity=config.esn_sparsity,
            ridge=config.esn_ridge,
            washout=config.esn_washout,
            bias_scale=config.esn_bias_scale,
        )
    if is_nvar_variant(config):
        return SpatioTemporalNVAR(
            h=config.train_horizon,
            input_size=config.input_size,
            feat_dim=config.feat_dim,
            k=config.nvar_k,
            ridge_param=config.nvar_ridge_param,
            include_constant=config.nvar_include_constant,
        )
    if is_llmmixer_variant(config):
        return SpatioTemporalLLMMixer(
            h=config.train_horizon,
            input_size=config.input_size,
            feat_dim=config.feat_dim,
            llm_path=config.llmmixer_llm_path,
            tokenizer_path=config.llmmixer_tokenizer_path,
            d_model=config.llmmixer_d_model,
            d_ff=config.llmmixer_d_ff,
            e_layers=config.llmmixer_e_layers,
            moving_avg=config.llmmixer_moving_avg,
            top_k=config.llmmixer_top_k,
            down_sampling_layers=config.llmmixer_down_sampling_layers,
            down_sampling_window=config.llmmixer_down_sampling_window,
            down_sampling_method=config.llmmixer_down_sampling_method,
            channel_independence=config.llmmixer_channel_independence,
            decomp_method=config.llmmixer_decomp_method,
            use_norm=config.llmmixer_use_norm,
            use_future_temporal_feature=config.llmmixer_use_future_temporal_feature,
            dropout=config.dropout,
            embed=config.embed,
            freq=config.freq,
            factor=config.llmmixer_factor,
            n_heads=config.llmmixer_n_heads,
        )

    if is_timellm_variant(config):
        timellm_config = copy(config)
        timellm_config.task_name = "long_term_forecast"
        timellm_config.pred_len = config.train_horizon
        timellm_config.seq_len = config.input_size
        timellm_config.enc_in = config.feat_dim
        timellm_config.dec_in = config.feat_dim
        timellm_config.c_out = config.feat_dim
        timellm_config.llm_model = "GPT2"
        timellm_config.d_ff = config.d_ff
        timellm_config.content = getattr(config, "content", "Chaotic multivariate trajectory forecasting.")
        return TimeLLM(timellm_config, patch_len=timellm_config.patch_len, stride=timellm_config.stride)
    if is_nbeats_variant(config):
        return SpatioTemporalNBeats(
            h=config.train_horizon,
            input_size=config.input_size,
            feat_dim=config.feat_dim,
            hidden_units=config.nbeats_hidden_units,
            nb_blocks_per_stack=config.nbeats_nb_blocks_per_stack,
            num_stacks=config.nbeats_num_stacks,
            theta_dim=config.nbeats_theta_dim,
            share_weights_in_stack=config.nbeats_share_weights_in_stack,
            backcast_length=config.nbeats_backcast_length,
        )
    if config.model_variant == "crossformer":
        return Crossformer(config).float()
    if config.model_variant == "timesnet":
        return TimesNet(config).float()
    if is_panda_variant(config):
        return PandaPipelineAdapter(
            pretrain_path=config.panda_pretrain_path,
            device=config.device,
            device_map=config.panda_device_map,
            limit_prediction_length=config.panda_limit_prediction_length,
            sliding_context=config.panda_sliding_context,
        )

    supported = [
        "pac_llm",
        "esn",
        "nvar",
        "llmmixer",
        "timellm",
        "nbeats",
        "crossformer",
        "timesnet",
        "panda",
    ]
    raise ValueError(f"Unsupported model_variant='{config.model_variant}'. Supported variants: {supported}")


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, config):
    model.train()
    total_loss = 0.0
    metric_sums = {
        "mse": 0.0,
        "diff": 0.0,
        "main_mse": 0.0,
        "aux_mse": 0.0,
        "alpha_mean": 0.0,
        "sim_gate_mean": 0.0,
    }
    valid_steps = 0
    nan_batches = 0

    loop = tqdm(train_loader, desc="Training", leave=False)
    use_amp = should_use_amp(config, device)

    for batch in loop:
        batch_x, batch_y, prompt_embeddings = unpack_batch(batch, device, config)
        batch_x_mark, batch_y_mark, x_dec = build_decoder_inputs(
            batch_x, batch_y, config.train_horizon, config.label_len, device
        )
        
        optimizer.zero_grad(set_to_none=True)

        # with autocast(device_type=device.type, enabled=use_amp):
        #     outputs = forward_train_step(
        #         model, batch_x, batch_x_mark, x_dec, batch_y_mark, config, prompt_embeddings=prompt_embeddings
        #     )
        #     batch_y_target = batch_y[:, : config.train_horizon, :]
        #     loss, details = compute_training_loss(outputs, batch_y_target, criterion, config)
        with autocast(device_type=device.type, enabled=use_amp):
            batch_y_target = batch_y[:, : config.train_horizon, :]
            outputs = forward_train_step(
                model, batch_x, batch_x_mark, x_dec, batch_y_mark, config, prompt_embeddings=prompt_embeddings
            )
            loss, details = compute_training_loss(outputs, batch_y_target, criterion, config)
        if not torch.isfinite(loss):
            nan_batches += 1
            optimizer.zero_grad(set_to_none=True)
            loop.set_postfix(loss="nan", skipped=nan_batches)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        for key in metric_sums:
            value = details[key]
            if value == value:
                metric_sums[key] += value
        valid_steps += 1

        postfix = {"loss": loss.item(), "main_mse": details["main_mse"]}
        if is_pac_llm_variant(config):
            postfix["aux_mse"] = details["aux_mse"]
            postfix["alpha"] = details["alpha_mean"]
            postfix["sim_gate"] = details["sim_gate_mean"]
        loop.set_postfix(**postfix)

    if valid_steps == 0:
        return {
            "total": float("inf"),
            "mse": float("inf"),
            "diff": float("inf"),
            "main_mse": float("inf"),
            "aux_mse": float("inf"),
            "alpha_mean": float("inf"),
            "sim_gate_mean": float("inf"),
            "nan_batches": nan_batches,
        }

    avg_loss = total_loss / valid_steps
    avg_details = {k: v / valid_steps for k, v in metric_sums.items()}
    if not is_pac_llm_variant(config):
        avg_details["aux_mse"] = float("nan")
        avg_details["alpha_mean"] = float("nan")
        avg_details["sim_gate_mean"] = float("nan")
    return {"total": avg_loss, **avg_details, "nan_batches": nan_batches}



def rollout_predict(model, batch_x, config, device, target_horizon=None, prompt_embeddings=None):
    history = batch_x.clone()
    preds = []
    rollout_steps = config.eval_horizon if target_horizon is None else target_horizon

    if uses_predict_interface(config):
        generated = 0
        while generated < rollout_steps:
            chunk = min(config.train_horizon, rollout_steps - generated)
            step_pred = model.predict(history, prediction_length=chunk)
            step_pred = step_pred[:, :chunk, :]
            preds.append(step_pred)
            history = torch.cat([history, step_pred], dim=1)[:, -config.input_size :, :]
            generated += chunk
        return torch.cat(preds, dim=1)

    generated = 0
    while generated < rollout_steps:
        chunk = min(config.train_horizon, rollout_steps - generated)

        batch_x_mark, batch_y_mark, x_dec = build_decoder_inputs(
            history, history[:, -1:, :], config.train_horizon, config.label_len, device
        )

        if is_pac_llm_variant(config):
            step_pred = model(history, batch_x_mark, x_dec, batch_y_mark, return_aux=False)
        else:
            step_pred = model(history, batch_x_mark, x_dec, batch_y_mark)

        step_pred = step_pred[:, :chunk, :]
        step_pred = step_pred.to(history.device)
        preds.append(step_pred)
        history = torch.cat([history, step_pred], dim=1)[:, -config.input_size :, :]
        generated += chunk

    return torch.cat(preds, dim=1)

def validate_rollout_batch(model, data_loader, device, config, target_horizon):
    model.eval()
    use_amp = should_use_amp(config, device)

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    valid_steps = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Rollout Eval", leave=False):
            batch_x, batch_y, prompt_embeddings = unpack_batch(batch, device, config)

            with autocast(device_type=device.type, enabled=use_amp):
                rollout_preds = rollout_predict(
                    model, batch_x, config, device, target_horizon=target_horizon, prompt_embeddings=prompt_embeddings
                )

            if not torch.isfinite(rollout_preds).all():
                continue

            pred_h = rollout_preds[:, :target_horizon, :]
            true_h = batch_y[:, :target_horizon, :]
            mse = torch.mean((pred_h - true_h) ** 2)
            mae = torch.mean(torch.abs(pred_h - true_h))

            if not (torch.isfinite(mse) and torch.isfinite(mae)):
                continue

            total_loss += mse.item()
            total_mse += mse.item()
            total_mae += mae.item()
            valid_steps += 1

    if valid_steps == 0:
        return {"total": float("inf"), "mse": float("inf"), "mae": float("inf")}

    return {
        "total": total_loss / valid_steps,
        "mse": total_mse / valid_steps,
        "mae": total_mae / valid_steps,
    }


def evaluate_rollout_global(model, data_loader, device, config, horizons=None):
    model.eval()
    use_amp = should_use_amp(config, device)

    horizons = config.rollout_horizons if horizons is None else horizons
    max_horizon = max(horizons)
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Rollout Eval", leave=False):
            batch_x, batch_y, prompt_embeddings = unpack_batch(batch, device, config)

            with autocast(device_type=device.type, enabled=use_amp):
                rollout_preds = rollout_predict(
                    model, batch_x, config, device, target_horizon=max_horizon, prompt_embeddings=prompt_embeddings
                )

            if not torch.isfinite(rollout_preds).all():
                continue

            all_preds.append(rollout_preds.detach().cpu())
            all_trues.append(batch_y[:, :max_horizon, :].detach().cpu())

    metrics = {}
    avg_mse = 0.0
    valid_horizons = 0
    if not all_preds:
        for horizon in horizons:
            metrics[horizon] = {"mse": float("inf"), "mae": float("inf")}
        metrics["avg_mse"] = float("inf")
        return metrics

    full_preds = torch.cat(all_preds, dim=0)
    full_trues = torch.cat(all_trues, dim=0)

    for horizon in horizons:
        pred_h = full_preds[:, :horizon, :]
        true_h = full_trues[:, :horizon, :]
        mse = torch.mean((pred_h - true_h) ** 2)
        mae = torch.mean(torch.abs(pred_h - true_h))

        if not (torch.isfinite(mse) and torch.isfinite(mae)):
            metrics[horizon] = {"mse": float("inf"), "mae": float("inf")}
            continue
        mse_value = mse.item()
        mae_value = mae.item()
        metrics[horizon] = {"mse": mse_value, "mae": mae_value}
        avg_mse += mse_value
        valid_horizons += 1

    metrics["avg_mse"] = avg_mse / max(valid_horizons, 1)
    return metrics


def collect_rollout_predictions_normalized(model, data_loader, device, config, target_horizon):
    model.eval()
    preds = []
    trues = []
    use_amp = should_use_amp(config, device)


    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting Norm", leave=False):
            batch_x, batch_y, prompt_embeddings = unpack_batch(batch, device, config)

            with autocast(device_type=device.type, enabled=use_amp):
                rollout_preds = rollout_predict(
                    model, batch_x, config, device, target_horizon=target_horizon, prompt_embeddings=prompt_embeddings
                )

            pred_h = rollout_preds[:, :target_horizon, :]
            true_h = batch_y[:, :target_horizon, :]
            if not torch.isfinite(pred_h).all():
                continue

            preds.append(pred_h.detach().cpu().numpy())
            trues.append(true_h.detach().cpu().numpy())

    if not preds:
        empty = np.empty((0, target_horizon, config.feat_dim))
        return empty, empty

    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

def collect_rollout_visualization_cache(model, data_loader, device, config, target_horizon, data_scaler=None):
    model.eval()
    histories = []
    preds = []
    trues = []
    use_amp = should_use_amp(config, device)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting Viz", leave=False):
            batch_x, batch_y, prompt_embeddings = unpack_batch(batch, device, config)

            with autocast(device_type=device.type, enabled=use_amp):
                rollout_preds = rollout_predict(
                    model, batch_x, config, device, target_horizon=target_horizon, prompt_embeddings=prompt_embeddings
                )

            pred_h = rollout_preds[:, :target_horizon, :]
            true_h = batch_y[:, :target_horizon, :]
            if not torch.isfinite(pred_h).all():
                continue

            histories.append(batch_x.detach().cpu().numpy())
            preds.append(pred_h.detach().cpu().numpy())
            trues.append(true_h.detach().cpu().numpy())

    empty_history = np.empty((0, config.lookback, config.feat_dim), dtype=np.float32)
    empty_future = np.empty((0, target_horizon, config.feat_dim), dtype=np.float32)
    if not preds:
        return {
            "histories_norm": empty_history,
            "preds_norm": empty_future,
            "targets_norm": empty_future,
            "sample_ids": np.empty((0,), dtype=np.int64),
        }

    histories_norm = np.concatenate(histories, axis=0)
    preds_norm = np.concatenate(preds, axis=0)
    targets_norm = np.concatenate(trues, axis=0)
    cache = {
        "histories_norm": histories_norm,
        "preds_norm": preds_norm,
        "targets_norm": targets_norm,
        "sample_ids": np.arange(preds_norm.shape[0], dtype=np.int64),
    }

    if data_scaler is not None:
        cache["histories_raw"] = inverse_scale_numpy(histories_norm, data_scaler)
        cache["preds_raw"] = inverse_scale_numpy(preds_norm, data_scaler)
        cache["targets_raw"] = inverse_scale_numpy(targets_norm, data_scaler)

    return cache
def save_rollout_visualization_cache(config, cache, rollout_horizon, metadata=None):
    if not getattr(config, "export_viz_cache", False):
        return None

    os.makedirs(config.viz_cache_dir, exist_ok=True)
    cache_path = os.path.join(config.viz_cache_dir, f"{config.model_variant}_{config.system}.npz")
    payload = {
        "system": config.system,
        "model_variant": config.model_variant,
        "lookback": np.int64(config.lookback),
        "rollout_horizon": np.int64(rollout_horizon),
        "rollout_horizons": np.asarray(config.rollout_horizons, dtype=np.int64),
        **cache,
    }
    if metadata:
        for key, value in metadata.items():
            payload[key] = value
    np.savez_compressed(cache_path, **payload)
    print(f"Saved visualization cache to: {cache_path}")
    return cache_path
def _to_python_floats(value):
    if isinstance(value, dict):
        return {key: _to_python_floats(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_floats(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_benchmark_results(config, vali_metrics, test_metrics, final_metrics):
    os.makedirs(config.results_dir, exist_ok=True)
    result_path = os.path.join(config.results_dir, f"{config.model_variant}_{config.system}.json")
    payload = {
        "system": config.system,
        "model_variant": config.model_variant,
        #"data_path": config.data_path,
        "benchmark_split_mode": config.benchmark_split_mode,
        "train_val_ratio": config.train_val_ratio,
        "num_test_trajectories": config.num_test_trajectories,
        "lookback": config.lookback,
        "train_horizon": config.train_horizon,
        "eval_horizon": config.eval_horizon,
        "rollout_horizons": list(config.rollout_horizons),
        "train_window_stride": config.train_window_stride,
        "eval_window_stride": config.eval_window_stride,
        "validation": _to_python_floats(vali_metrics),
        "test": _to_python_floats(test_metrics),
        "final_normalized": _to_python_floats(final_metrics),
    }
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved benchmark summary to: {result_path}")
def print_horizon_table(title, metrics):
    print(title)
    print("-" * 60)
    for horizon in metrics:
        if horizon == "avg_mse":
            continue
        print(f"{horizon:>2}-step | MAE: {metrics[horizon]['mae']:.6f} | MSE: {metrics[horizon]['mse']:.6f}")
    print("-" * 60)
    print(f"Average rollout MSE: {metrics['avg_mse']:.6f}")
def compute_final_normalized_metrics(preds_norm, targets_norm, config, data_scaler=None):
    return {
        "mse": MSE(preds_norm.flatten(), targets_norm.flatten()),
        "mae": MAE(preds_norm.flatten(), targets_norm.flatten()),
        "smape": SMAPE(targets_norm, preds_norm),
        "vpt": VPT(targets_norm, preds_norm),
        "kld": KLD(targets_norm, preds_norm),
        "corr_dim_error": gpdim_rmse(targets_norm, preds_norm),
    }


def print_final_metric_table(title, metrics_by_horizon):
    print(title)
    print("-" * 96)
    print(
        f"{'Horizon':>8} | {'MSE':>10} | {'MAE':>10} | {'SMAPE':>10} | "
        f"{'VPT':>10} | {'KLD':>10} | {'CorrDim':>10}"
    )
    print("-" * 96)
    for horizon, metrics in metrics_by_horizon.items():
        print(
            f"{int(horizon):>8}-step | "
            f"{metrics['mse']:>10.6f} | "
            f"{metrics['mae']:>10.6f} | "
            f"{metrics['smape']:>10.6f} | "
            f"{metrics['vpt']:>10.6f} | "
            f"{metrics['kld']:>10.6f} | "
            f"{metrics['corr_dim_error']:>10.6f}"
        )
    print("-" * 96)
def train_main(evaluate_after_training=False):
    fix_random_seed(seed=2025)
    config = apply_cli_overrides(Config())
    config.checkpoint_dir = get_checkpoint_dir(config)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    if getattr(config, "export_viz_cache", False):
        os.makedirs(config.viz_cache_dir, exist_ok=True)
    print(f"System: {config.system} | Device: {config.device}")
    print(f"Variant: {config.model_variant}")
    print(
        f"Paper rollout setting | train_horizon={config.train_horizon}, "
        f"eval_horizon={config.eval_horizon}, valid_horizon={config.valid_horizon}, "
        f"rollout_horizons={config.rollout_horizons}, "
        f"window_stride(train/eval)={config.train_window_stride}/{config.eval_window_stride}"
    )

    
    train_loader, vali_loader, test_loader, data_scaler = load_data(config)
    sample_batch = next(iter(train_loader))
    print("system:", config.system)
    #print("data_path:", config.data_path)
    print("sample_x shape:", sample_batch[0].shape)
    print("sample_y shape:", sample_batch[1].shape)
    print("feat_dim:", config.feat_dim)
    print("enc_in/dec_in/c_out:", config.enc_in, config.dec_in, config.c_out)

    if getattr(config, "benchmark_efficiency", False):
        efficiency_batch = next(iter(vali_loader))
        efficiency_results = collect_efficiency_results(config, efficiency_batch)
        print_efficiency_table(efficiency_results)

    model = build_model(config, sample_batch=sample_batch).to(config.device)

  
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {model_params}")

    criterion = ForecastLoss(
        lambda_mse=config.lambda_mse,
        lambda_diff=config.lambda_diff,
    ).to(config.device)
    if is_nvar_variant(config):
        print("Fitting nVAR baseline with original ridge-regression training...")
        model.fit(train_loader, config.device)
    elif is_esn_variant(config):
        print("Fitting ESN baseline with closed-form ridge-regression readout...")
        model.fit(train_loader, config.device)
    elif not uses_predict_interface(config):
        learning_rate = config.learning_rate
        weight_decay = config.weight_decay
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = config.epochs * len(train_loader)
        warmup_steps = max(1, config.warmup_epochs * len(train_loader))
        warmup_scheduler = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        #scaler = GradScaler(device=config.device.type, enabled=config.device.type == "cuda")
        scaler = GradScaler(device=config.device.type, enabled=should_use_amp(config, config.device))
     
        best_vali_loss = float("inf")
        early_stop_counter = 0
        best_model_path = os.path.join(
            config.checkpoint_dir,
            f"{config.input_size}best_model_rollout_{config.model_variant}_{config.system}.pth",
        )

        for epoch in range(config.epochs):
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler, config.device, config
            )
            vali_metrics_batch = validate_rollout_batch(
                model, vali_loader, config.device, config, target_horizon=config.valid_horizon
            )

            train_extra = f"MainMSE:{train_metrics['main_mse']:.4f}"
            if is_pac_llm_variant(config):
                train_extra += (
                    f" AuxMSE:{summarize_metric(train_metrics['aux_mse'])}"
                    f" Alpha:{summarize_metric(train_metrics['alpha_mean'])}"
                    f" SimGate:{summarize_metric(train_metrics['sim_gate_mean'])}"
                )

            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train: {train_metrics['total']:.4f} ({train_extra}) | "
                f"Valid: {vali_metrics_batch['total']:.4f} "
                f"(MSE:{vali_metrics_batch['mse']:.4f} MAE:{vali_metrics_batch['mae']:.4f}) | "
                f"Skipped:{train_metrics.get('nan_batches', 0)} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if vali_metrics_batch["mse"] < best_vali_loss:
                best_vali_loss = vali_metrics_batch["mse"]
                early_stop_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  --> Best {config.model_variant} model ({config.valid_horizon}-step batch validation) saved.")
            else:
                early_stop_counter += 1

            if early_stop_counter >= config.patience:
                print("Early stopping triggered.")
                break

        print(f"Loading best {config.model_variant} model for final rollout evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    elif uses_predict_interface(config):
        print(f"{config.model_variant} runs in zero-shot mode and skips supervised training.")

    if not evaluate_after_training:
        print("Training finished. Run `python evaluate.py` to compute rollout metrics from the saved checkpoint.")
        return

    vali_metrics = evaluate_rollout_global(model, vali_loader, config.device, config, horizons=config.rollout_horizons)
    test_metrics = evaluate_rollout_global(model, test_loader, config.device, config, horizons=config.rollout_horizons)
    print_horizon_table("Validation Rollout Metrics", vali_metrics)
    print_horizon_table("Test Rollout Metrics", test_metrics)

    final_horizon = config.rollout_horizons[-1]
    final_preds_norm, final_targets_norm = collect_rollout_predictions_normalized(
         model, test_loader, config.device, config, final_horizon
     )
    viz_cache = collect_rollout_visualization_cache(
        model,
        test_loader,
        config.device,
        config,
        final_horizon,
        data_scaler=data_scaler,
    )
    final_preds_norm = viz_cache["preds_norm"]
    final_targets_norm = viz_cache["targets_norm"]
    final_metrics = {}
    final_metrics_by_horizon = {}
    if len(final_preds_norm) > 0:
        for horizon in config.rollout_horizons:
            pred_h = final_preds_norm[:, :horizon, :]
            true_h = final_targets_norm[:, :horizon, :]
            final_metrics_by_horizon[horizon] = compute_final_normalized_metrics(
                pred_h,
                true_h,
                config,
                data_scaler=data_scaler,
            )
        final_metrics = final_metrics_by_horizon[final_horizon]
        print_final_metric_table(
            f"Final {config.model_variant} rollout metrics are computed in normalized space.",
            final_metrics_by_horizon,
        )
    save_rollout_visualization_cache(
        config,
        viz_cache,
        rollout_horizon=final_horizon,
        metadata=resolve_dataset_metadata(config),
    )
    save_benchmark_results(config, vali_metrics, test_metrics, final_metrics)


def evaluate_main():
    fix_random_seed(seed=2025)
    config = apply_cli_overrides(Config())
    config.checkpoint_dir = get_checkpoint_dir(config)

    os.makedirs(config.results_dir, exist_ok=True)
    if getattr(config, "export_viz_cache", False):
        os.makedirs(config.viz_cache_dir, exist_ok=True)

    print(f"System: {config.system} | Device: {config.device}")
    print(f"Evaluating variant: {config.model_variant}")

    train_loader, vali_loader, test_loader, data_scaler = load_data(config)
    sample_batch = next(iter(train_loader))

    model = build_model(config, sample_batch=sample_batch).to(config.device)

    if is_nvar_variant(config):
        print("Fitting nVAR baseline with original ridge-regression training before evaluation...")
        model.fit(train_loader, config.device)
    elif is_esn_variant(config):
        print("Fitting ESN baseline with closed-form ridge-regression readout before evaluation...")
        model.fit(train_loader, config.device)
    elif uses_predict_interface(config):
        print(f"{config.model_variant} runs in zero-shot mode and skips checkpoint loading.")
    else:
        best_model_path = os.path.join(
            config.checkpoint_dir,
            f"{config.input_size}best_model_rollout_{config.model_variant}_{config.system}.pth",
        )
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Cannot find checkpoint for evaluation: {best_model_path}. "
                "Run `python train.py` first or update Config.checkpoint_root/model_variant/system."
            )
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))
        print(f"Loaded checkpoint: {best_model_path}")

    vali_metrics = evaluate_rollout_global(model, vali_loader, config.device, config, horizons=config.rollout_horizons)
    test_metrics = evaluate_rollout_global(model, test_loader, config.device, config, horizons=config.rollout_horizons)
    print_horizon_table("Validation Rollout Metrics", vali_metrics)
    print_horizon_table("Test Rollout Metrics", test_metrics)

    final_horizon = config.rollout_horizons[-1]
    viz_cache = collect_rollout_visualization_cache(
        model,
        test_loader,
        config.device,
        config,
        final_horizon,
        data_scaler=data_scaler,
    )
    final_preds_norm = viz_cache["preds_norm"]
    final_targets_norm = viz_cache["targets_norm"]
    final_metrics = {}
    final_metrics_by_horizon = {}
    if len(final_preds_norm) > 0:
        for horizon in config.rollout_horizons:
            pred_h = final_preds_norm[:, :horizon, :]
            true_h = final_targets_norm[:, :horizon, :]
            final_metrics_by_horizon[horizon] = compute_final_normalized_metrics(
                pred_h,
                true_h,
                config,
                data_scaler=data_scaler,
            )
        final_metrics = final_metrics_by_horizon[final_horizon]
        print_final_metric_table(
            f"Final {config.model_variant} rollout metrics are computed in normalized space.",
            final_metrics_by_horizon,
        )

    save_rollout_visualization_cache(
        config,
        viz_cache,
        rollout_horizon=final_horizon,
        metadata=resolve_dataset_metadata(config),
    )
    save_benchmark_results(config, vali_metrics, test_metrics, final_metrics)


def main():
    train_main(evaluate_after_training=True)


if __name__ == "__main__":
    main()
