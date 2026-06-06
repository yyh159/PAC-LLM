from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


def _resolve_dynamix_root(explicit_root: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))

    this_file = Path(__file__).resolve()
    repo_root = this_file.parent
    candidates.extend(
        [
            repo_root / "external" / "dynamix",
            repo_root / "panda-main" / "external" / "dynamix",
            repo_root.parent / "panda-main" / "external" / "dynamix",
            Path.cwd() / "external" / "dynamix",
        ]
    )

    for candidate in candidates:
        src_path = candidate / "src"
        if src_path.exists():
            for path in (candidate, src_path):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "DynaMix sources not found. Checked: "
        f"{searched}. Initialize the DynaMix submodule or set `dynamix_root` explicitly."
    )


def _resolve_panda_root(explicit_root: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))

    this_file = Path(__file__).resolve()
    repo_root = this_file.parent
    candidates.extend(
        [
            repo_root / "panda-main",
            repo_root.parent / "panda-main",
            Path.cwd() / "panda-main",
        ]
    )

    for candidate in candidates:
        pipeline_path = candidate / "panda" / "patchtst" / "pipeline.py"
        if pipeline_path.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Panda sources not found. Checked: "
        f"{searched}. Place `panda-main` next to this project or set `panda_root` explicitly."
    )


class ZeroShotPipelineBase(nn.Module):
    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)

    def to(self, device: str | torch.device):  # type: ignore[override]
        self.device = torch.device(device)
        return self


def _flatten_btc_series(series: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    if series.ndim != 3:
        raise ValueError("Expected tensor with shape (batch_size, time_length, dim)")
    batch_size, time_length, feat_dim = series.shape
    flat = series.permute(0, 2, 1).reshape(batch_size * feat_dim, time_length)
    return flat, batch_size, time_length, feat_dim


def _restore_flat_predictions(
    flat_preds: torch.Tensor,
    batch_size: int,
    feat_dim: int,
    prediction_length: int,
) -> torch.Tensor:
    return flat_preds.reshape(batch_size, feat_dim, prediction_length).permute(0, 2, 1)


class DynaMixPipelineAdapter(ZeroShotPipelineBase):
    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        preprocessing_method: Literal[
            "pos_embedding",
            "zero_embedding",
            "delay_embedding",
            "delay_embedding_random",
        ] = "delay_embedding",
        standardize: bool = True,
        fit_nonstationary: bool = False,
        dynamix_root: str | None = None,
    ) -> None:
        super().__init__(device=device)
        self.torch_dtype = torch_dtype
        self.preprocessing_method = preprocessing_method
        self.standardize = standardize
        self.fit_nonstationary = fit_nonstationary
        _resolve_dynamix_root(dynamix_root)

        try:
            from dynamix.model.forecaster import DynaMixForecaster  # type: ignore
            from dynamix.utilities.utilities import load_hf_model  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import DynaMix modules after resolving the source path.") from exc

        self.model = load_hf_model(model_name)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device, dtype=self.torch_dtype)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.forecaster = DynaMixForecaster(self.model)

    def to(self, device: str | torch.device):  # type: ignore[override]
        super().to(device)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device, dtype=self.torch_dtype)
        return self

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        preprocessing_method: str | None = None,
        standardize: bool | None = None,
        fit_nonstationary: bool | None = None,
        initial_x: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("Expected context tensor with shape (batch_size, context_length, dim)")

        method = preprocessing_method or self.preprocessing_method
        do_standardize = self.standardize if standardize is None else standardize
        do_fit_nonstationary = self.fit_nonstationary if fit_nonstationary is None else fit_nonstationary

        seq = context.to(self.device, dtype=self.torch_dtype).permute(1, 0, 2)
        preds = self.forecaster.forecast(
            context=seq,
            horizon=prediction_length,
            preprocessing_method=method,
            standardize=do_standardize,
            fit_nonstationary=do_fit_nonstationary,
            initial_x=initial_x,
        )
        return preds.permute(1, 0, 2).to(context.device, dtype=context.dtype)


class PandaPipelineAdapter(ZeroShotPipelineBase):
    def __init__(
        self,
        pretrain_path: str,
        device: str | torch.device = "cpu",
        device_map: str | None = None,
        limit_prediction_length: bool = False,
        sliding_context: bool = True,
        panda_root: str | None = None,
    ) -> None:
        super().__init__(device=device)
        self.limit_prediction_length = limit_prediction_length
        self.sliding_context = sliding_context
        self.panda_root = _resolve_panda_root(panda_root)

        try:
            from panda.patchtst.pipeline import PatchTSTPipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed to import Panda PatchTST pipeline after resolving the source path."
            ) from exc

        try:
            self.pipeline = PatchTSTPipeline.from_pretrained(
                mode="predict",
                pretrain_path=pretrain_path,
                device_map=device_map or str(self.device),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Panda pretrained pipeline. Check dependencies and pretrained weights."
            ) from exc

        model = getattr(self.pipeline, "model", None)
        if model is not None and hasattr(model, "eval"):
            model.eval()

    def to(self, device: str | torch.device):  # type: ignore[override]
        super().to(device)
        model = getattr(self.pipeline, "model", None)
        if model is not None and hasattr(model, "to"):
            model.to(self.device)
        return self

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        limit_prediction_length: bool | None = None,
        sliding_context: bool | None = None,
        **_: object,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("Expected context tensor with shape (batch_size, context_length, dim)")

        outputs = self.pipeline.predict(
            context=context.to(self.device),
            prediction_length=prediction_length,
            limit_prediction_length=(
                self.limit_prediction_length if limit_prediction_length is None else limit_prediction_length
            ),
            sliding_context=self.sliding_context if sliding_context is None else sliding_context,
            verbose=False,
        )

        if outputs.ndim != 4:
            raise ValueError(
                f"Unexpected Panda prediction shape {tuple(outputs.shape)}. Expected (batch, samples, horizon, dim)."
            )

        preds = outputs.median(dim=1).values
        return preds[:, :prediction_length, :].to(context.device, dtype=context.dtype)


class ChronosPipelineAdapter(ZeroShotPipelineBase):
    def __init__(
        self,
        pretrain_path: str,
        device: str | torch.device = "cpu",
        device_map: str | None = None,
        limit_prediction_length: bool = False,
        num_samples: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        deterministic: bool = False,
        panda_root: str | None = None,
    ) -> None:
        super().__init__(device=device)
        self.limit_prediction_length = limit_prediction_length
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.deterministic = deterministic
        self.panda_root = _resolve_panda_root(panda_root)

        try:
            from panda.chronos.pipeline import ChronosPipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import Panda Chronos pipeline after resolving the source path.") from exc

        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                pretrain_path,
                device_map=device_map or str(self.device),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Chronos pretrained pipeline. Check dependencies and pretrained weights."
            ) from exc

        model = getattr(self.pipeline, "model", None)
        if model is not None and hasattr(model, "eval"):
            model.eval()

    def to(self, device: str | torch.device):  # type: ignore[override]
        super().to(device)
        model = getattr(self.pipeline, "model", None)
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "to"):
            inner.to(self.device)
        return self

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        limit_prediction_length: bool | None = None,
        num_samples: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        deterministic: bool | None = None,
        **_: object,
    ) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("Expected context tensor with shape (batch_size, context_length, dim)")

        batch_size, context_length, feat_dim = context.shape
        # Chronos tokenization uses CPU-resident bucket boundaries internally, so keep
        # preprocessing on CPU and let the pipeline move token ids to the model device.
        flat_context = context.detach().cpu().permute(0, 2, 1).reshape(batch_size * feat_dim, context_length)
        outputs = self.pipeline.predict(
            context=flat_context,
            prediction_length=prediction_length,
            num_samples=self.num_samples if num_samples is None else num_samples,
            temperature=self.temperature if temperature is None else temperature,
            top_k=self.top_k if top_k is None else top_k,
            top_p=self.top_p if top_p is None else top_p,
            limit_prediction_length=(
                self.limit_prediction_length if limit_prediction_length is None else limit_prediction_length
            ),
            verbose=False,
            deterministic=self.deterministic if deterministic is None else deterministic,
        )

        if outputs.ndim != 3:
            raise ValueError(
                f"Unexpected Chronos prediction shape {tuple(outputs.shape)}. Expected (batch, samples, horizon)."
            )

        preds = outputs.median(dim=1).values
        preds = _restore_flat_predictions(preds, batch_size, feat_dim, prediction_length)
        return preds[:, :prediction_length, :].to(context.device, dtype=context.dtype)


class ChronosSFTAdapter(ZeroShotPipelineBase):
    def __init__(
        self,
        pretrain_path: str,
        device: str | torch.device = "cpu",
        device_map: str | None = None,
        limit_prediction_length: bool = False,
        num_samples: int | None = 1,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        deterministic: bool = True,
        panda_root: str | None = None,
    ) -> None:
        super().__init__(device=device)
        self.limit_prediction_length = limit_prediction_length
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.deterministic = deterministic
        self.panda_root = _resolve_panda_root(panda_root)

        try:
            from panda.chronos.pipeline import ChronosPipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to import Panda Chronos pipeline after resolving the source path.") from exc

        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                pretrain_path,
                device_map=device_map or str(self.device),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Chronos pretrained pipeline. Check dependencies and pretrained weights."
            ) from exc

        self.chronos_model = self.pipeline.model
        self.backbone = self.chronos_model.model
        self.tokenizer = self.pipeline.tokenizer
        if getattr(self.chronos_model.config, "model_type", None) != "seq2seq":
            raise NotImplementedError(
                "ChronosSFTAdapter currently supports seq2seq Chronos checkpoints only."
            )
        self.backbone.train()

    def to(self, device: str | torch.device):  # type: ignore[override]
        super().to(device)
        if hasattr(self.backbone, "to"):
            self.backbone.to(self.device)
        return self

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.backbone.train(mode)
        return self

    def eval(self):  # type: ignore[override]
        return self.train(False)

    def prepare_univariate_batch(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        flat_context, batch_size, _, feat_dim = _flatten_btc_series(context)
        flat_target, _, _, target_feat_dim = _flatten_btc_series(target)
        if target_feat_dim != feat_dim:
            raise ValueError("Context and target feature dimensions must match for Chronos SFT.")
        return flat_context, flat_target, batch_size, feat_dim

    def tokenize_for_training(
        self,
        contexts: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids, attention_mask, scale = self.tokenizer.context_input_transform(contexts)
        label_ids, _ = self.tokenizer.label_input_transform(targets, scale)
        return token_ids, attention_mask, label_ids

    def forward_loss(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 3:
            raise ValueError("Expected target tensor with shape (batch_size, prediction_length, dim)")

        flat_context, flat_target, _, _ = self.prepare_univariate_batch(context, target)
        token_ids, attention_mask, label_ids = self.tokenize_for_training(flat_context.cpu(), flat_target.cpu())
        outputs = self.backbone(
            input_ids=token_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=label_ids.to(self.device),
        )
        return outputs.loss

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        limit_prediction_length: bool | None = None,
        num_samples: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        deterministic: bool | None = None,
        **_: object,
    ) -> torch.Tensor:
        flat_context, batch_size, _, feat_dim = _flatten_btc_series(context)
        outputs = self.pipeline.predict(
            context=flat_context.cpu(),
            prediction_length=prediction_length,
            num_samples=self.num_samples if num_samples is None else num_samples,
            temperature=self.temperature if temperature is None else temperature,
            top_k=self.top_k if top_k is None else top_k,
            top_p=self.top_p if top_p is None else top_p,
            limit_prediction_length=(
                self.limit_prediction_length if limit_prediction_length is None else limit_prediction_length
            ),
            verbose=False,
            deterministic=self.deterministic if deterministic is None else deterministic,
        )

        if outputs.ndim != 3:
            raise ValueError(
                f"Unexpected Chronos prediction shape {tuple(outputs.shape)}. Expected (batch, samples, horizon)."
            )

        preds = outputs.median(dim=1).values
        preds = _restore_flat_predictions(preds, batch_size, feat_dim, prediction_length)
        return preds[:, :prediction_length, :].to(context.device, dtype=context.dtype)


class TimesFMPipelineAdapter(ZeroShotPipelineBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(device=kwargs.get("device", "cpu"))
        raise RuntimeError("TimesFMPipelineAdapter is not implemented in this repository.")
