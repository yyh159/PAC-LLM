import itertools

import numpy as np
import torch
import torch.nn as nn


class SpatioTemporalNVAR(nn.Module):
    def __init__(
        self,
        h,
        input_size,
        feat_dim,
        k=1,
        ridge_param=1e-2,
        include_constant=True,
        clip_value=5.0,
    ):
        super().__init__()
        self.h = h
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.k = k
        self.ridge_param = ridge_param
        self.include_constant = include_constant
        self.clip_value = clip_value
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)
        self.W_out = None
        self.mean_ = None
        self.std_ = None
        self.linear_dim = self.k * self.feat_dim
        self.quad_pairs = list(itertools.combinations_with_replacement(range(self.linear_dim), 2))
        self.feature_dim = self.linear_dim + len(self.quad_pairs) + (1 if self.include_constant else 0)

    @property
    def outputs_normalized(self):
        return True

    def _build_linear_state(self, history_np):
        batch_size = history_np.shape[0]
        linear = np.zeros((batch_size, self.linear_dim), dtype=np.float64)
        for delay in range(self.k):
            linear[:, delay * self.feat_dim : (delay + 1) * self.feat_dim] = history_np[:, -(delay + 1), :]
        return linear

    def _build_feature_matrix(self, history_np):
        linear = self._build_linear_state(history_np)
        feats = []
        if self.include_constant:
            feats.append(np.ones((linear.shape[0], 1), dtype=np.float64))
        feats.append(linear)
        quadratic = np.empty((linear.shape[0], len(self.quad_pairs)), dtype=np.float64)
        for idx, (row, col) in enumerate(self.quad_pairs):
            quadratic[:, idx] = linear[:, row] * linear[:, col]
        feats.append(quadratic)
        return np.concatenate(feats, axis=1)

    @torch.no_grad()
    def fit(self, train_loader, device):
        history_rows = []
        feature_rows = []
        delta_rows = []
        for batch in train_loader:
            batch_x, batch_y = batch[:2]
            history = batch_x.detach().cpu().numpy().astype(np.float64)
            history_rows.append(history)

        if not history_rows:
            raise RuntimeError("nVAR fit received an empty training loader.")

        stacked_history = np.concatenate(history_rows, axis=0)
        self.mean_ = stacked_history.mean(axis=(0, 1), keepdims=True)
        self.std_ = stacked_history.std(axis=(0, 1), keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0

        for batch in train_loader:
            batch_x, batch_y = batch[:2]
            history = batch_x.detach().cpu().numpy().astype(np.float64)
            target = batch_y[:, 0, :].detach().cpu().numpy().astype(np.float64)
            history_norm = self._normalize(history)
            target_norm = self._normalize(target[:, None, :])[:, 0, :]
            features = self._build_feature_matrix(history_norm)
            delta = target_norm - history_norm[:, -1, :]
            feature_rows.append(features)
            delta_rows.append(delta)

        Phi = np.concatenate(feature_rows, axis=0)
        Delta = np.concatenate(delta_rows, axis=0)
        gram = Phi.T @ Phi + self.ridge_param * np.eye(self.feature_dim, dtype=np.float64)
        self.W_out = np.linalg.solve(gram, Phi.T @ Delta)
        return self

    def _normalize(self, x):
        return (x - self.mean_) / self.std_

    def _denormalize(self, x):
        return x * self.std_ + self.mean_

    def _predict_one_step_np(self, history_np):
        features = self._build_feature_matrix(history_np)
        delta = features @ self.W_out
        next_step = history_np[:, -1, :] + delta
        next_step = np.nan_to_num(next_step, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        return np.clip(next_step, -self.clip_value, self.clip_value)

    @torch.no_grad()
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.W_out is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("nVAR model is not fitted. Call fit(train_loader, device) before inference.")

        history = x_enc.detach().cpu().numpy().astype(np.float64)
        history = self._normalize(history)
        preds = []
        for _ in range(self.h):
            next_step = self._predict_one_step_np(history)
            preds.append(next_step)
            history = np.concatenate([history[:, 1:, :], next_step[:, None, :]], axis=1)

        # Keep outputs in normalized space so rollout/loss metrics stay comparable
        # with the rest of the baselines in the current training pipeline.
        pred_np = np.stack(preds, axis=1).astype(np.float32)
        return torch.from_numpy(pred_np).to(x_enc.device, dtype=x_enc.dtype)
