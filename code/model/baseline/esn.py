import numpy as np
import torch
import torch.nn as nn


class ESNAdapter(nn.Module):
    def __init__(
        self,
        h,
        input_size,
        feat_dim,
        reservoir_size=500,
        spectral_radius=0.9,
        input_scaling=0.5,
        leak_rate=1.0,
        sparsity=0.9,
        ridge=1e-4,
        washout=5,
        bias_scale=0.2,
        random_seed=2025,
        clip_value=5.0,
    ):
        super().__init__()
        self.h = h
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.sparsity = sparsity
        self.ridge = ridge
        self.washout = washout
        self.bias_scale = bias_scale
        self.random_seed = random_seed
        self.clip_value = clip_value
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

        rng = np.random.default_rng(random_seed)
        w_in = rng.uniform(
            -input_scaling,
            input_scaling,
            size=(reservoir_size, feat_dim),
        ).astype(np.float64)
        w_bias = rng.uniform(
            -bias_scale,
            bias_scale,
            size=(reservoir_size,),
        ).astype(np.float64)
        w_res = rng.uniform(-1.0, 1.0, size=(reservoir_size, reservoir_size)).astype(np.float64)
        mask = rng.random(size=(reservoir_size, reservoir_size)) < (1.0 - sparsity)
        w_res *= mask
        eigvals = np.linalg.eigvals(w_res)
        max_eig = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 1.0
        if max_eig < 1e-8:
            max_eig = 1.0
        w_res *= spectral_radius / max_eig

        self.register_buffer("W_in", torch.from_numpy(w_in).float(), persistent=False)
        self.register_buffer("W_bias", torch.from_numpy(w_bias).float(), persistent=False)
        self.register_buffer("W_res", torch.from_numpy(w_res).float(), persistent=False)

        self.readout = None

    @property
    def outputs_normalized(self):
        return True

    def _step(self, state, current_input):
        preact = (
            current_input @ self.W_in.t()
            + state @ self.W_res.t()
            + self.W_bias.unsqueeze(0)
        )
        candidate = torch.tanh(preact)
        return (1.0 - self.leak_rate) * state + self.leak_rate * candidate

    def _consume_history(self, history):
        batch_size, seq_len, _ = history.shape
        state = torch.zeros(
            batch_size,
            self.reservoir_size,
            device=history.device,
            dtype=history.dtype,
        )
        for t in range(seq_len):
            state = self._step(state, history[:, t, :])
        return state

    @torch.no_grad()
    def fit(self, train_loader, device):
        state_rows = []
        target_rows = []
        for batch in train_loader:
            batch_x, batch_y = batch[:2]
            history = batch_x.float().to(device)
            target = batch_y[:, 0, :].float().to(device)
            state = self._consume_history(history)
            state_rows.append(state.detach().cpu().numpy().astype(np.float64))
            target_rows.append(target.detach().cpu().numpy().astype(np.float64))

        if not state_rows:
            raise RuntimeError("ESN fit received an empty training loader.")

        X = np.concatenate(state_rows, axis=0)
        Y = np.concatenate(target_rows, axis=0)
        gram = X.T @ X + self.ridge * np.eye(X.shape[1], dtype=np.float64)
        weights = np.linalg.solve(gram, X.T @ Y)
        self.readout = torch.from_numpy(weights.astype(np.float32)).to(device)
        return self

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if self.readout is None:
            raise RuntimeError("ESN model is not fitted. Call fit(train_loader, device) before inference.")

        history = x_enc
        state = self._consume_history(history)
        current_input = history[:, -1, :]
        preds = []
        for _ in range(self.h):
            state = self._step(state, current_input)
            next_step = state @ self.readout
            next_step = torch.nan_to_num(
                next_step,
                nan=0.0,
                posinf=self.clip_value,
                neginf=-self.clip_value,
            )
            next_step = torch.clamp(next_step, -self.clip_value, self.clip_value)
            preds.append(next_step)
            current_input = next_step
        return torch.stack(preds, dim=1)
