import torch
import torch.nn as nn

from .pac_backbone import ChannelMixingLayer, PACLLMBackbone


class ChaosFeatureExtractor(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.num_features_per_channel = 5

    def output_dim(self, feat_dim):
        return self.num_features_per_channel * feat_dim

    def combined_output_dim(self, feat_dim):
        raw_dim = self.output_dim(feat_dim)
        takens_dim = self.takens_output_dim(feat_dim)
        return raw_dim + takens_dim

    def takens_output_dim(self, feat_dim):
        # mean + std pooling over Takens delay dimension
        return self.num_features_per_channel * 2 * feat_dim

    def forward(self, x):
        """
        Input:
            x: [B, T, D] or [B, T]
        Output:
            features: [B, D * num_features_per_channel] or [B, num_features_per_channel]
        """
        if x.dim() == 3:
            batch_size, seq_len, feat_dim = x.shape
            x_flat = x.permute(0, 2, 1).reshape(batch_size * feat_dim, seq_len)
            feats = self.compute_features(x_flat)
            return feats.reshape(batch_size, -1)
        return self.compute_features(x)

    def extract_takens_features(self, x_phase):
        """
        Input:
            x_phase: [B, T, F, M]
        Output:
            pooled Takens statistics: [B, F * (2 * num_features_per_channel)]
        """
        if x_phase.dim() != 4:
            raise ValueError(f"Expected x_phase with 4 dims [B, T, F, M], got {x_phase.shape}")

        batch_size, seq_len, feat_dim, takens_dim = x_phase.shape
        phase_flat = x_phase.permute(0, 2, 3, 1).reshape(batch_size * feat_dim * takens_dim, seq_len)
        takens_feats = self.compute_features(phase_flat)
        takens_feats = takens_feats.reshape(batch_size, feat_dim, takens_dim, self.num_features_per_channel)

        takens_mean = takens_feats.mean(dim=2)
        takens_std = takens_feats.std(dim=2, unbiased=False)
        pooled = torch.cat([takens_mean, takens_std], dim=-1)
        return pooled.reshape(batch_size, -1)

    def compute_features(self, x):
        """
        Extract lightweight chaos-aware summary features for each univariate series.
        x: [B, T]
        """
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = centered.pow(2).mean(dim=1, keepdim=True)
        std = torch.sqrt(var + self.eps)

        diff_x = x[:, 1:] - x[:, :-1] if x.size(1) > 1 else x.new_zeros(x.size(0), 1)
        diff_std = torch.sqrt(diff_x.pow(2).mean(dim=1, keepdim=True) + self.eps)

        second_diff = (
            diff_x[:, 1:] - diff_x[:, :-1] if diff_x.size(1) > 1 else x.new_zeros(x.size(0), 1)
        )
        second_diff_std = torch.sqrt(second_diff.pow(2).mean(dim=1, keepdim=True) + self.eps)

        kurtosis = centered.pow(4).mean(dim=1, keepdim=True) / (var.pow(2) + self.eps)
        abs_energy = x.pow(2).mean(dim=1, keepdim=True)
        cid_ce_norm = torch.sqrt(diff_x.pow(2).sum(dim=1, keepdim=True) + self.eps) / (std + self.eps)

        energy = x.pow(2)
        energy_prob = energy / (energy.sum(dim=1, keepdim=True) + self.eps)
        energy_entropy = -(energy_prob * torch.log(energy_prob + self.eps)).sum(dim=1, keepdim=True)

        autocorr_lag1 = self._autocorr(x, lag=1, mean=mean, var=var)
        autocorr_lag2 = self._autocorr(x, lag=2, mean=mean, var=var)
        spectral_entropy = self._spectral_entropy(x)

        features = [
    std,
    cid_ce_norm,
    autocorr_lag1,
    second_diff_std,
    spectral_entropy,
]
        return torch.cat(features, dim=1)

    def _autocorr(self, x, lag, mean, var):
        if x.size(1) <= lag:
            return x.new_zeros(x.size(0), 1)
        x1 = x[:, :-lag] - mean
        x2 = x[:, lag:] - mean
        autocov = (x1 * x2).mean(dim=1, keepdim=True)
        return autocov / (var + self.eps)

    def _spectral_entropy(self, x):
        if x.size(1) < 2:
            return x.new_zeros(x.size(0), 1)
        spectrum = torch.fft.rfft(x, dim=1)
        power = spectrum.abs().pow(2)
        power = power / (power.sum(dim=1, keepdim=True) + self.eps)
        return -(power * torch.log(power + self.eps)).sum(dim=1, keepdim=True)


class LearnableTakensEmbedding(nn.Module):
    def __init__(self, feat_dim, embed_dim, initial_delay=2.0):
        super().__init__()
        delay_template = torch.linspace(initial_delay, initial_delay * (embed_dim - 1), embed_dim - 1)
        self.raw_delays = nn.Parameter(delay_template.unsqueeze(1).repeat(1, feat_dim))

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        t_idx = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        x_perm = x.permute(0, 2, 1)
        embeddings = [x]

        safe_delays = torch.nan_to_num(self.raw_delays, nan=0.0, posinf=seq_len - 1, neginf=0.0)
        safe_delays = torch.clamp(torch.abs(safe_delays), 0, seq_len - 1)

        for delay_vec in safe_delays:
            target_idx = torch.clamp(t_idx.unsqueeze(1) - delay_vec.unsqueeze(0), 0, seq_len - 1)
            idx_low = target_idx.floor().long()
            idx_high = torch.clamp(idx_low + 1, 0, seq_len - 1)
            w_high = target_idx - idx_low.to(x.dtype)
            w_low = 1.0 - w_high

            gather_low = idx_low.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
            gather_high = idx_high.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)
            low_vals = torch.gather(x_perm, 2, gather_low)
            high_vals = torch.gather(x_perm, 2, gather_high)
            shifted_x = (w_low.transpose(0, 1).unsqueeze(0) * low_vals + w_high.transpose(0, 1).unsqueeze(0) * high_vals)
            shifted_x = shifted_x.permute(0, 2, 1)

            mask = (t_idx.unsqueeze(1) < delay_vec.unsqueeze(0)).to(x.dtype).unsqueeze(0)
            embeddings.append(shifted_x * (1.0 - mask))

        return torch.cat(embeddings, dim=-1)

class FiLMFusion(nn.Module):
    def __init__(self, d_model, stats_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(stats_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, x, stats):
        gamma, beta = torch.chunk(self.mlp(stats).unsqueeze(1), 2, dim=-1)
        return (1.0 + gamma) * x + beta


class PACLLM(nn.Module):
    def __init__(
        self,
        h,
        input_size,
        patch_len,
        stride,
        d_llm,
        feat_dim=5,
        d_model=32,
        d_ff=128,
        n_heads=8,
        dropout=0.1,
        llm_layers=2,
        embed_dim=3,
        lambda_main=0.9,
        use_amu_fusion=True,
        use_similarity_gate=False,
        feature_mode="raw",
        aux_source="patch",
        aux_model_type="transformer",
        prompt_prefix=None,
        dataset_description=None,
    ):
        super().__init__()
        self.pred_len = h
        self.d_model = d_model
        self.feat_dim = feat_dim
        self.takens_m = embed_dim
        self.lambda_main = lambda_main
        self.feature_mode = feature_mode
        self.aux_source = aux_source
        self.aux_model_type = aux_model_type

        self.takens_embed = LearnableTakensEmbedding(feat_dim, embed_dim)
        self.input_projection = nn.Linear(embed_dim, d_model)
        self.stats_extractor = ChaosFeatureExtractor()
        self.stats_dim = self._get_stats_dim(feat_dim)
        self.film_fusion = FiLMFusion(d_model, self.stats_dim) if self.stats_dim > 0 else None
        self.fusion_norm = nn.LayerNorm(d_model)

        self.time_model = PACLLMBackbone(
            h=h,
            input_size=input_size,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            d_llm=d_llm,
            llm_layers=llm_layers,
            enc_in=1,
            lambda_main=lambda_main,
            use_amu_fusion=use_amu_fusion,
            use_similarity_gate=use_similarity_gate,
            aux_source=aux_source,
            aux_model_type=aux_model_type,
            prompt_prefix=prompt_prefix,
            dataset_description=dataset_description,
        )

        aux_token_dim = d_llm if aux_source == "reprogrammed" else d_model
        self.time_model.channel_mixing = True
        self.time_model.channel_mixer = ChannelMixingLayer(n_vars=feat_dim, d_model=aux_token_dim)

    def _get_stats_dim(self, feat_dim):
        if self.feature_mode == "none":
            return 0
        if self.feature_mode == "raw":
            return self.stats_extractor.output_dim(feat_dim)
        if self.feature_mode == "takens":
            return self.stats_extractor.takens_output_dim(feat_dim)
        if self.feature_mode == "raw_takens":
            return self.stats_extractor.combined_output_dim(feat_dim)
        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, return_aux=False):
        batch_size, seq_len, feat_dim = x_enc.shape

        x_phase = self.takens_embed(x_enc)
        x_phase = x_phase.reshape(batch_size, seq_len, feat_dim, self.takens_m)
        x_projected = self.input_projection(x_phase)

        x_flat = x_projected.permute(0, 2, 1, 3).contiguous().reshape(
            batch_size * feat_dim, seq_len, self.d_model
        )
        if self.feature_mode == "none":
            x_fused = self.fusion_norm(x_flat)
        else:
            stats_parts = []
            if self.feature_mode in {"raw", "raw_takens"}:
                stats_parts.append(self.stats_extractor(x_enc))
            if self.feature_mode in {"takens", "raw_takens"}:
                stats_parts.append(self.stats_extractor.extract_takens_features(x_phase))
            film_stats = torch.cat(stats_parts, dim=-1)
            stats_expanded = film_stats.unsqueeze(1).repeat(1, feat_dim, 1).reshape(batch_size * feat_dim, -1)
            x_fused = self.film_fusion(x_flat, stats_expanded)
            x_fused = self.fusion_norm(x_fused)

        forecast_out = self.time_model.forecast(x_node_emb=x_fused, x_original=x_enc, return_aux=return_aux)
        if not return_aux:
            return forecast_out.reshape(batch_size, feat_dim, self.pred_len).permute(0, 2, 1)

        main_forecast = forecast_out["main"].reshape(batch_size, feat_dim, self.pred_len).permute(0, 2, 1)
        aux_forecast = forecast_out["aux"].reshape(batch_size, feat_dim, self.pred_len).permute(0, 2, 1)
        return {
            "main": main_forecast,
            "aux": aux_forecast,
            "alpha": forecast_out["alpha"],
            "sim_gate_mean": forecast_out["sim_gate_mean"],
        }




__all__ = [
    "ChaosFeatureExtractor",
    "FiLMFusion",
    "LearnableTakensEmbedding",
    "PACLLM",
]
