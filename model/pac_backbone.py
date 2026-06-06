import math

import torch
import torch.nn as nn
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import MAE

try:
    from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

    IS_TRANSFORMERS_INSTALLED = True
except ImportError:
    IS_TRANSFORMERS_INSTALLED = False


class ReplicationPad1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, input):
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        return torch.cat([input, replicate_padding], dim=-1)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout, in_channels=1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(in_channels * patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, channels, _ = x.shape
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, x.shape[-2], -1)
        x = self.value_embedding(x)
        return self.dropout(x), channels


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.3):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        batch_size, length, _ = target_embedding.shape
        source_len, _ = source_embedding.shape
        heads = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(batch_size, length, heads, -1)
        source_embedding = self.key_projection(source_embedding).view(source_len, heads, -1)
        value_embedding = self.value_projection(value_embedding).view(source_len, heads, -1)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        attn = self.dropout(torch.softmax(scores / math.sqrt(target_embedding.shape[-1]), dim=-1))
        out = torch.einsum("bhls,she->blhe", attn, value_embedding).reshape(batch_size, length, -1)
        return self.out_projection(out)


class ChannelMixingLayer(nn.Module):
    def __init__(self, n_vars, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x):
        batch_vars, seq_len, dim = x.shape
        if batch_vars % self.n_vars != 0:
            raise ValueError("Batch dimension mismatch in ChannelMixingLayer.")

        batch_size = batch_vars // self.n_vars
        vars_num = self.n_vars
        x_reshaped = x.reshape(batch_size, vars_num, seq_len, dim)
        x_reshaped = x_reshaped.permute(0, 2, 1, 3).reshape(batch_size * seq_len, vars_num, dim)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_mixed = self.norm(x_reshaped + self.dropout(attn_out))
        x_mixed = x_mixed * self.gate(x_mixed)
        return x_mixed.reshape(batch_size, seq_len, vars_num, dim).permute(0, 2, 1, 3).reshape(
            batch_vars, seq_len, dim
        )


class DynamicsAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, output_dim)
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        return self.head(self.mlp(x)) + self.residual_proj(x)


class AuxiliaryTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.encoder(x)
        return self.post_norm(x)


class AuxiliaryMLPEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.post_norm(x + self.net(x))


class AuxiliaryTemporalConvEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.post_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.pre_norm(x).transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        x = self.dropout(x).transpose(1, 2)
        return self.post_norm(residual + x)


class AuxiliaryInteractionEncoder(nn.Module):
    def __init__(self, d_model, aux_model_type="transformer", n_heads=4, dropout=0.1):
        super().__init__()
        if aux_model_type == "transformer":
            self.encoder = AuxiliaryTransformerEncoder(d_model, n_heads=n_heads, dropout=dropout)
        elif aux_model_type == "mlp":
            self.encoder = AuxiliaryMLPEncoder(d_model, dropout=dropout)
        elif aux_model_type == "tcn":
            self.encoder = AuxiliaryTemporalConvEncoder(d_model, dropout=dropout)
        else:
            raise ValueError(f"Unsupported aux_model_type: {aux_model_type}")

    def forward(self, x):
        return self.encoder(x)


class AMUGate(nn.Module):
    def __init__(self, feat_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, base_feat, aux_feat):
        gate_input = torch.cat([base_feat, aux_feat, torch.abs(base_feat - aux_feat)], dim=-1)
        return torch.sigmoid(self.net(gate_input))


class SimilarityGate(nn.Module):
    def __init__(self, feat_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = feat_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, base_feat, aux_feat):
        cosine = torch.cosine_similarity(base_feat, aux_feat, dim=-1, eps=1e-8).unsqueeze(-1)
        gate_input = torch.cat([base_feat, aux_feat, torch.abs(base_feat - aux_feat), cosine], dim=-1)
        gate = torch.sigmoid(self.net(gate_input))
        refined = gate * aux_feat
        return refined, gate


class PACLLMBackbone(BaseModel):
    SAMPLING_TYPE = "windows"
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False

    def __init__(
        self,
        h,
        input_size,
        patch_len: int = 16,
        stride: int = 8,
        d_ff: int = 128,
        top_k: int = 5,
        d_llm: int = 768,
        d_model: int = 32,
        n_heads: int = 8,
        enc_in: int = 3,
        dec_in: int = 3,
        llm=None,
        llm_config=None,
        llm_tokenizer=None,
        llm_layers=2,
        llm_output_attention: bool = True,
        llm_output_hidden_states: bool = True,
        prompt_prefix=None,
        dataset_description=None,
        dropout: float = 0.1,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        loss=MAE(),
        valid_loss=None,
        learning_rate: float = 0.0001,
        max_steps: int = 5,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size=None,
        windows_batch_size: int = 1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled: bool = False,
        step_size: int = 1,
        num_lr_decays: int = 0,
        early_stop_patience_steps: int = -1,
        scaler_type: str = "identity",
        num_workers_loader: int = 0,
        drop_last_loader: bool = False,
        random_seed: int = 2021,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        hidden_size=128,
        lambda_main: float = 0.9,
        use_amu_fusion: bool = True,
        use_similarity_gate: bool = False,
        aux_source: str = "patch",
        aux_model_type: str = "transformer",
    ):
        super().__init__(
            h=h,
            input_size=input_size,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            futr_exog_list=futr_exog_list,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            num_workers_loader=num_workers_loader,
            drop_last_loader=drop_last_loader,
            random_seed=random_seed,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )


        if aux_source not in {"x_fused", "patch", "reprogrammed"}:
            raise ValueError(f"Unsupported aux_source: {aux_source}")

        self.patch_len = patch_len
        self.stride = stride
        self.d_ff = d_ff
        self.top_k = top_k
        self.d_llm = d_llm
        self.d_model = d_model
        self.dropout = dropout
        self.n_heads = n_heads
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.h = h
        self.input_size = input_size
        self.prompt_prefix = prompt_prefix
        self.dataset_description = dataset_description or "Chaotic multivariate trajectory forecasting."
        self.lambda_main = lambda_main
        self.use_amu_fusion = use_amu_fusion
        self.use_similarity_gate = use_similarity_gate
        self.aux_source = aux_source
        self.aux_model_type = aux_model_type
        self.llm_layers=llm_layers

        config = GPT2Config.from_pretrained("openai-community/gpt2")
        config.num_hidden_layers =llm_layers
        self.llm_model = GPT2Model.from_pretrained("openai-community/gpt2", config=config)
        self.tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout, in_channels=d_model)
        self.aux_patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout, in_channels=d_model)
        self.mapping_layer = nn.Linear(self.llm_model.get_input_embeddings().weight.shape[0], 1000)
        self.reprogramming_layer = ReprogrammingLayer(d_model, n_heads, d_ff, d_llm)

        self.patch_nums = int((input_size - patch_len) / stride + 2)
        self.head_nf = d_ff * self.patch_nums

        # external injection from wrapper
        self.channel_mixing = False
        self.channel_mixer = None

        aux_token_dim = d_llm if aux_source == "reprogrammed" else d_model
        self.aux_interaction_encoder = AuxiliaryInteractionEncoder(
            aux_token_dim,
            aux_model_type=aux_model_type,
            n_heads=4,
            dropout=dropout,
        )
        self.aux_token_projector = nn.Linear(aux_token_dim, d_ff)
        self.aux_delta_head = nn.Sequential(
            nn.Linear(self.head_nf, self.head_nf),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_nf, self.head_nf),
        )
        self.similarity_gate = SimilarityGate(self.head_nf, hidden_dim=self.head_nf, dropout=dropout)
        self.uncertainty_gate = AMUGate(self.head_nf, dropout=dropout)
        self.main_head = DynamicsAdapter(
            input_dim=self.head_nf,
            output_dim=h * enc_in,
            hidden_dim=128,
            dropout=dropout,
        )
        self.aux_head = DynamicsAdapter(
            input_dim=self.head_nf,
            output_dim=h * enc_in,
            hidden_dim=128,
            dropout=dropout,
        )
        self.output_projection_latent = self.main_head
        self.last_aux_output = None
        self.last_alpha = None
        self.last_sim_gate = None

    def _build_prompts(self, obs_data):
        min_vals, max_vals = obs_data.min(dim=1)[0], obs_data.max(dim=1)[0]
        trends = obs_data[:, -1] - obs_data[:, 0]
        prompts = []
        for idx in range(obs_data.shape[0]):
            trend_text = "upward" if trends[idx].item() > 0 else "downward"
            prompt = (
                f"<|start_prompt|>{self.prompt_prefix or ''}"
                f"Dataset Description: {self.dataset_description}\n"
                "Method: time-delay embedding is applied.\n"
                f"Task description: Forecast the next {self.h} steps given the previous {self.input_size} observations.\n"
                "Input statistics: "
                f"minimum value {min_vals[idx].item():.4f}, "
                f"maximum value {max_vals[idx].item():.4f}, "
                f"trend {trend_text}."
                "<|end_prompt|>"
            )
            prompts.append(prompt)
        return prompts

    def _prepare_aux_tokens(self, x_node_emb, patch_tokens, reprogrammed_tokens):
        if not (self.channel_mixing and self.channel_mixer is not None):
            return None

        if self.aux_source == "x_fused":
            aux_input = x_node_emb.permute(0, 2, 1).contiguous()
            aux_tokens, _ = self.aux_patch_embedding(aux_input)
        elif self.aux_source == "patch":
            aux_tokens = patch_tokens
        else:
            aux_tokens = reprogrammed_tokens

        mixed_out = self.channel_mixer(aux_tokens)
        encoded_tokens = self.aux_interaction_encoder(mixed_out)
        projected_tokens = self.aux_token_projector(encoded_tokens)
        return projected_tokens.reshape(projected_tokens.shape[0], -1)

    def forecast(self, x_node_emb, x_original=None, return_aux=False):
        if x_original is not None:
            batch_size, seq_len, vars_num = x_original.shape
            obs_data = x_original.permute(0, 2, 1).contiguous().reshape(batch_size * vars_num, seq_len)
        else:
            obs_data = x_node_emb.mean(dim=-1)

        batch_total = obs_data.shape[0]
        prompt_ids = self.tokenizer(
            self._build_prompts(obs_data),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(x_node_emb.device))

        x_enc = x_node_emb.permute(0, 2, 1).contiguous()
        patch_tokens, _ = self.patch_embedding(x_enc)

        vocab_embeddings = self.llm_model.get_input_embeddings().weight
        source_emb = self.mapping_layer(vocab_embeddings.permute(1, 0)).permute(1, 0)
        reprogrammed_tokens = self.reprogramming_layer(patch_tokens, source_emb, source_emb)

        aux_feat = self._prepare_aux_tokens(x_node_emb, patch_tokens, reprogrammed_tokens)

        llm_in = torch.cat([prompt_embeddings, reprogrammed_tokens], dim=1)
        dec_out = self.llm_model(inputs_embeds=llm_in).last_hidden_state
        dec_out = dec_out[:, -reprogrammed_tokens.shape[1] :, : self.d_ff]
        base_feat = dec_out.reshape(batch_total, -1)

        alpha = None
        aux_output = None
        if self.use_amu_fusion and aux_feat is not None:
            aux_refined = aux_feat
            sim_gate = None
            if self.use_similarity_gate:
                aux_refined, sim_gate = self.similarity_gate(base_feat, aux_feat)
            delta_feat = self.aux_delta_head(aux_refined)
            alpha = self.uncertainty_gate(base_feat, aux_refined)
            fused_feat = base_feat + alpha * delta_feat
            aux_output = self.aux_head(aux_refined).reshape(batch_total, self.h, self.enc_in)
            self.last_sim_gate = sim_gate
        else:
            fused_feat = base_feat
            self.last_sim_gate = None

        main_output = self.output_projection_latent(fused_feat).reshape(batch_total, self.h, self.enc_in)
        self.last_aux_output = aux_output
        self.last_alpha = alpha

        if return_aux:
            if alpha is None:
                alpha = main_output.new_ones(main_output.shape[0], self.head_nf)
            if aux_output is None:
                aux_output = main_output
            sim_gate_mean = (
                self.last_sim_gate.mean(dim=-1, keepdim=True)
                if self.last_sim_gate is not None
                else main_output.new_full((main_output.shape[0], 1), float("nan"))
            )
            return {"main": main_output, "aux": aux_output, "alpha": alpha, "sim_gate_mean": sim_gate_mean}
        return main_output




__all__ = [
    "AMUGate",
    "AuxiliaryInteractionEncoder",
    "ChannelMixingLayer",
    "DynamicsAdapter",
    "PACLLMBackbone",
    "PatchEmbedding",
    "ReprogrammingLayer",
    "SimilarityGate",
]
