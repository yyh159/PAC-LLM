import torch
import torch.nn as nn


class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_size, layer_size)]
            + [nn.Linear(layer_size, layer_size) for _ in range(layers - 1)]
        )
        self.basis_parameters = nn.Linear(layer_size, theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, input_mask):
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast


class GenericBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta):
        return theta[:, : self.backcast_size], theta[:, -self.forecast_size :]


class SpatioTemporalNBeats(nn.Module):
    def __init__(
        self,
        h,
        input_size,
        feat_dim=3,
        hidden_units=128,
        nb_blocks_per_stack=3,
        num_stacks=2,
        theta_dim=128,
        share_weights_in_stack=False,
        backcast_length=None,
    ):
        super().__init__()
        self.pred_len = h
        self.input_size = input_size
        self.feat_dim = feat_dim
        self.backcast_length = input_size if backcast_length is None else backcast_length
        self.hidden_units = hidden_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.num_stacks = num_stacks
        self.theta_dim = theta_dim
        self.share_weights_in_stack = share_weights_in_stack

        self.backbone = self._build_official_backbone()

    def _build_official_backbone(self):
        blocks = []
        theta_size = self.backcast_length + self.pred_len

        for _ in range(self.num_stacks):
            if self.share_weights_in_stack:
                shared_block = NBeatsBlock(
                    input_size=self.backcast_length,
                    theta_size=theta_size,
                    basis_function=GenericBasis(self.backcast_length, self.pred_len),
                    layers=4,
                    layer_size=self.hidden_units,
                )
                blocks.extend(shared_block for _ in range(self.nb_blocks_per_stack))
            else:
                for _ in range(self.nb_blocks_per_stack):
                    blocks.append(
                        NBeatsBlock(
                            input_size=self.backcast_length,
                            theta_size=theta_size,
                            basis_function=GenericBasis(self.backcast_length, self.pred_len),
                            layers=4,
                            layer_size=self.hidden_units,
                        )
                    )

        return NBeats(nn.ModuleList(blocks))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = x_enc[:, -self.backcast_length :, :]
        batch_size, _, feat_dim = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * feat_dim, self.backcast_length)
        input_mask = torch.ones_like(x)
        forecast = self.backbone(x, input_mask)
        forecast = forecast.reshape(batch_size, feat_dim, self.pred_len).permute(0, 2, 1)
        return forecast
