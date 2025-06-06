"""Base model classes for LogoNav."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_trajectory_pred: int = 5,
        learn_angle: bool = True,
    ) -> None:
        """
        Base model for all navigation models
        Args:
            context_size (int): how many previous observations to used for
                context
            len_trajectory_pred (int): how many waypoints to predict in the
                future
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(BaseModel, self).__init__()
        self.context_size = context_size
        self.len_trajectory_pred = len_trajectory_pred
        self.learn_angle = learn_angle
        self.num_action_params = 4 if learn_angle else 2

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the tensor except for batch dimension"""
        return x.reshape(x.shape[0], -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        """
        Positional encoding for transformer models

        Args:
            d_model (int): dimension of the model
            max_seq_len (int): maximum sequence length
        """
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        """
        Add positional encoding to input

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, : x.size(1), :]
        return x


class MultiLayerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        seq_len=6,
        output_layers=[256, 128, 64, 32],
        nhead=8,
        num_layers=8,
        ff_dim_factor=4,
    ):
        """
        Multi-layer decoder with self-attention

        Args:
            embed_dim (int): embedding dimension
            seq_len (int): sequence length
            output_layers (list): list of output layer dimensions
            nhead (int): number of attention heads
            num_layers (int): number of transformer layers
            ff_dim_factor (int): factor to multiply embedding dim for
                feedforward network
        """
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(
            embed_dim, max_seq_len=seq_len
        )
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_decoder = nn.TransformerEncoder(
            self.sa_layer, num_layers=num_layers
        )
        self.output_layers = nn.ModuleList(
            [nn.Linear(seq_len * embed_dim, embed_dim)]
        )
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers) - 1):
            self.output_layers.append(
                nn.Linear(output_layers[i], output_layers[i + 1])
            )

    def forward(self, x):
        """
        Forward pass of the decoder

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Decoded output
        """
        if self.positional_encoding:
            x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # Currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
