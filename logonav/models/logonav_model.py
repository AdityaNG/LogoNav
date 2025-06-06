"""LogoNav model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from logonav.models.base_model import BaseModel, MultiLayerDecoder


class LogoNavModel(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: int = 5,
        learn_angle: bool = True,
        obs_encoder: str = "efficientnet-b0",
        obs_encoding_size: int = 512,
        late_fusion: bool = False,
        mha_num_attention_heads: int = 2,
        mha_num_attention_layers: int = 2,
        mha_ff_dim_factor: int = 4,
    ) -> None:
        """
        LogoNav model: uses a Transformer-based architecture to encode
            visual observations
        and navigates to a goal specified by GPS-like coordinates.

        Args:
            context_size (int): how many previous observations to used for
                context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use
                for encoding observations
            obs_encoding_size (int): size of the encoding of the
                observation images
            late_fusion (bool): whether to use late fusion for goal and
                observation
            mha_num_attention_heads (int): number of attention heads in
                multi-head attention
            mha_num_attention_layers (int): number of transformer layers
            mha_ff_dim_factor (int): factor to multiply embedding dim for
                feedforward network
        """
        super(LogoNavModel, self).__init__(
            context_size, len_traj_pred, learn_angle
        )
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.late_fusion = late_fusion

        # Initialize the observation encoder
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(
                obs_encoder, in_channels=3
            )
            self.num_obs_features = self.obs_encoder._fc.in_features
        else:
            raise NotImplementedError(
                f"Observation encoder {obs_encoder} not supported"
            )

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(
                self.num_obs_features, self.obs_encoding_size
            )
        else:
            self.compress_obs_enc = nn.Identity()  # type: ignore

        self.compress_goal_enc = nn.Identity()

        # Initialize transformer decoder
        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size + 2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        # Initialize action predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

        # Constants for velocity scaling
        self.max_linvel = 0.5
        self.max_angvel = 1.0

        # Goal encoder for position input
        self.local_goal = nn.Sequential(
            nn.Linear(4, self.goal_encoding_size),
        )

    def forward(
        self, obs_img: torch.Tensor, goal_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the LogoNav model

        Args:
            obs_img (torch.Tensor): Batch of observation images with shape
                [B, 3*context_size, H, W]
            goal_pose (torch.Tensor): Batch of goal poses with shape [B, 4]
                where each element is [dx, dy, cos(theta), sin(theta)]

        Returns:
            torch.Tensor: Predicted waypoints with shape
                [B, len_trajectory_pred, num_action_params]
        """
        # Encode the goal pose
        goal_encoding = self.local_goal(goal_pose).unsqueeze(1)

        # Split the observation into context based on the context size
        obs_img_chunks = torch.split(obs_img, 3, dim=1)
        obs_img_concat = torch.concat(obs_img_chunks, dim=0)

        # Get the observation encoding
        obs_encoding = self.obs_encoder.extract_features(obs_img_concat)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)

        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)

        # Compress observation encoding if needed
        obs_encoding = self.compress_obs_enc(obs_encoding)

        # Reshape to [batch_size, context_size+1, obs_encoding_size]
        obs_encoding = obs_encoding.reshape(
            (self.context_size + 1, -1, self.obs_encoding_size)
        )
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        # Concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)

        # Process through transformer decoder
        final_repr = self.decoder(tokens)

        # Predict actions
        action_pred = self.action_predictor(final_repr)

        # Reshape to match expected output format
        action_pred = action_pred.reshape(
            (
                action_pred.shape[0],
                self.len_trajectory_pred,
                self.num_action_params,
            )
        )

        # Convert position deltas into waypoints with cumulative sum
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)

        # Normalize the angle prediction
        action_pred[:, :, 2:] = F.normalize(
            action_pred[:, :, 2:].clone(), dim=-1
        )

        return action_pred
