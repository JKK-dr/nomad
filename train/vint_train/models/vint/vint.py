import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder
from vint_train.models.future_prediction import (
    FuturePredictionHead,
    compress_sequence_encoding,
)

class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        future_prediction: Optional[Dict] = None,
    ) -> None:
        """
        ViNT class: uses a Transformer-based architecture to encode (current and past) visual observations 
        and goals using an EfficientNet CNN, and predicts temporal distance and normalized actions 
        in an embodiment-agnostic manner
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoder (str): name of the EfficientNet architecture to use for encoding observations (ex. "efficientnet-b0")
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(ViNT, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        future_prediction = future_prediction or {}
        self.future_prediction_enabled = future_prediction.get("enabled", False)
        self.future_horizon = int(future_prediction.get("horizon", 1))
        self.future_predict_deltas = future_prediction.get("predict_deltas", True)
        self.future_detach_anchor = future_prediction.get("detach_anchor", True)
        future_encoding_size = future_prediction.get("encoding_size", None)
        self.future_encoding_size = (
            self.obs_encoding_size
            if future_encoding_size is None
            else int(future_encoding_size)
        )
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.num_obs_features = self.obs_encoder._fc.in_features
            if self.late_fusion:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.num_goal_features = self.goal_encoder._fc.in_features
        else:
            raise NotImplementedError
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )
        if self.future_prediction_enabled:
            self.future_predictor = FuturePredictionHead(
                input_size=32,
                horizon=self.future_horizon,
                encoding_size=self.future_encoding_size,
                predict_deltas=self.future_predict_deltas,
                detach_anchor=self.future_detach_anchor,
            )

    def _encode_single_images(self, img: torch.Tensor) -> torch.Tensor:
        return self.compress_obs_enc(
            self._encode_with_efficientnet(self.obs_encoder, img)
        )

    def _encode_with_efficientnet(
        self,
        encoder: EfficientNet,
        img: torch.Tensor,
    ) -> torch.Tensor:
        encoding = encoder.extract_features(img)
        encoding = encoder._avg_pooling(encoding)
        if encoder._global_params.include_top:
            encoding = encoding.flatten(start_dim=1)
            encoding = encoder._dropout(encoding)
        return encoding

    def _encode_image_sequence(
        self,
        img: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        batch_size = img.shape[0]
        img_sequence = torch.split(img, 3, dim=1)
        assert len(img_sequence) == seq_len, f"{len(img_sequence)} != {seq_len}"
        img = torch.cat(img_sequence, dim=0)
        encoding = self._encode_single_images(img)
        encoding = encoding.reshape((seq_len, batch_size, self.obs_encoding_size))
        return torch.transpose(encoding, 0, 1)

    def _encode_goal(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        if self.late_fusion:
            goal_input = goal_img
        else:
            current_img = obs_img[:, 3 * self.context_size:, :, :]
            goal_input = torch.cat([current_img, goal_img], dim=1)
        goal_encoding = self._encode_with_efficientnet(self.goal_encoder, goal_input)
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        assert goal_encoding.shape[2] == self.goal_encoding_size
        return goal_encoding

    def _predict_actions(self, final_repr: torch.Tensor) -> torch.Tensor:
        action_pred = self.action_predictor(final_repr)
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )
        return action_pred

    def _predict_future_encoding(
        self,
        final_repr: torch.Tensor,
        obs_encoding: torch.Tensor,
    ) -> torch.Tensor:
        future_anchor = compress_sequence_encoding(
            obs_encoding[:, -1],
            self.future_encoding_size,
        )
        return self.future_predictor(final_repr, anchor=future_anchor)

    def encode_future_images(self, future_img: torch.Tensor) -> torch.Tensor:
        future_encoding = self._encode_image_sequence(
            future_img,
            self.future_horizon,
        )
        future_encoding = compress_sequence_encoding(
            future_encoding,
            self.future_encoding_size,
        )
        return future_encoding.detach()

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        goal_encoding = self._encode_goal(obs_img, goal_img)
        obs_encoding = self._encode_image_sequence(obs_img, self.context_size + 1)
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_repr = self.decoder(tokens)

        dist_pred = self.dist_predictor(final_repr)
        action_pred = self._predict_actions(final_repr)
        if self.future_prediction_enabled:
            future_encoding_pred = self._predict_future_encoding(
                final_repr,
                obs_encoding,
            )
            return dist_pred, action_pred, future_encoding_pred
        return dist_pred, action_pred
