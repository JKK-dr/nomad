import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
from vint_train.models.gnm.modified_mobilenetv2 import MobileNetEncoder
from vint_train.models.base_model import BaseModel
from vint_train.models.future_prediction import (
    FuturePredictionHead,
    compress_sequence_encoding,
)


class GNM(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
        future_prediction: Optional[Dict] = None,
    ) -> None:
        """
        GNM main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(GNM, self).__init__(context_size, len_traj_pred, learn_angle)
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size
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
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )
        stacked_mobilenet = MobileNetEncoder(
            num_images=2 + self.context_size
        )  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_size = goal_encoding_size
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
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

    def _encode_observation_stack(self, obs_img: torch.Tensor) -> torch.Tensor:
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        return self.compress_observation(obs_encoding)

    def encode_future_images(self, future_img: torch.Tensor) -> torch.Tensor:
        future_img = torch.split(future_img, 3, dim=1)
        future_img = torch.cat(future_img, dim=0)
        future_img = future_img.repeat(1, self.context_size + 1, 1, 1)
        future_encoding = self._encode_observation_stack(future_img)
        future_encoding = future_encoding.reshape(
            (self.future_horizon, -1, self.obs_encoding_size)
        )
        future_encoding = torch.transpose(future_encoding, 0, 1)
        future_encoding = compress_sequence_encoding(
            future_encoding,
            self.future_encoding_size,
        )
        return future_encoding.detach()

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_encoding = self._encode_observation_stack(obs_img)
        if self.future_prediction_enabled:
            future_anchor = compress_sequence_encoding(
                obs_encoding,
                self.future_encoding_size,
            )

        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        z = torch.cat([obs_encoding, goal_encoding], dim=1)
        z = self.linear_layers(z)
        dist_pred = self.dist_predictor(z)
        action_pred = self.action_predictor(z)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        if self.future_prediction_enabled:
            future_encoding_pred = self.future_predictor(
                z,
                anchor=future_anchor,
            )
            return dist_pred, action_pred, future_encoding_pred
        return dist_pred, action_pred
