import torch
import torch.nn as nn
import torch.nn.functional as F


class FuturePredictionHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        encoding_size: int,
        hidden_size: int = 128,
        predict_deltas: bool = True,
        detach_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoding_size = encoding_size
        self.predict_deltas = predict_deltas
        self.detach_anchor = detach_anchor
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * encoding_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        anchor: torch.Tensor = None,
    ) -> torch.Tensor:
        future_delta = self.net(x)
        future_delta = future_delta.reshape(
            future_delta.shape[0],
            self.horizon,
            self.encoding_size,
        )
        if not self.predict_deltas:
            return future_delta

        future = torch.cumsum(future_delta, dim=1)
        if anchor is not None:
            assert (
                anchor.shape == future[:, 0].shape
            ), f"{anchor.shape} != {future[:, 0].shape}"
            if self.detach_anchor:
                anchor = anchor.detach()
            future = future + anchor.unsqueeze(1)
        return future


def compress_sequence_encoding(
    encoding: torch.Tensor,
    encoding_size: int,
) -> torch.Tensor:
    if encoding.shape[-1] == encoding_size:
        return encoding
    leading_shape = encoding.shape[:-1]
    encoding = encoding.reshape(-1, 1, encoding.shape[-1])
    encoding = F.adaptive_avg_pool1d(encoding, encoding_size)
    return encoding.reshape(*leading_shape, encoding_size)
