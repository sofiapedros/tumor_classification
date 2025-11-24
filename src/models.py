import torch
import torch.nn as nn


class BreastMLPClassifier(nn.Module):
    """
    Simple, efficient MLP for tabular data
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dims:tuple=(128, 64),
        dropout: float=0.2,
    ):
        """
        Args:
        - input_dim (int)
        - num_classes (int)
        - hidden_dims (tuple)
        - dropout (float)
        """
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h

        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
