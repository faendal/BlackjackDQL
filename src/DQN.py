import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Deep Q-Network model for approximating Q-values."""

    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        """
        Initialize parameters and build the DQN model.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            seed (int): Random seed for reproducibility.
        """
        try:
            super(DQN, self).__init__()
            self.seed = torch.manual_seed(seed)
            self.fc1 = nn.Linear(state_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, action_size)
        except Exception as e:
            raise ValueError(f"Error initializing DQN: {str(e)}") from e

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Current state.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        try:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)
        except Exception as e:
            raise ValueError(f"Error during forward pass in DQN: {str(e)}") from e
