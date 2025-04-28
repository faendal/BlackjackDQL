import torch
import random
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples for training a DQN agent."""

    def __init__(
        self, action_size: int, buffer_size: int, batch_size: int, device: torch.device
    ) -> None:
        """
        Initialize a ReplayBuffer object.

        Args:
            action_size (int): Dimension of action space.
            buffer_size (int): Maximum size of buffer.
            batch_size (int): Size of each training batch.
            device (torch.device): Device to move tensors to (CPU or GPU).
        """
        try:
            self.action_size: int = action_size
            self.memory: deque = deque(maxlen=buffer_size)
            self.batch_size: int = batch_size
            self.device: torch.device = device
            self.experience = namedtuple(
                "Experience",
                field_names=["state", "action", "reward", "next_state", "done"],
            )
        except Exception as e:
            raise ValueError(f"Error initializing ReplayBuffer: {str(e)}") from e

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a new experience to memory.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode is finished.
        """
        try:
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
        except Exception as e:
            raise ValueError(
                f"Error adding experience to ReplayBuffer: {str(e)}"
            ) from e

    def sample(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.

        Returns:
            Tuple containing (states, actions, rewards, next_states, dones) as torch.Tensors
        """
        try:
            experiences = random.sample(self.memory, k=self.batch_size)

            states = (
                torch.from_numpy(np.vstack([e.state for e in experiences]))
                .float()
                .to(self.device)
            )
            actions = (
                torch.from_numpy(np.vstack([e.action for e in experiences]))
                .long()
                .to(self.device)
            )
            rewards = (
                torch.from_numpy(np.vstack([e.reward for e in experiences]))
                .float()
                .to(self.device)
            )
            next_states = (
                torch.from_numpy(np.vstack([e.next_state for e in experiences]))
                .float()
                .to(self.device)
            )
            dones = (
                torch.from_numpy(
                    np.vstack([e.done for e in experiences]).astype(np.uint8)
                )
                .float()
                .to(self.device)
            )

            return (states, actions, rewards, next_states, dones)
        except Exception as e:
            raise ValueError(f"Error sampling from ReplayBuffer: {str(e)}") from e

    def __len__(self) -> int:
        """
        Return the current size of internal memory.

        Returns:
            int: Number of experiences currently stored.
        """
        return len(self.memory)
