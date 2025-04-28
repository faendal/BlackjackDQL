import torch
import random
import numpy as np
from src.DQN import DQN
import torch.nn.functional as F
from src.ReplayBuffer import ReplayBuffer


class Agent:
    """Interacts with and learns from the Blackjack environment using Deep Q-Learning."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: torch.device,
        seed: int = 0,
        buffer_size: int = int(1e5),
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr: float = 1e-3,
        update_every: int = 4,
    ) -> None:
        """
        Initialize an Agent object.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of action space.
            device (torch.device): Device to use (CPU or CUDA).
            seed (int, optional): Random seed. Defaults to 0.
            buffer_size (int, optional): Replay buffer size. Defaults to 1e5.
            batch_size (int, optional): Minibatch size. Defaults to 64.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Soft update interpolation parameter. Defaults to 1e-3.
            lr (float, optional): Learning rate. Defaults to 5e-4.
            update_every (int, optional): Frequency of learning steps. Defaults to 4.
        """
        try:
            self.state_size = state_size
            self.action_size = action_size
            self.device = device
            self.seed = random.seed(seed)

            self.qnetwork_local = DQN(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = DQN(state_size, action_size, seed).to(self.device)
            self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)

            self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device)
            self.batch_size = batch_size

            self.gamma = gamma
            self.tau = tau
            self.update_every = update_every

            self.t_step = 0

        except Exception as e:
            raise ValueError(f"Error initializing Agent: {str(e)}") from e

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Save experience in replay memory and learn every few time steps.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode ended.
        """
        try:
            self.memory.add(state, action, reward, next_state, done)
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0 and len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
        except Exception as e:
            raise ValueError(f"Error in step() of Agent: {str(e)}") from e

    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        """
        Returns actions for given state as per current policy.

        Args:
            state (np.ndarray): Current state.
            eps (float, optional): Epsilon for exploration. Defaults to 0.0.

        Returns:
            int: Selected action.
        """
        try:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))
        except Exception as e:
            raise ValueError(f"Error in act() of Agent: {str(e)}") from e

    def learn(self, experiences: tuple, gamma: float) -> None:
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences (tuple): Tuple of (s, a, r, s', done) tuples.
            gamma (float): Discount factor.
        """
        try:
            states, actions, rewards, next_states, dones = experiences

            Q_targets_next = (
                self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            )
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            Q_expected = self.qnetwork_local(states).gather(1, actions)

            loss = F.mse_loss(Q_expected, Q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        except Exception as e:
            raise ValueError(f"Error in learn() of Agent: {str(e)}") from e

    def soft_update(
        self, local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float
    ) -> None:
        """
        Soft update model parameters.

        Args:
            local_model (torch.nn.Module): Weights will be copied from this model.
            target_model (torch.nn.Module): Weights will be copied to this model.
            tau (float): Interpolation parameter.
        """
        try:
            for target_param, local_param in zip(
                target_model.parameters(), local_model.parameters()
            ):
                target_param.data.copy_(
                    tau * local_param.data + (1.0 - tau) * target_param.data
                )
        except Exception as e:
            raise ValueError(f"Error in soft_update() of Agent: {str(e)}") from e
