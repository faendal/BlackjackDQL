import os
import torch
import numpy as np
import seaborn as sns
from src.Agent import Agent
import matplotlib.pyplot as plt
from src.BlackjackEnv import BlackjackEnv

sns.set_style("darkgrid")
sns.set_palette("Set1")


class Trainer:
    """Trainer class to handle the training of a Double DQN agent on the Blackjack environment."""

    def __init__(
        self,
        n_episodes: int = 50000,
        max_t: int = 100,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        model_save_path: str = "models/",
        load_model_path: str | None = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        seed: int = 0,
    ) -> None:
        """Initialize the Trainer."""
        try:
            self.device = device
            self.env = BlackjackEnv()
            self.agent = Agent(
                state_size=3, action_size=2, device=self.device, seed=seed
            )
            self.n_episodes = n_episodes
            self.max_t = max_t
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay = eps_decay
            self.model_save_path = model_save_path
            self.scores: list[float] = []
            self.best_avg_score = -float("inf")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.agent.optimizer, mode="max", factor=0.5, patience=300
            )

            if load_model_path:
                self.load_model(load_model_path)

        except Exception as e:
            raise ValueError(f"Error initializing Trainer: {str(e)}") from e

    def train(self) -> None:
        """Train the agent in the Blackjack environment."""
        try:
            eps = self.eps_start
            for i_episode in range(1, self.n_episodes + 1):
                state = np.array(self.env.reset(), dtype=np.float32)
                score = 0
                for t in range(self.max_t):
                    action = self.agent.act(state, eps)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.array(next_state, dtype=np.float32)
                    self.agent.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    if done:
                        break
                self.scores.append(score)
                eps = max(self.eps_end, self.eps_decay * eps)

                if i_episode % 100 == 0:
                    moving_avg = np.mean(self.scores[-100:])
                    print(
                        f"Episode {i_episode}/{self.n_episodes}, Moving Average Score: {moving_avg:.2f}"
                    )

                    self.scheduler.step(moving_avg)

                    if moving_avg > self.best_avg_score:
                        self.best_avg_score = moving_avg
                        self.save_model("dqn_blackjack_best.pth")

        except Exception as e:
            raise ValueError(f"Error during training in Trainer: {str(e)}") from e

    def save_model(self, filename: str = "dqn_blackjack.pth") -> None:
        """Save the local model to file."""
        try:
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)
            torch.save(
                self.agent.qnetwork_local.state_dict(),
                os.path.join(self.model_save_path, filename),
            )
        except Exception as e:
            raise ValueError(f"Error saving model in Trainer: {str(e)}") from e

    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model from file."""
        try:
            self.agent.qnetwork_local.load_state_dict(
                torch.load(filepath, map_location=self.device)
            )
            self.agent.qnetwork_target.load_state_dict(
                torch.load(filepath, map_location=self.device)
            )
            print(f"Loaded model from {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading model in Trainer: {str(e)}") from e

    def plot_scores(self) -> None:
        """Plot the scores over episodes."""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(
                np.arange(len(self.scores)), self.scores, label="Score per Episode"
            )
            moving_avg = np.convolve(self.scores, np.ones(100) / 100, mode="valid")
            plt.plot(
                np.arange(len(moving_avg)),
                moving_avg,
                label="Moving Average (100 episodes)",
                linestyle="--",
            )
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.title("Training Progress")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            raise ValueError(f"Error plotting scores in Trainer: {str(e)}") from e
