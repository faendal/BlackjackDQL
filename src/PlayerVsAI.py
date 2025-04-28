import torch
import numpy as np
import streamlit as st
from src.Agent import Agent
from src.BlackjackEnv import BlackjackEnv


class PlayerVsAI:
    """Streamlit interface for a human player to play Blackjack against a trained AI agent."""

    def __init__(
        self,
        model_path: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        """
        Initialize the PlayerVsAI interface.

        Args:
            model_path (str): Path to the trained model file.
            device (torch.device, optional): Device for model inference. Defaults to CUDA if available.
        """
        try:
            self.device = device
            self.env = BlackjackEnv()
            self.agent = Agent(state_size=3, action_size=2, device=self.device)
            self.agent.qnetwork_local.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.agent.qnetwork_local.eval()
            self.reset_game()
        except Exception as e:
            raise ValueError(f"Error initializing PlayerVsAI: {str(e)}") from e

    def reset_game(self) -> None:
        """
        Reset the environment and session state for a new game.
        """
        try:
            initial_state = self.env.reset()
            st.session_state["state"] = np.array(initial_state, dtype=np.float32)
            st.session_state["done"] = False
            st.session_state["result"] = ""
            st.session_state["player_turn"] = True
        except Exception as e:
            raise ValueError(f"Error resetting game in PlayerVsAI: {str(e)}") from e

    def player_action(self, action: int) -> None:
        """
        Process player's action.

        Args:
            action (int): 0 = stick, 1 = hit
        """
        try:
            next_state, reward, done, info = self.env.step(action)
            st.session_state["state"] = np.array(next_state, dtype=np.float32)
            st.session_state["done"] = done

            if done:
                st.session_state["result"] = info.get("result", "")
                st.session_state["player_turn"] = False
        except Exception as e:
            raise ValueError(
                f"Error processing player action in PlayerVsAI: {str(e)}"
            ) from e

    def ai_turn(self) -> None:
        """
        Let the AI play its turn after the human finishes.
        """
        try:
            while not st.session_state["done"]:
                state = st.session_state["state"]
                action = self.agent.act(state, eps=0.0)  # Always exploit
                next_state, reward, done, info = self.env.step(action)
                st.session_state["state"] = np.array(next_state, dtype=np.float32)
                st.session_state["done"] = done
                if done:
                    st.session_state["result"] = info.get("result", "")
        except Exception as e:
            raise ValueError(f"Error during AI turn in PlayerVsAI: {str(e)}") from e

    def render_game(self) -> None:
        """
        Render the current game state in Streamlit.
        """
        try:
            st.title("Blackjack: Player vs AI")

            if st.button("New Game"):
                self.reset_game()

            self.env.render()

            if st.session_state["done"]:
                st.success(
                    f"Game Over! Result: {st.session_state['result'].replace('_', ' ').title()}"
                )
                self.ai_turn()
                return

            if st.session_state["player_turn"]:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Hit"):
                        self.player_action(1)
                with col2:
                    if st.button("Stick"):
                        self.player_action(0)

                if not st.session_state["player_turn"]:
                    st.info("AI's turn...")
                    self.ai_turn()
        except Exception as e:
            raise ValueError(f"Error rendering game in PlayerVsAI: {str(e)}") from e
