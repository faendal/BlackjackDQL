from src.Shoe import Shoe
from src.Hand import Hand


class BlackjackEnv:
    """A Blackjack environment where a player competes against a dealer with a shoe and better reward shaping."""

    def __init__(self) -> None:
        """Initialize the Blackjack environment."""
        try:
            self.deck: Shoe = Shoe(num_decks=4)
            self.player_hand: Hand = Hand()
            self.dealer_hand: Hand = Hand()
            self.done: bool = False
        except Exception as e:
            raise ValueError(f"Error initializing BlackjackEnv: {str(e)}") from e

    def reset(self) -> tuple[float, float, bool]:
        """Reset the environment and deal initial cards.

        Returns:
            tuple[float, float, bool]: The initial normalized state.
        """
        try:
            self.player_hand = Hand()
            self.dealer_hand = Hand()
            self.done = False

            for _ in range(2):
                self.player_hand.add_card(self.deck.deal_card())
                self.dealer_hand.add_card(self.deck.deal_card())

            return self._get_state()
        except Exception as e:
            raise ValueError(f"Error resetting BlackjackEnv: {str(e)}") from e

    def step(self, action: int) -> tuple[tuple[float, float, bool], float, bool, dict]:
        """Take an action in the environment.

        Args:
            action (int): 0 for stick, 1 for hit.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        try:
            if self.done:
                raise ValueError(
                    "Cannot call step() on a finished game. Please reset()."
                )

            if action not in [0, 1]:
                raise ValueError("Invalid action. Must be 0 (stick) or 1 (hit).")

            if action == 1:
                self.player_hand.add_card(self.deck.deal_card())
                player_value = self.player_hand.get_value()

                if player_value > 21:
                    self.done = True
                    return self._get_state(), -1.0, self.done, {"result": "player_bust"}
                elif 17 <= player_value <= 21:
                    return self._get_state(), 0.2, self.done, {}
                else:
                    return self._get_state(), -0.1, self.done, {}

            # Player sticks
            self.done = True
            player_value = self.player_hand.get_value()

            if 18 <= player_value <= 21:
                reward = 0.5
            else:
                reward = 0.0

            while self.dealer_hand.get_value() < 17:
                self.dealer_hand.add_card(self.deck.deal_card())

            dealer_value = self.dealer_hand.get_value()

            if dealer_value > 21 or player_value > dealer_value:
                reward += 1.0
                result = "player_win"
            elif player_value < dealer_value:
                reward -= 1.0
                result = "player_lose"
            else:
                result = "draw"

            return self._get_state(), reward, self.done, {"result": result}

        except Exception as e:
            raise ValueError(f"Error during step in BlackjackEnv: {str(e)}") from e

    def _get_state(self) -> tuple[float, float, bool]:
        """Get the normalized current state representation.

        Returns:
            tuple[float, float, bool]: (normalized_player_total, normalized_dealer_visible_card, usable_ace)
        """
        try:
            player_value = self.player_hand.get_value()
            dealer_visible_card = self.dealer_hand.cards[0].value
            usable_ace = self.player_hand.has_usable_ace()

            normalized_player = player_value / 32.0
            normalized_dealer = dealer_visible_card / 11.0

            return (normalized_player, normalized_dealer, usable_ace)
        except Exception as e:
            raise ValueError(f"Error getting state in BlackjackEnv: {str(e)}") from e

    def render(self) -> None:
        """Render the current game state."""
        try:
            print(
                f"\nPlayer's hand: {self.player_hand} (Value: {self.player_hand.get_value()})"
            )
            print(f"Dealer's visible card: {self.dealer_hand.cards[0]}")
        except Exception as e:
            raise ValueError(f"Error rendering BlackjackEnv: {str(e)}") from e
