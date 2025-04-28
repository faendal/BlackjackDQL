from src.Card import Card


class Hand:
    """A class representing a player's hand in Blackjack"""

    def __init__(self) -> None:
        """
        Initialize an empty hand.
        """
        try:
            self.cards: list[Card] = []
        except Exception as e:
            raise ValueError(f"Error initializing Hand: {str(e)}") from e

    def add_card(self, card: Card) -> None:
        """
        Add a card to the hand.

        Args:
            card (Card): The card to add.
        """
        try:
            if not isinstance(card, Card):
                raise TypeError("Only Card instances can be added to the hand.")
            self.cards.append(card)
        except Exception as e:
            raise ValueError(f"Error adding card to Hand: {str(e)}") from e

    def get_value(self) -> int:
        """
        Calculate the total value of the hand considering Aces.

        Returns:
            int: The total value of the hand.
        """
        try:
            value = sum(card.value for card in self.cards)
            num_aces = sum(1 for card in self.cards if card.rank == "A")

            while value > 21 and num_aces:
                value -= 10
                num_aces -= 1

            return value
        except Exception as e:
            raise ValueError(f"Error calculating Hand value: {str(e)}") from e

    def has_usable_ace(self) -> bool:
        """
        Check if the hand has an Ace counted as 11 without busting.

        Returns:
            bool: True if a usable Ace exists, False otherwise.
        """
        try:
            value = sum(card.value for card in self.cards)
            for card in self.cards:
                if card.rank == "A" and value <= 21:
                    return True
            return False
        except Exception as e:
            raise ValueError(f"Error checking usable Ace: {str(e)}") from e

    def __repr__(self) -> str:
        """
        Return a string representation of the hand.

        Returns:
            str: List of cards in the hand.
        """
        return f"Hand({', '.join(str(card) for card in self.cards)})"
