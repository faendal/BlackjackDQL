import random
from src.Card import Card


class Deck:
    """A class representing a standard deck of 52 playing cards"""

    def __init__(self) -> None:
        """
        Initialize the deck by creating and shuffling 52 cards.
        """
        try:
            self.cards: list[Card] = self.build_deck()
            self.shuffle()
        except Exception as e:
            raise ValueError(f"Error initializing Deck: {str(e)}") from e

    def build_deck(self) -> list[Card]:
        """
        Build a complete deck of 52 cards.

        Returns:
            list[Card]: A list of Card instances representing the full deck.
        """
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return [Card(suit, rank) for suit in suits for rank in ranks]

    def shuffle(self) -> None:
        """
        Shuffle the deck randomly.
        """
        try:
            random.shuffle(self.cards)
        except Exception as e:
            raise ValueError(f"Error shuffling Deck: {str(e)}") from e

    def deal_card(self) -> Card:
        """
        Deal (pop) a card from the deck.

        Returns:
            Card: The dealt Card object.

        Raises:
            ValueError: If the deck is empty.
        """
        try:
            if not self.cards:
                raise ValueError("No more cards in the deck to deal.")
            return self.cards.pop()
        except Exception as e:
            raise ValueError(f"Error dealing card: {str(e)}") from e

    def __len__(self) -> int:
        """
        Return the number of cards remaining in the deck.

        Returns:
            int: Number of cards left.
        """
        return len(self.cards)
