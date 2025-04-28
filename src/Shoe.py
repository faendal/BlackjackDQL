import random
from src.Card import Card
from src.Deck import Deck


class Shoe:
    """A class representing a shoe containing multiple decks used in Blackjack."""

    def __init__(self, num_decks: int = 4) -> None:
        """
        Initialize the shoe by creating and shuffling multiple decks.

        Args:
            num_decks (int): Number of standard decks to include in the shoe.
        """
        try:
            self.num_decks = num_decks
            self.cards: list[Card] = self.build_shoe()
            self.shuffle()
        except Exception as e:
            raise ValueError(f"Error initializing Shoe: {str(e)}") from e

    def build_shoe(self) -> list[Card]:
        """
        Build a shoe consisting of multiple standard Decks.

        Returns:
            list[Card]: A list of Card instances.
        """
        shoe_cards = []
        for _ in range(self.num_decks):
            deck = Deck()
            shoe_cards.extend(deck.cards)
        return shoe_cards

    def shuffle(self) -> None:
        """
        Shuffle the shoe randomly.
        """
        try:
            random.shuffle(self.cards)
        except Exception as e:
            raise ValueError(f"Error shuffling Shoe: {str(e)}") from e

    def deal_card(self) -> Card:
        """
        Deal (pop) a card from the shoe. Automatically reshuffle when empty.

        Returns:
            Card: The dealt Card object.
        """
        try:
            if not self.cards:
                self.cards = self.build_shoe()
                self.shuffle()
            return self.cards.pop()
        except Exception as e:
            raise ValueError(f"Error dealing card from Shoe: {str(e)}") from e

    def __len__(self) -> int:
        """
        Return the number of cards remaining in the shoe.

        Returns:
            int: Number of cards left.
        """
        return len(self.cards)
