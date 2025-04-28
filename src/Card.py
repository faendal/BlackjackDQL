class Card:
    """A class representing a playing card"""

    def __init__(self, suit: str, rank: str) -> None:
        """
        Initialize a card with a suit and rank.
        Args:
            suit (str): The suit of the card ("Hearts", "Diamonds", "Clubs", "Spades").
            rank (str): The rank of the card ("2", "3", ..., "10", "J", "Q", "K", "A").
        """

        try:
            valid_suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
            valid_ranks = [
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "J",
                "Q",
                "K",
                "A",
            ]
            if suit not in valid_suits:
                raise ValueError(
                    f"Invalid suit '{suit}'. Must be one of {valid_suits}."
                )
            if rank not in valid_ranks:
                raise ValueError(
                    f"Invalid rank '{rank}'. Must be one of {valid_ranks}."
                )

            self.suit: str = suit
            self.rank: str = rank
            self.value: int = self.determine_value()

        except Exception as e:
            raise ValueError(f"Error initializing Card: {str(e)}") from e

    def determine_value(self) -> int:
        """
        Determine the value of the card based on its rank.
        J, Q, K are worth 10 points, A is worth either 1 or 11 points, but is initially counted as 11, 
        and numbered cards are worth their face value.

        Returns:
            int: The value of the card.
        """

        if self.rank in ["J", "Q", "K"]:
            return 10
        elif self.rank == "A":
            return 11
        else:
            return int(self.rank)

    def __repr__(self) -> str:
        return f"{self.rank} of {self.suit}"
