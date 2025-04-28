from src.PlayerVsAI import PlayerVsAI


def main():
    model_path = "models/dqn_blackjack.pth"
    game = PlayerVsAI(model_path=model_path)
    game.render_game()


if __name__ == "__main__":
    main()
