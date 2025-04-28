# Blackjack DQN Agent

Train an AI agent to play Blackjack using Deep Q-Learning (DQN) with PyTorch and play against it using a simple Streamlit app!

---

## Project Structure

```
BlackjackDQN/
├── models/                 # Saved models
├── src/
│   ├── Agent.py            # DQN agent
│   ├── BlackjackEnv.py     # Blackjack environment
│   ├── Card.py             # Card class
│   ├── Deck.py             # Deck class
│   ├── DQN.py              # DQN neural network
│   ├── Hand.py             # Hand class
│   ├── PlaverVsAI.py       # Streamlit interface
│   ├── ReplayBuffer.py     # Replay buffer
│   └── Trainer.py          # Trainer class
├── Playing.py              # Launch Streamlit app
├── Training.ipynb          # Train the agent
├── README.md
└── requirements.txt
```

---

## 🎮 How to Play Against the Trained AI

1. Make sure the model `models/dqn_blackjack.pth` exists.
2. Run the Streamlit app:

```bash
streamlit run Playing.py
```

- Click `Hit` or `Stick` to play.
- After your turn, the AI will automatically play its turn.
- Result (win/lose/draw) will be shown at the end.


---

## Project Features

- Fully Object-Oriented Design (OOP)
- Error Handling and Documentation
- Lightweight Neural Network
- Local and Target Networks with Soft Updates
- Epsilon-Greedy Strategy
- Simple, Beautiful Streamlit Interface
