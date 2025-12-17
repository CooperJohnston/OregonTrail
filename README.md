# Oregon Trail RL (Q-Learning)
<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/2a0a63d6-f4fc-44d9-91de-d6dd85c70cf2" />

A simplified, reinforcement-learning–friendly recreation of **Oregon Trail**, built to experiment with strategy discovery using a **tabular Q-learning** agent. The environment is “Gym-ish” (reset/step), supports different location types (trail, rivers, forts, landmarks), and includes randomized Oregon Trail–style events (e.g., sickness, breakdowns) to keep the problem stochastic and challenging.

The primary goal is to train an agent that learns which decisions—pace, rations, resting, hunting, river-crossing choices, and purchasing supplies—lead to the best long-term outcome: reaching Oregon.

---

## What This Project Does

- Simulates an Oregon Trail–like journey with:
  - **Terrains** (plains, hills, mountains)
  - **Locations** (trail, river, fort, landmark)
  - **Resources** (food, ammo, money, etc. — simplified)
  - **Random events** (disease, cart breakdowns, and other “trail chaos”)
- Trains a **tabular Q-learning agent** to select actions from each state
- Collects and summarizes results, including:
  - action frequencies by terrain and location
  - survival statistics
  - win rate and distance traveled before failure

---

## How Q-Learning Is Used

This project uses **Q-learning**, a reinforcement learning algorithm in which an agent learns the expected long-term value of taking a specific action in a given state. The agent interacts with the environment repeatedly, receiving rewards or penalties, and updates a table of Q-values over time.

Because Oregon Trail involves delayed rewards, uncertainty, and sequential decision-making, Q-learning is well-suited to exploring effective strategies. The agent gradually learns to favor actions that maximize long-term success rather than short-term gains.

---

## Repository Structure

> Filenames may vary slightly depending on your local setup.

### Core Environment and Game Logic

- **`game_files/`**
  - `Oregon_Trail_Classes.py`  
    Defines core data structures and enums, including party members, inventory, terrain types, locations, actions, and overall game state.
  - `Oregon_Trail_Encoding.py`  
    Encodes the current game state into a compact representation used by the Q-learning agent.

- **`oregon_trail_game_state.py`** 
  Implements the reinforcement-learning environment using a Gym-style API:
  - `reset()` returns the initial observation
  - `step(action)` advances the simulation and returns `(observation, reward, done, info)`

---

### Training and Evaluation

- **`Main.py`**  
  Runs Q-learning training loops, manages exploration vs. exploitation, updates the Q-table, and logs performance metrics.

---

### Analysis and Reporting

- **`Q_Learning.py`** 
  Aggregates logs and produces summaries such as:
  - actions taken by terrain and location
  - average number of survivors
  - success rates

---
