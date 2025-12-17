import random
from typing import List
from collections import defaultdict
import numpy as np


def discretize_state(env, obs: np.ndarray) -> tuple:
    """
    Convert continuous observation vector into a small discrete tuple for tabular Q-learning.

    IMPORTANT:
    - This is not "exact Oregon Trail state" (that would be huge).
    - It's a compact abstraction good for getting Q-learning to work first.

    You can refine this over time.
    """
    s = env.state
    assert s is not None

    # Helper bins
    def bin_int(x: int, edges: List[int]) -> int:
        # returns which bucket x falls into
        for i, e in enumerate(edges):
            if x < e:
                return i
        return len(edges)

    def bin_float(x: float, edges: List[float]) -> int:
        for i, e in enumerate(edges):
            if x < e:
                return i
        return len(edges)

    # Key features to discretize (tunable!)
    miles_bin = bin_int(s.trail.miles_total, [200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000])
    money_bin = bin_int(
        s.inventory.money,
        [0, 20, 50, 100, 200, 400, 800]
    )
    wheels_bin = bin_int(s.inventory.wagon_wheels, [0, 1, 2])
    axles_bin = bin_int(s.inventory.wagon_axles, [0, 1, 2])
    tongues_bin = bin_int(s.inventory.wagon_tongues, [0, 1, 2])
    food_bin  = bin_int(s.inventory.food, [50, 100, 200, 350, 500, 700, 900])
    ammo_bin  = bin_int(s.inventory.ammo, [0, 20, 50, 100, 200, 400, 800])
    oxen_bin  = bin_int(s.inventory.oxen, [0, 1, 2, 4, 6, 10])

    # Party survival summary
    alive_count = sum(1 for m in s.party if m.alive)
    alive_bin = alive_count  # already small (0..5)

    # Average health bin (0..1)
    alive_members = [m.health for m in s.party if m.alive]
    avg_health = float(sum(alive_members) / max(len(alive_members), 1))
    health_bin = bin_float(avg_health, [0.2, 0.4, 0.6, 0.8])

    # Discrete contextual variables
    month = s.env.month  # 1..12
    weather = int(s.env.weather.value)
    terrain = int(s.env.terrain.value)
    location = int(s.location_type.value)
    pace = int(s.pace.value)
    rations = int(s.rations.value)

    # Keep key small-ish; tuples are hashable
    return (
        miles_bin,
        food_bin,
        ammo_bin,
        oxen_bin,
        money_bin,  # ← NEW
        wheels_bin,  # ← NEW
        axles_bin,  # ← NEW
        tongues_bin,
        alive_bin,
        health_bin,
        month,
        weather,
        terrain,
        location,
        pace,
        rations,
    )
from collections import defaultdict

def train_q_learning(
    env,
    episodes: int = 2000,
    alpha: float = 0.1,        # learning rate
    gamma: float = 0.99,       # discount
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.999,  # closer to 1.0 = slower decay
    max_steps_per_episode: int = 500,
    seed: int = 0,
):
    rng = random.Random(seed)

    # Q[state_key][action] -> value
    Q = defaultdict(lambda: np.zeros(env.action_space_n, dtype=np.float32))

    eps = eps_start

    stats = {
        "wins": 0,
        "losses": 0,
        "avg_miles_last_100": [],
    }


    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        steps = 0

        # Track for reporting
        start_miles = env.state.trail.miles_total  # should be 0
        last_miles = start_miles

        while not done and steps < max_steps_per_episode:
            steps += 1

            state_key = discretize_state(env, obs)
            legal = env.legal_actions()
            if not legal:
                break  # safety

            # Epsilon-greedy among LEGAL actions only
            if rng.random() < eps:
                action = rng.choice(legal)
            else:
                q_vals = Q[state_key]
                # Mask illegal actions
                masked = np.full(env.action_space_n, -1e9, dtype=np.float32)
                masked[legal] = q_vals[legal]
                action = int(np.argmax(masked))


            next_obs, reward, done, info = env.step(action)
            next_key = discretize_state(env, next_obs)

            # Q-learning update using max over legal next actions
            next_legal = env.legal_actions() if not done else []
            if next_legal:
                next_q = Q[next_key]
                best_next = float(np.max(next_q[next_legal]))
            else:
                best_next = 0.0

            td_target = reward + gamma * best_next
            td_error = td_target - float(Q[state_key][action])
            Q[state_key][action] = float(Q[state_key][action]) + alpha * td_error

            obs = next_obs
            last_miles = env.state.trail.miles_total

        # Episode end stats
        if env.state.win:
            stats["wins"] += 1
        elif env.state.game_over:
            stats["losses"] += 1

        # Decay epsilon
        eps = max(eps_end, eps * eps_decay)

        # Report every 100 episodes
        if ep % 100 == 0:
            avg_miles = last_miles
            stats["avg_miles_last_100"].append(avg_miles)

            win_rate = stats["wins"] / ep
            print(
                f"Episode {ep:5d} | eps={eps:.3f} | win_rate={win_rate:.3f} | last_miles={last_miles} | alive={sum(1 for m in env.state.party if m.alive)} "
                f"| food={env.state.inventory.food} | oxen={env.state.inventory.oxen}"
            )
    #print(Q, stats)

    return Q, stats
def run_greedy_episode(env, Q, max_steps: int = 500, seed: int = 123):
    rng = random.Random(seed)
    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        steps += 1
        key = discretize_state(env, obs)
        legal = env.legal_actions()
        if not legal:
            break

        q_vals = Q[key]
        masked = np.full(env.action_space_n, -1e9, dtype=np.float32)
        masked[legal] = q_vals[legal]
        action = int(np.argmax(masked))

        obs, reward, done, info = env.step(action)

    return {
        "win": env.state.win,
        "miles": env.state.trail.miles_total,
        "alive": sum(1 for m in env.state.party if m.alive),
        "food": env.state.inventory.food,
        "oxen": env.state.inventory.oxen,
        "month": env.state.env.month,
    }
