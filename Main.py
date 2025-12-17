from collections import defaultdict

from game_files.Oregon_Trail_Classes import Action
from game_files.Oregon_Trail_Game_State import OregonTrailEnv
from q_learning.Q_learning import train_q_learning, run_greedy_episode


def main():

    env = OregonTrailEnv(seed=0)

    Q, stats = train_q_learning(
        env,
        episodes = 2000000,
    alpha = 0.08,  # learning rate
    gamma = 0.9990,  # discount
    eps_start = 1.0,
    eps_end = 0.05,
    eps_decay = 0.999,  # closer to 1.0 = slower decay
    max_steps_per_episode = 10000,
    seed = 0,
    )

    # Evaluate greedy runs
    total_survivors = 0
    wins = 0
    N = 2000
    terrain_counts = defaultdict(lambda: defaultdict(int))
    location_counts = defaultdict(lambda: defaultdict(int))

    for i in range(N):
        eval_env = OregonTrailEnv(seed=1000 + i)
        result = run_greedy_episode(eval_env, Q, seed=2000 + i, max_steps=1500)

        for terrain, adict in eval_env.action_counts_by_terrain.items():
            for a, c in adict.items():
                terrain_counts[terrain][a] += c

        for loc, adict in eval_env.action_counts_by_location.items():
            for a, c in adict.items():
                location_counts[loc][a] += c

    def print_action_table(title, mapping):
        print("\n" + title)
        for k in mapping:
            print(f"== {k} ==")
            for a in sorted(mapping[k]):
                print(f"  {Action(a).name}: {mapping[k][a]}")


    print_action_table("Actions by Terrain", terrain_counts)
    print_action_table("Actions by Location", location_counts)

    print(f"Greedy policy win rate over {N} eval episodes: {wins / N:.5f}")
    print(f"Average total who made it per journey: {total_survivors / wins:.5f}")

if __name__ == "__main__":
    main()