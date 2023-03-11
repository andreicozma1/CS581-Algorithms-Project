import numpy as np


episodes = [
    [["A", "a1", 3], ["A", "a2", 2], ["B", "a1", -4], ["A", "a1", 4], ["B", "a1", -3]],
    [["B", "a1", -2], ["A", "a1", 3], ["B", "a2", -3]],
]

index_map = {
    "states": {
        "A": 0,
        "B": 1,
    },
    "actions": {
        "a1": 0,
        "a2": 1,
    },
}


def main_r():
    print("# MonteCarloAgent.py")
    alpha = 0.1
    num_states = 2

    v = np.zeros(num_states)
    rets = {s: [] for s in index_map["states"].keys()}

    for ep in episodes:
        print("=" * 80)
        g = 0
        ep_len = len(ep)
        print(f"# Episode: {ep} (steps: {ep_len}) G: {g}")
        for t in range(ep_len - 1, -1, -1):
            s, a, r = ep[t]
            si = index_map["states"][s]
            g = g + r
            print(f"# Step {t + 1}:")
            print(f"\ts: {s}, a: {a}, r: {r}")
            print(f"\tG: {g}")
            # unless st appears in the episode before time t
            if s not in [x[0] for x in ep[:t]]:
                rets[s].append(g)
                v[si] = alpha * (sum(rets[s]) / len(rets[s]))
                # v[si] = v[si] + alpha * (g - v[si])

                print(f"\tV[{s}] = {v[si]}")


if __name__ == "__main__":
    main_r()
