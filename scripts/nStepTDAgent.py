# ExpectedSarsaAgent.py

import sys

from tabulate import tabulate
import numpy as np

episode = [["s1", "E", 0], 
           ["s2", "E", 1],
           ["s3", "N", 2], 
           ["s3", "N", 3], 
           ["s3", "S", 4],
           ["s6", "S", 5],
           ["s9", None, None]]

index_map = {
    "s1": 0,
    "s2": 1,
    "s3": 2,
    "s4": 3,
    "s5": 4,
    "s6": 5,
    "s7": 6,
    "s8": 7,
    "s9": 8,
    "N": 0,
    "E": 1,
    "S": 2,
    "W": 3
}

class nStepTDAgent():
    def __init__(self, alpha, gamma, num_state, num_actions):
        """
        Constructor
        Args:
            epsilon: The degree of exploration
            gamma: The discount factor
            num_state: The number of states
            num_actions: The number of actions
        """
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = num_state
        self.num_actions = num_actions

        self.Q = np.zeros((self.num_state, self.num_actions))

    def run_episode(self, n, episode):
        """
        Update the action value function using the n-step TD update.
        """
        
        rew = [0, ]
        
        bigT = sys.maxsize
        print("T: ", bigT)
        for t, step in enumerate(episode.reverse()):
            print("=" * 80)
            print("Step: ", t)
            if t < bigT:
                s_t, a_t, r_t1 = step
                print(f" s_t: {s_t}, a_t: {a_t}, r_t1: {r_t1}")
                s_t1, _, _ = episode[t + 1]
                rew.append(r_t1)
                
                _, _, r_t2 = episode[t + 1]
                if r_t2 is None:
                    bigT = t + 1
                    print("TERMINAL => T: ", bigT)
                    
            Tt = t - n + 1
            print(f" Tt: {Tt}")
            if Tt >= 0:
                print(f' ==============')
                bigG = 0
                for i in range(Tt + 1, min(Tt + n , bigT) + 1):
                    print(f" i: {i}")
                    r_t1 = rew[i]
                    print(f" r_t{i}: {r_t1}")
                    print(f"      {bigG} += {self.gamma}^{i - Tt - 1} * {r_t1}")
                    bigG += self.gamma**(i - Tt - 1) * r_t1
                print(f" G: {bigG}")
                print(f' --------------')
                if Tt + n < bigT:
                    s_Tn, a_Tn = episode[Tt + n][0], episode[Tt + n][1]
                    
                    print(f"   s_Tn: {s_Tn}, a_Tn: {a_Tn}")
                    s_Tn, a_Tn = index_map[s_Tn], index_map[a_Tn]
                    print(f"      {bigG} += {self.gamma}^{n} * {self.Q[s_Tn, a_Tn]}")
                    bigG += (self.gamma**n) * self.Q[s_Tn, a_Tn]
                print(f" G: {bigG}")
                print(f' ==============')
                
                s_Tt, a_Tt = episode[Tt][0], episode[Tt][1]
                print(f" => Update Q[{s_Tt}, {a_Tt}]")
                s_Tti, a_Tti = index_map[s_Tt], index_map[a_Tt]
                print(f" Q[{s_Tt}, {a_Tt}] = {self.Q[s_Tti, a_Tti]}")
                self.Q[s_Tti, a_Tti] += self.alpha * (bigG - self.Q[s_Tti, a_Tti])
                print(f" Q[{s_Tt}, {a_Tt}] = {self.Q[s_Tti, a_Tti]}")
            print(f"Q:")
            print(tabulate(self.Q, tablefmt="fancy_grid"))
            if Tt == bigT - 1:
                break
                    
                




def main_r():
    print("# nStepTDAgent.py")
    agent = nStepTDAgent(0.1, 0.9, 9, 4)
    print(agent.Q)
    agent.run_episode(3, episode)



if __name__ == "__main__":
    main_r()
