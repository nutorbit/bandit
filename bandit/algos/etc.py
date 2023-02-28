import numpy as np

from bandit.algos.greedy import GreedyAgent


class ETCAgent(GreedyAgent):
    """
    A Explore-Then-Commit (ETC) agent

    Args:
        n_actions: number of possible actions
        n_explore: number of timesteps for exploration 
        seed: random seed
    """

    def __init__(self, n_actions: int, n_explore: int, seed: int = 123):
        super().__init__(n_actions)
        self.n_explore = n_explore
        self.rng = np.random.default_rng()

    def get_action(self) -> int:
        if self.timesteps <= self.n_explore:
            action = self.rng.integers(0, self.n_actions)
        else:
            action = super().get_action()
        return action
