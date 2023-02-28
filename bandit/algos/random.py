import numpy as np

from bandit.algos.base import BaseAgent


class RandomAgent(BaseAgent):
    """
    A random agent

    Args:
        n_actions: number of possible actions
        seed: random seed
    """

    def __init__(self, n_actions: int, seed: int = 123):
        super().__init__()
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def get_action(self) -> int:
        return self.rng.integers(0, self.n_actions)
