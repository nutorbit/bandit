import numpy as np

from bandit.algos.greedy import GreedyAgent


class EpsilonGreedyAgent(GreedyAgent):
    """
    An epsilon greedy agent

    Args:
        n_actions: number of possible actions
        epsilon: epsilon hyperparameter
        seed: random seed
    """

    def __init__(self, n_actions: int, epsilon: float = 0.2, seed: int = 123):
        super().__init__(n_actions)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def get_action(self) -> int:
        if self.rng.random() <= self.epsilon:
            action = self.rng.integers(0, self.n_actions)
        else:
            action = super().get_action()
        return action
