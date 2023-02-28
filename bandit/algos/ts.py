import numpy as np
import pandas as pd

from bandit.algos.greedy import GreedyAgent


class TSAgent(GreedyAgent):
    """
    A Thompson Sampling agent with Beta distribution

    Args:
        n_actions: number of possible actions
        c: hyperparamer
    """

    def __init__(self, n_actions: int, c: float = 0.01, seed: int = 123):
        super().__init__(n_actions)
        self.c = c
        self.rng = np.random.default_rng(seed)

    def calculate_expected_reward(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)
        n_action = df.groupby("action")["reward"].count()
        n_success = df.groupby("action")["reward"].sum()
        n_failure = n_action - n_success
        expected_reward_action = self.rng.beta((n_success + 1) / self.c, (n_failure + 1) / self.c)
        return pd.Series(expected_reward_action, index=n_action.index).sort_values(ascending=False)
