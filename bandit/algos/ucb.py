import numpy as np
import pandas as pd

from bandit.algos.greedy import GreedyAgent


class UCBAgent(GreedyAgent):
    """
    An Upper Confidence Bound agent

    Args:
        n_actions: number of possible actions
        c: hyperparamer
    """

    def __init__(self, n_actions: int, c: float = 0.01):
        super().__init__(n_actions)
        self.c = c

    def calculate_expected_reward(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)
        expected_reward_action = df.groupby("action")["reward"].mean()
        n_action = df.groupby("action")["reward"].count()
        expected_reward_action += self.c * np.sqrt(np.log(self.timesteps) / n_action)
        return expected_reward_action.sort_values(ascending=False)
