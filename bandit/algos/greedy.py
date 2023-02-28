import numpy as np
import pandas as pd

from bandit.algos.base import BaseAgent


class GreedyAgent(BaseAgent):
    """
    A greedy agent who takes only action with highest expected reward

    Args:
        n_actions: number of possible actions
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.initilize_history()

    def initilize_history(self):
        for act in range(self.n_actions):
            self.history["reward"].append(1 + np.random.rand())
            self.history["action"].append(act)

    def calculate_expected_reward(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)
        expected_reward_action = df.groupby("action")["reward"].mean()
        return expected_reward_action.sort_values(ascending=False)

    def get_action(self) -> int:
        expected_reward_action = self.calculate_expected_reward()
        best_action = expected_reward_action.index[0]
        return best_action
