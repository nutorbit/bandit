import numpy as np

from typing import List, Dict


class SlotMachine:
    """
    A simple bernoulli slot machine simulation

    Args:
        reward_rates: the percentage of jackpot for each particular slot
        seed: random seed
    """

    def __init__(self, reward_rates: List = [0.2, 0.2, 0.3], seed: int = 123):
        self.reward_rates = reward_rates
        self.n_slots = len(reward_rates)
        self.rng = np.random.default_rng(seed)
        self.history = {}
        self.reset()

    @property
    def timesteps(self) -> int:
        return len(self.history["rewards"])

    def reset(self):
        self.history = {"rewards": [], "action": []}

    def get_performance_report(self) -> Dict:
        rewards = np.array(self.history["rewards"])
        action = np.array(self.history["action"])
        best_action = rewards.argmax(axis=1)

        indices = np.arange(self.timesteps).astype(int)

        reward_action = rewards[indices, action]
        best_reward_action = rewards[indices, best_action]

        return {
            "total_timesteps": self.timesteps,
            "total_reward": reward_action.sum(),
            "cummulative_reward": reward_action.cumsum(),
            "cummulative_regret": (best_reward_action - reward_action).cumsum()
        }

    def pull(self, slot_id: int) -> int:
        assert slot_id < self.n_slots, \
        f"slot_id should be less than {self.n_slots}"

        s = self.rng.uniform(size=self.n_slots)
        rewards = (s <= self.reward_rates).astype(int)
        self.history["rewards"].append(rewards)
        self.history["action"].append(slot_id)
        return rewards[slot_id]
