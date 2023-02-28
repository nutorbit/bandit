from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Base agent class
    """

    def __init__(self):
        self.history = {"reward": [], "action": []}

    @property
    def timesteps(self) -> int:
        return len(self.history["reward"])

    @abstractmethod
    def get_action(self) -> int:
        pass

    def save_feedback(self, action: int, reward: int):
        self.history["reward"].append(reward)
        self.history["action"].append(action)
