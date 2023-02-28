from bandit.algos.base import BaseAgent


class NaiveAgent(BaseAgent):
    """
    An agent who only takes one specific action.

    Args:
        action: action
    """

    def __init__(self, action: int):
        super().__init__()
        self.action = action

    def get_action(self) -> int:
        return self.action
