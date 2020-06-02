from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *import numpy as np

from models import AbstractModel


class RandomModel(AbstractModel):
    """ Prediction model which randomly chooses the next action. """

    def __init__(self, game):
        super().__init__(game)

    def q(self, state):
        """ Return Q value for all action for a certain state.

            :return np.ndarray: Q values
        """
        return np.array([0, 0, 0, 0])

    def predict(self, **kwargs):
        """ Randomly choose the next action.

            :return int: Chosen action.
        """
        return random.choice(self.environment.actions)