vfrom .qrandom import *
from .qreplaynetwork import *
from .qtable imporfrom .qrandom import *
from .qreplaynetwork import *
from .qtable import *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *t *
from .qtable_trace import *
from .sarsa import *
from .sarsa_trace import *from abc import ABC, abstractmethod

class AbstractModel(ABC):
    def __init__(self, maze, **kwargs):
        self.environment = maze
        self.name = kwargs.get("name", "model")

    def load(self, filename):
        """ Load model from file. """
        pass

    def save(self, filename):
        """ Save model to file. """
        pass

    def train(self, stop_at_convergence, **kwargs):
        """ Train model. """
        pass

    @abstractmethod
    def q(self, state):
        """ Return q values for state. """
        pass

    @abstractmethod
    def predict(self, state):
        """ Predict value based on state. """
        pass