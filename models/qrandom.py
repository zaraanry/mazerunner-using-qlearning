from abc import ABC, abstractmethod

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
        passfrom abc import ABC, abstractmethod

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
        passfrom abc import ABC, abstractmethod

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
        passfrom abc import ABC, abstractmethod

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
        passfrom abc import ABC, abstractmethod

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
        passimport random

import numpy as np

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