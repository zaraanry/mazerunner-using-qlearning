import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognitionimport numpy as np

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