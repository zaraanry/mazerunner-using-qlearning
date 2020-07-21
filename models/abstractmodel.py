import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognitionimport numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognitionimport numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognitionimport numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognitionfrom abc import ABC, abstractmethod

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