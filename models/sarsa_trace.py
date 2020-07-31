import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognition

from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import apply_offsets
from utils.inference import load_detection_model

from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_bounding_box
from utils.preprocessor import preprocess_input

import matplotlib.pyplot as plt

from environment import Maze
from models import *

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S")

maze = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0]
])  # 0 = free, 1 = occupied

game = Maze(maze)

if 0:  # only show the maze
    game.render("moves")
    game.reset()

if 0:  # play using random model
    model = RandomModel(game)
    model.train()

if 0:  # train using tabular Q-learning
    model = QTableModel(game, name="QTableModel")
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)

if 0:  # train using tabular Q-learning and an eligibility trace (aka TD-lamba)
    model = QTableTraceModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)

if 0:  # train using tabular SARSA learning
    model = SarsaTableModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)

if 1:  # train using tabular SARSA learning and an eligibility trace
    game.render("training")  # shows all moves and the q table; nice but slow.
    model = SarsaTableTraceModel(game)
    h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)

if 0:  # train using a neural network with experience replay (also saves the resulting model)
    model = QReplayNetworkModel(game)
    h, w, _, _ = model.train(discount=0.80, exploration_rate=0.10, episodes=maze.size * 10, max_memory=maze.size * 4)

try:
    h  # force a NameError exception if h does not exist (and thus don't try to show win rate and cumulative reward)
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.set_window_title(model.name)
    ax1.plot(*zip(*w))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass

if 0:  # load a previously trained model
    model = QReplayNetworkModel(game, load=True)

if 0:  # compare learning speed (cumulative rewards and win rate) of several models in a diagram
    rhist = list()
    whist = list()
    names = list()

    models = [0, 1, 2, 3, 4]

    for model_id in models:
        logging.disable(logging.WARNING)
        if model_id == 0:
            model = QTableModel(game, name="QTableModel")
        elif model_id == 1:
            model = SarsaTableModel(game, name="SarsaTableModel")
        elif model_id == 2:
            model = QTableTraceModel(game, name="QTableTraceModel")
        elif model_id == 3:
            model = SarsaTableTraceModel(game, name="SarsaTableTraceModel")
        elif model_id == 4:
            model = QReplayNetworkModel(game, name="QReplayNetworkModel")

        r, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, exploration_decay=0.999, learning_rate=0.10,
                                 episodes=300)
        rhist.append(r)
        whist.append(w)
        names.append(model.name)

    f, (rhist_ax, whist_ax) = plt.subplots(2, len(models), sharex="row", sharey="row", tight_layout=True)

    for i in range(len(rhist)):
        rhist_ax[i].set_title(names[i])
        rhist_ax[i].set_ylabel("cumulative reward")
        rhist_ax[i].plot(rhist[i])

    for i in range(len(whist)):
        whist_ax[i].set_xlabel("episode")
        whist_ax[i].set_ylabel("win rate")
        whist_ax[i].plot(*zip(*(whist[i])))

    plt.show()

if 0:  # run a number of training episodes and plot the training time and episodes needed in histograms (time consuming)
    runs = 10

    epi = list()
    nme = list()
    sec = list()

    models = [0, 1, 2, 3, 4]

    for model_id in models:
        episodes = list()
        seconds = list()

        logging.disable(logging.WARNING)
        for r in range(runs):
            if model_id == 0:
                model = QTableModel(game, name="QTableModel")
            elif model_id == 1:
                model = SarsaTableModel(game, name="SarsaTableModel")
            elif model_id == 2:
                model = QTableTraceModel(game, name="QTableTraceModel")
            elif model_id == 3:
                model = SarsaTableTraceModel(game, name="SarsaTableTraceModel")
            elif model_id == 4:
                model = QReplayNetworkModel(game, name="QReplayNetworkModel")

            _, _, e, s = model.train(stop_at_convergence=True, discount=0.90, exploration_rate=0.10,
                                     exploration_decay=0.999, learning_rate=0.10, episodes=1000)

            print(e, s)

            episodes.append(e)
            seconds.append(s.seconds)

        logging.disable(logging.NOTSET)
        logging.info("model: {} | trained {} times | average no of episodes: {}| average training time {}"
                     .format(model.name, runs, np.average(episodes), np.sum(seconds) / len(seconds)))

        epi.append(episodes)
        sec.append(seconds)
        nme.append(model.name)

    f, (epi_ax, sec_ax) = plt.subplots(2, len(models), sharex="row", sharey="row", tight_layout=True)

    for i in range(len(epi)):
        epi_ax[i].set_title(nme[i])
        epi_ax[i].set_xlabel("training episodes")
        epi_ax[i].hist(epi[i], edgecolor="black")

    for i in range(len(sec)):
        sec_ax[i].set_xlabel("seconds per episode")
        sec_ax[i].hist(sec[i], edgecolor="black")

    plt.show()

game.render("moves")
game.play(model, start_cell=(0, 0))
# game.play(model, start_cell=(2, 5))
# game.play(model, start_cell=(4, 1))

plt.show()  # must be placed here else the image disappears immediately at the end of the programimport random
from datetime import datetime

import numpy as np

from models import AbstractModel


class SarsaTableTraceModel(AbstractModel):
    """ Tabular SARSA based prediction model with eligibility trace.

        For every state (here: the agents current location ) the value for each of the actions is stored in a table.
        The key for this table is (state + action). Initially all values are 0. When playing training games
        after every move the value in the table is updated based on the reward gained after making the move. Training
        ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).

        To speed up learning the model keeps track of the (state, action) pairs which have been visited before and
        also updates their values based on the current reward (a.k.a. eligibility trace). With every step the amount
        in which previous values are updated decays.

        :param class Maze game: Maze game object.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.Q = dict()  # table with Q per (state, action) combination

    def train(self, stop_at_convergence=False, **kwargs):
        """ Hyperparameters:

            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
            :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
            :keyword float eligibility_decay: (lambda) eligibility trace decay rate per step (0 = no trace, 1 = no decay)
            :keyword int episodes: number of training games to play
            :return int, datetime: number of training episodes, total time spent
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        eligibility_decay = kwargs.get("eligibility_decay", 0.80)  # 0.80 = 20% reduction
        episodes = kwargs.get("episodes", 1000)

        # variables for performance reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()

        for episode in range(1, episodes + 1):
            # optimization: make sure to start from all possible cells
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change np.ndarray to tuple so it can be used as dictionary key

            etrace = dict()

            if np.random.random() < exploration_rate:
                action = random.choice(self.environment.actions)
            else:
                action = self.predict(state)

            while True:
                try:
                    etrace[(state, action)] += 1
                except KeyError:
                    etrace[(state, action)] = 1

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())
                next_action = self.predict(next_state)

                cumulative_reward += reward

                if (state, action) not in self.Q.keys():
                    self.Q[(state, action)] = 0.0

                next_Q = self.Q.get((next_state, next_action), 0.0)

                delta = reward + discount * next_Q - self.Q[(state, action)]

                for key in etrace.keys():
                    self.Q[key] += learning_rate * delta * etrace[key]

                # decay eligibility trace
                for key in etrace.keys():
                    etrace[key] *= (discount * eligibility_decay)

                if status in ("win", "lose"):  # terminal state reached, stop training episode
                    break

                state = next_state
                action = next_action  # SARSA is on-policy: always follow the predicted action

                self.environment.render_q(self)

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status, exploration_rate))

            if episode % 5 == 0:
                # check if the current model wins from all starting cells
                # can only do this if there is a finite number of starting states
                w_all, win_rate = self.environment.win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  # explore less as training progresses

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: Game state.
            :return int: Chosen action.
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)
