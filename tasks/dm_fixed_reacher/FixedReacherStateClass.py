# Python imports
import numpy as np

# Local imports
from simple_rl.mdp.StateClass import State


class FixedReacherState(State):
    ''' Fixed State class '''

    def __init__(self, observation, is_terminal=False):
        self.position = observation["position"]
        self.velocity = observation["velocity"]
        self.to_target = observation["to_target"]
        data = np.concatenate((self.position, self.velocity, self.to_target), axis=0)

        State.__init__(self, data=data, is_terminal=is_terminal)

    def __str__(self):
        return "position: {}\tvelocity: {}\tto_target: {}\tterminal: {}".format(self.position, self.velocity,
                                                                                self.to_target, self.is_terminal())

    def __repr__(self):
        return str(self)

    def to_rgb(self, x_dim, y_dim):

        # 3 by x_length by y_length array with values 0 (0) --> 1 (255)
        board = np.zeros(shape=[3, x_dim, y_dim])
        # print self.data, self.data.shape, x_dim, y_dim
        return self.data