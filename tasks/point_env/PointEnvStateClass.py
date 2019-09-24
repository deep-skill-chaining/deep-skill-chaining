# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State


class PointEnvState(State):
    def __init__(self, position, velocity, done):
        self.position = position
        self.velocity = velocity

        State.__init__(self, np.concatenate((position, velocity), axis=0), is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\txdot: {}\tydot: {}\tterminal: {}\n".format(self.position[0], self.position[1],
                                                                         self.velocity[0], self.velocity[1],
                                                                         self.is_terminal())

    def __repr__(self):
        return str(self)

