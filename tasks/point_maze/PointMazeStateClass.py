# Python imports.
import numpy as np

# Other imports.
from simple_rl.mdp.StateClass import State

class PointMazeState(State):
    def __init__(self, position, theta, velocity, theta_dot, done):
        """
        Args:
            position (np.ndarray)
            theta (float)
            velocity (np.ndarray)
            theta_dot (float)
            done (bool)
        """
        self.position = position
        self.theta = theta
        self.velocity = velocity
        self.theta_dot = theta_dot
        features = [position[0], position[1], theta, velocity[0], velocity[1], theta_dot]

        State.__init__(self, data=features, is_terminal=done)

    def __str__(self):
        return "x: {}\ty: {}\ttheta: {}\txdot: {}\tydot: {}\tthetadot: {}\tterminal: {}\n".format(self.position[0],
                                                                                                  self.position[1],
                                                                                                  self.theta,
                                                                                                  self.velocity[0],
                                                                                                  self.velocity[1],
                                                                                                  self.theta_dot,
                                                                                                  self.is_terminal())

    def __repr__(self):
        return str(self)