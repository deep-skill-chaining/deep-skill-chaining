'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import random
import pdb
import time

# Other imports.
import gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.fixed_reacher.FixedReacherStateClass import FixedReacherState

class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

class FixedReacherMDP(MDP):
    ''' Class for Mujoco + Gym Reacher MDP with a fixed goal '''

    def __init__(self, env_name='Reacher-v2', render=False):
        '''
        Args:
            env_name (str)
        '''
        self.env_name = env_name
        self.env = NormalizedEnv(gym.make(env_name))
        self.render = render

        MDP.__init__(self, range(self.env.action_space.shape[0]), self._transition_func, self._reward_func,
                     init_state=FixedReacherState(self.env.reset()))

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, gym_reward, gym_terminal, info = self.env.step(action)

        goal_distance = abs(info["reward_dist"])
        done = goal_distance < 0.02

        reward = +10 if done else info["reward_ctrl"] - 0.01

        if self.render:
            self.env.render()

        self.next_state = FixedReacherState(obs, is_terminal=done)

        # if done: time.sleep(3)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "gym-" + str(self.env_name)
