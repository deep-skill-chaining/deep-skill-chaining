# Python imports.
import numpy as np
import pdb
import sys

# Other imports.
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.point_env.PointEnvStateClass import PointEnvState


class PointEnvMDP(MDP):
    def __init__(self, init_mean=(-0.2, -0.2), control_cost=False, dense_reward=False, render=False):
        xml = os.path.join(os.path.expanduser("~"), "git-repos/dm_control/dm_control/suite/point_mass.xml")
        model = load_model_from_path(xml)
        self.sim = MjSim(model)
        self.render = render
        self.init_mean = init_mean
        self.control_cost = control_cost
        self.dense_reward = dense_reward

        if self.render: self.viewer = MjViewer(self.sim)

        # Config
        self.env_name = "Point-Mass-Environment"
        self.target_position = np.array([0., 0.])
        self.target_tolerance = 0.02
        self.init_noise = 0.05

        self._initialize_mujoco_state()
        self.init_state = self.get_state()

        print("Loaded {} with dense_reward={}".format(self.env_name, self.dense_reward))

        MDP.__init__(self, [0, 1], self._transition_func, self._reward_func, self.init_state)

    def _reward_func(self, state, action):
        self.next_state = self._step(action)
        if self.render: self.viewer.render()
        if self.dense_reward:
            reward = -np.linalg.norm(self.next_state.position - self.target_position)
        else:
            reward = +0. if self.next_state.is_terminal() else -1.
        control_cost = 0.1 * np.linalg.norm(action)
        if self.control_cost: reward = reward - control_cost
        return reward

    def _transition_func(self, state, action):
        return self.next_state

    def execute_agent_action(self, action, option_idx=None):
        if self.render and option_idx is not None:
            self.viewer.add_marker(pos=np.array([0, 0, 0.1]), size=0.001 * np.ones(3), label="Option {}".format(option_idx))
        reward, next_state = super(PointEnvMDP, self).execute_agent_action(action)
        return reward, next_state

    def is_goal_state(self, state):
        position = state.features()[:2] if isinstance(state, PointEnvState) else state[:2]
        return self.is_goal_position(position)

    def is_goal_position(self, position):
        distance = np.linalg.norm(position - self.target_position)
        return distance <= self.target_tolerance

    def get_state(self):

        # Individually indexing to prevent Mujoco from altering state history
        x, y = self.sim.data.qpos[0], self.sim.data.qpos[1]
        x_dot, y_dot = self.sim.data.qvel[0], self.sim.data.qvel[1]

        # Create State object from simulator state
        position = np.array([x, y])
        velocity = np.array([x_dot, y_dot])

        # State is terminal when it is the goal state
        done = self.is_goal_position(position)
        state = PointEnvState(position, velocity, done)

        return state

    def _step(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.step()
        return self.get_state()

    def _initialize_mujoco_state(self):
        init_position = np.array(self.init_mean) + np.random.uniform(0., self.init_noise, 2)
        init_state = MjSimState(time=0., qpos=init_position, qvel=np.array([0., 0.]), act=None, udd_state={})
        self.sim.set_state(init_state)

    @staticmethod
    def state_space_size():
        return 4

    @staticmethod
    def action_space_size():
        return 2

    @staticmethod
    def is_primitive_action(action):
        return -1. <= action.all() <= 1.

    def reset(self):
        self._initialize_mujoco_state()
        self.init_state = self.get_state()

        super(PointEnvMDP, self).reset()

    def __str__(self):
        return self.env_name
