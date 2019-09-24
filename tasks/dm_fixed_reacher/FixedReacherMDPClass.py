# Python imports.
import numpy as np
import pdb
from PIL import Image

# Other imports.
from dm_control import suite
from dm_control import viewer
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.dm_fixed_reacher.FixedReacherStateClass import FixedReacherState


class FixedReacherMDP(MDP):
    def __init__(self, seed, difficulty="easy", render=False):
        self.seed = seed
        self.env_name = "reacher"
        self.env = suite.load(self.env_name, difficulty, visualize_reward=True, task_kwargs={"random": seed})
        self.render = render

        # Debug logs
        self.committed_actions = []
        self.time_step = 0

        if render:
            viewer.launch(self.env)

        MDP.__init__(self, range(self.env.action_spec().minimum.shape[0]), self._transition_func, self._reward_func,
                     init_state=FixedReacherState(self.env.reset().observation))

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (np.array)

        Returns
            (float)
        '''
        time_limit = self.env.step(action)

        # DM Control Suite gives us a reward of {None, 0, 1}. We want the
        # reward to be -1 as a step penalty and zero terminal reward for
        # hitting the goal state.
        reward = time_limit.reward if time_limit.reward is not None else -1.
        reward = 0. if reward > 0 else -1.
        observation = time_limit.observation
        done = reward == 0.

        # TODO: Figure out how to render in DMCS
        if self.render:
            image_data = self.env.physics.render(height=480, width=480)
            img = Image.fromarray(image_data, "RGB")
            img.save("frames/frame-{}.png".format(self.time_step))
            # viewer.launch(self.env)

        self.next_state = FixedReacherState(observation, is_terminal=done)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (np.array)

        Returns
            (State)
        '''
        return self.next_state

    def execute_agent_action(self, action, option_idx=None):
        # TODO: Do something with option_idx
        # Debugging
        self.committed_actions.append(action)
        self.time_step += 1

        reward, next_state = super(FixedReacherMDP, self).execute_agent_action(action)
        return reward, next_state

    @staticmethod
    def is_goal_state(state):
        """ We are defining terminal state in this MDP as a goal state. """
        return state.is_terminal()

    def is_primitive_action(self, action):
        x_min, y_min = self.env.action_spec().minimum[0], self.env.action_spec().minimum[1]
        x_max, y_max = self.env.action_spec().maximum[0], self.env.action_spec().maximum[1]
        return x_min <= action[0] <= x_max and y_min <= action[1] <= y_max

    def reset(self):
        self.init_state = FixedReacherState(self.env.reset().observation)
        super(FixedReacherMDP, self).reset()

    def __str__(self):
        return "dm_control_suite_" + str(self.env_name)
