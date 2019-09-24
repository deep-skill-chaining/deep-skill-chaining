# Python imports.
from collections import deque
import random
import numpy as np

# Other imports.
from simple_rl.agents.func_approx.ddpg.hyperparameters import BUFFER_SIZE, BATCH_SIZE

class ReplayBuffer(object):
    def __init__(self, buffer_size=BUFFER_SIZE, name_buffer='', seed=0):
        self.buffer_size = buffer_size
        self.num_exp = 0
        self.memory = deque(maxlen=buffer_size)
        self.name = name_buffer

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, terminal):
        assert isinstance(state, np.ndarray) and isinstance(action, np.ndarray) and \
               isinstance(reward, (int, float)) and isinstance(next_state, np.ndarray)
        experience = state, action, reward, next_state, terminal
        self.memory.append(experience)
        self.num_exp += 1

    def size(self):
        return self.buffer_size

    def __len__(self):
        return self.num_exp

    def sample(self, batch_size=BATCH_SIZE):
        if self.num_exp < batch_size:
            batch = random.sample(self.memory, self.num_exp)
        else:
            batch = random.sample(self.memory, batch_size)

        state, action, reward, next_state, terminal = map(np.stack, zip(*batch))

        return state, action, reward, next_state, terminal

    def clear(self):
        self.memory = deque(maxlen=self.buffer_size)
        self.num_exp = 0
