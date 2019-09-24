import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pdb
from copy import deepcopy
import shutil
import os
import time
import argparse
import pickle

import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from simple_rl.agents.AgentClass import Agent
from simple_rl.agents.func_approx.ddpg.utils import compute_gradient_norm

## Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-4  # learning rate
UPDATE_EVERY = 1  # how often to update the network
NUM_EPISODES = 3500
NUM_STEPS = 10000

class EpsilonSchedule:
    def __init__(self, eps_start, eps_end, eps_exp_decay, eps_linear_decay_length):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_exp_decay = eps_exp_decay
        self.eps_linear_decay_length = eps_linear_decay_length
        self.eps_linear_decay = (eps_start - eps_end) / eps_linear_decay_length

    def update_epsilon(self, current_epsilon, num_executions):
        pass

class GlobalEpsilonSchedule(EpsilonSchedule):
    def __init__(self, eps_start):
        EPS_END = 0.05
        EPS_EXPONENTIAL_DECAY = 0.999
        EPS_LINEAR_DECAY_LENGTH = 50000
        super(GlobalEpsilonSchedule, self).__init__(eps_start, EPS_END, EPS_EXPONENTIAL_DECAY, EPS_LINEAR_DECAY_LENGTH)

    def update_epsilon(self, current_epsilon, num_executions):
        if num_executions < self.eps_linear_decay_length:
            return current_epsilon - self.eps_linear_decay
        if num_executions == self.eps_linear_decay_length:
            print("Global Epsilon schedule switching to exponential decay")
        return max(self.eps_end, self.eps_exp_decay * current_epsilon)

class OptionEpsilonSchedule(EpsilonSchedule):
    def __init__(self, eps_start):
        EPS_END = 0.05
        EPS_EXPONENTIAL_DECAY = 0.999
        EPS_LINEAR_DECAY_LENGTH = 10000
        super(OptionEpsilonSchedule, self).__init__(eps_start, EPS_END, EPS_EXPONENTIAL_DECAY, EPS_LINEAR_DECAY_LENGTH)

    def update_epsilon(self, current_epsilon, num_executions):
        return max(self.eps_end, self.eps_exp_decay * current_epsilon)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """
        Set up the layers of the DQN
        Args:
            state_size (int): number of states in the state variable (can be continuous)
            action_size (int): number of actions in the discrete action domain
            seed (int): random seed
            fc1_units (int): size of the hidden layer
            fc2_units (int): size of the hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        DQN forward pass
        Args:
            state (torch.tensor): convert env.state into a tensor

        Returns:
            logits (torch.tensor): score for each possible action (1, num_actions)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def initialize_with_bigger_network(self, bigger_net):
        """
        { Used only when learning options in environments with discrete action spaces }.
        Given a policy network or target network from the global DQN, initialize the corresponding option DQN
        Args:
            bigger_net (QNetwork): Outputs q values over larger number of actions
        """
        for local_param, global_param in zip(self.fc1.parameters(), bigger_net.fc1.parameters()):
            local_param.data.copy_(global_param)
        for local_param, global_param in zip(self.fc2.parameters(), bigger_net.fc2.parameters()):
            local_param.data.copy_(global_param)

        num_original_actions = 4 # TODO: Assuming that we are in pinball domain
        self.fc3.weight.data.copy_(bigger_net.fc3.weight[:num_original_actions, :])
        self.fc3.bias.data.copy_(bigger_net.fc3.bias[:num_original_actions])

        assert self.fc3.out_features == 4, "Expected LunarLander with 4 actions, not {} ".format(self.fc3.out_features)

    def initialize_with_smaller_network(self, smaller_net, init_q_value):
        """
        Given a DQN over K actions, create a DQN over K + 1 actions. This is needed when we augment the
        MDP with a new action in the form of a learned option.
        Args:
            smaller_net (QNetwork)
            init_q_value (float)
        """
        for my_param, source_param in zip(self.fc1.parameters(), smaller_net.fc1.parameters()):
            my_param.data.copy_(source_param)
        for my_param, source_param in zip(self.fc2.parameters(), smaller_net.fc2.parameters()):
            my_param.data.copy_(source_param)

        smaller_num_labels = smaller_net.fc3.out_features
        self.fc3.weight[:smaller_num_labels, :].data.copy_(smaller_net.fc3.weight)
        self.fc3.bias[:smaller_num_labels].data.copy_(smaller_net.fc3.bias)

        new_action_idx = self.fc3.out_features - 1

        # Old way of initializing the weights and biases of the new option node:
        # self.fc3.weight[new_action_idx].data.copy_(torch.max(smaller_net.fc3.weight, dim=0)[0])
        # self.fc3.bias[new_action_idx].data.copy_(torch.max(smaller_net.fc3.bias, dim=0)[0])
        self.fc3.bias[new_action_idx].data.fill_(init_q_value)

class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, trained_options, seed, device, name="DQN-Agent",
                 eps_start=1., tensor_log=False, lr=LR, use_double_dqn=False, gamma=GAMMA, loss_function="huber",
                 gradient_clip=None, evaluation_epsilon=0.05, writer=None):
        self.state_size = state_size
        self.action_size = action_size
        self.trained_options = trained_options
        self.learning_rate = lr
        self.use_ddqn = use_double_dqn
        self.gamma = gamma
        self.loss_function = loss_function
        self.gradient_clip = gradient_clip
        self.evaluation_epsilon = evaluation_epsilon
        self.seed = random.seed(seed)
        self.tensor_log = tensor_log
        self.device = device

        # Q-Network
        self.policy_network = QNetwork(state_size, action_size, seed).to(self.device)
        self.target_network = QNetwork(state_size, action_size, seed).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Replay memory
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Epsilon strategy
        self.epsilon_schedule = GlobalEpsilonSchedule(eps_start) if "global" in name.lower() else OptionEpsilonSchedule(eps_start)
        self.epsilon = eps_start
        self.num_executions = 0 # Number of times act() is called (used for eps-decay)

        # Debugging attributes
        self.num_updates = 0
        self.num_epsilon_updates = 0

        if self.tensor_log:
            self.writer = SummaryWriter() if writer is None else writer

        print("\nCreating {} with lr={} and ddqn={} and buffer_sz={}\n".format(name, self.learning_rate,
                                                                               self.use_ddqn, BUFFER_SIZE))

        Agent.__init__(self, name, range(action_size), GAMMA)

    def set_new_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate

    def reduce_learning_rate(self):
        new_learning_rate = self.learning_rate / 100.
        self.set_new_learning_rate(new_learning_rate)

    def get_impossible_option_idx(self, state):

        # Arg-max only over actions that can be executed from the current state
        # -- In general an option can be executed from s if s is in its initiation set and NOT in its termination set
        # -- However, in the case of the goal option we just need to ensure that we are in its initiation set since
        # -- its termination set is terminal anyway and we are thus not in the risk of executing og from its
        # -- termination set.

        impossible_option_idx = []
        for idx, option in enumerate(self.trained_options):
            np_state = state.cpu().data.numpy()[0] if not isinstance(state, np.ndarray) else state

            if option.parent is None:
                assert option.name == "overall_goal_policy" or option.name == "global_option"
                impossible = not option.is_init_true(np_state)
            else:
                impossible = (not option.is_init_true(np_state)) or option.is_term_true(np_state)

            if impossible:
                impossible_option_idx.append(idx)

        return impossible_option_idx

    def act(self, state, train_mode=True):
        """
        Interface to the DQN agent: state can be output of env.step() and returned action can be input into next step().
        Args:
            state (np.array): numpy array state from Gym env
            train_mode (bool): if training, use the internal epsilon. If evaluating, set epsilon to min epsilon

        Returns:
            action (int): integer representing the action to take in the Gym env
        """
        self.num_executions += 1
        epsilon = self.epsilon if train_mode else self.evaluation_epsilon

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()

        impossible_option_idx = self.get_impossible_option_idx(state)

        for impossible_idx in impossible_option_idx:
            action_values[0][impossible_idx] = torch.min(action_values, dim=1)[0] - 1.

        action_values = action_values.cpu().data.numpy()
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values)

        all_option_idx = list(range(len(self.trained_options)))
        possible_option_idx = list(set(all_option_idx).difference(impossible_option_idx))
        randomly_chosen_option = random.choice(possible_option_idx)

        return randomly_chosen_option

    def get_best_actions_batched(self, states):
        q_values = self.get_batched_qvalues(states)
        return torch.argmax(q_values, dim=1)

    def get_value(self, state):
        action_values = self.get_qvalues(state)

        # Argmax only over actions that can be implemented from the current state
        impossible_option_idx = self.get_impossible_option_idx(state)
        for impossible_idx in impossible_option_idx:
            action_values[0][impossible_idx] = torch.min(action_values).item() - 1.

        return np.max(action_values.cpu().data.numpy())

    def get_qvalue(self, state, action_idx):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()
        return action_values[0][action_idx]

    def get_qvalues(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(state)
        self.policy_network.train()

        return action_values

    def get_batched_qvalues(self, states):
        """
        Q-values corresponding to `states` for all ** permissible ** actions/options given `states`.
        Args:
            states (torch.tensor) of shape (64 x 4)

        Returns:
            qvalues (torch.tensor) of shape (64 x |A|)
        """
        self.policy_network.eval()
        with torch.no_grad():
            action_values = self.policy_network(states)
        self.policy_network.train()

        if len(self.trained_options) > 0:
            # Move the states and action values to the cpu to allow numpy computations
            states = states.cpu().data.numpy()
            action_values = action_values.cpu().data.numpy()

            for idx, option in enumerate(self.trained_options): # type: Option
                try:
                    inits = option.batched_is_init_true(states)
                    terms = np.zeros(inits.shape) if option.parent is None else option.parent.batched_is_init_true(states)
                    action_values[(inits != 1) | (terms == 1), idx] = np.min(action_values) - 1.
                except:
                    pdb.set_trace()

            # Move the q-values back the GPU
            action_values = torch.from_numpy(action_values).float().to(self.device)

        return action_values

    def step(self, state, action, reward, next_state, done, num_steps):
        """
        Interface method to perform 1 step of learning/optimization during training.
        Args:
            state (np.array): state of the underlying gym env
            action (int)
            reward (float)
            next_state (np.array)
            done (bool): is_terminal
            num_steps (int): number of steps taken by the option to terminate
        """
        # Save experience in replay memory
        self.replay_buffer.add(state, action, reward, next_state, done, num_steps)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample()
                self._learn(experiences, GAMMA)
                if self.tensor_log:
                    self.writer.add_scalar("NumPositiveTransitions", self.replay_buffer.positive_transitions[-1], self.num_updates)
                self.num_updates += 1

    def _learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Args:
            experiences (tuple<torch.Tensor>): tuple of (s, a, r, s', done, tau) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, steps = experiences

        # Get max predicted Q values (for next states) from target model
        if self.use_ddqn:

            if len(self.trained_options) == 0:
                self.policy_network.eval()
                with torch.no_grad():
                    selected_actions = self.policy_network(next_states).argmax(dim=1).unsqueeze(1)
                self.policy_network.train()
            else:
                selected_actions = self.get_best_actions_batched(next_states).unsqueeze(1)

            Q_targets_next = self.target_network(next_states).detach().gather(1, selected_actions)
        else:
            Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            raise NotImplementedError("I have not fixed the Q(s',a') problem for vanilla DQN yet")

        # Options in SMDPs can take multiple steps to terminate, the Q-value needs to be discounted appropriately
        discount_factors = gamma ** steps

        # Compute Q targets for current states
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_targets = rewards + (discount_factors * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.policy_network(states).gather(1, actions)

        # Compute loss
        if self.loss_function == "huber":
            loss = F.smooth_l1_loss(Q_expected, Q_targets)
        elif self.loss_function == "mse":
            loss = F.mse_loss(Q_expected, Q_targets)
        else:
            raise NotImplementedError("{} loss function type not implemented".format(self.loss_function))

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()

        # # Gradient clipping: tried but the results looked worse -- needs more testing
        if self.gradient_clip is not None:
            for param in self.policy_network.parameters():
                param.grad.data.clamp_(-self.gradient_clip, self.gradient_clip)

        self.optimizer.step()

        if self.tensor_log:
            self.writer.add_scalar("DQN-Loss", loss.item(), self.num_updates)
            self.writer.add_scalar("DQN-AverageTargetQvalue", Q_targets.mean().item(), self.num_updates)
            self.writer.add_scalar("DQN-AverageQValue", Q_expected.mean().item(), self.num_updates)
            self.writer.add_scalar("DQN-GradientNorm", compute_gradient_norm(self.policy_network), self.num_updates)

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_network, self.target_network, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Args:
            local_model (nn.Module): weights will be copied from
            target_model (nn.Module): weights will be copied to
            tau (float): interpolation parameter - usually small eg 0.0001
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def update_epsilon(self):
        self.num_epsilon_updates += 1
        self.epsilon = self.epsilon_schedule.update_epsilon(self.epsilon, self.num_epsilon_updates)

        # Log epsilon decay
        if self.tensor_log:
            self.writer.add_scalar("DQN-Epsilon", self.epsilon, self.num_epsilon_updates)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """
        Initialize a ReplayBuffer object.
        Args:
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (torch.device): cpu / cuda:0 / cuda:1
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "num_steps"])
        self.seed = random.seed(seed)
        self.device = device

        self.positive_transitions = []

    def add(self, state, action, reward, next_state, done, num_steps):
        """
        Add new experience to memory.
        Args:
            state (np.array): We add numpy arrays from gym env to the buffer, but sampling from buffer returns tensor
            action (int)
            reward (float_
            next_state (np.array)
            done (bool)
            num_steps (int): number of steps taken by the action/option to terminate
        """
        e = self.experience(state, action, reward, next_state, done, num_steps)
        self.memory.append(e)

    def sample(self, batch_size=None):
        """Randomly sample a batch of experiences from memory."""
        size = self.batch_size if batch_size is None else batch_size
        experiences = random.sample(self.memory, k=size)

        # Log the number of times we see a non-negative reward (should be sparse)
        num_positive_transitions = sum([exp.reward >= 0 for exp in experiences])
        self.positive_transitions.append(num_positive_transitions)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        steps = torch.from_numpy(np.vstack([e.num_steps for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones, steps

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def train(agent, mdp, episodes, steps):
    per_episode_scores = []
    last_10_scores = deque(maxlen=10)
    iteration_counter = 0

    for episode in range(episodes):
        mdp.reset()
        state = deepcopy(mdp.init_state)
        score = 0.
        for step in range(steps):
            iteration_counter += 1
            action = agent.act(state.features(), agent.epsilon)
            reward, next_state = mdp.execute_agent_action(action)
            agent.step(state.features(), action, reward, next_state.features(), next_state.is_terminal(), num_steps=1)
            agent.update_epsilon()
            state = next_state
            score += reward
            if agent.tensor_log:
                agent.writer.add_scalar("Score", score, iteration_counter)
            if state.is_terminal():
                break
        last_10_scores.append(score)
        per_episode_scores.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)), end="")
        if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(last_10_scores)))
    return per_episode_scores

def test_forward_pass(dqn_agent, mdp):
    # load the weights from file
    mdp.reset()
    state = deepcopy(mdp.init_state)
    overall_reward = 0.
    mdp.render = True

    while not state.is_terminal():
        action = dqn_agent.act(state.features(), train_mode=False)
        reward, next_state = mdp.execute_agent_action(action)
        overall_reward += reward
        state = next_state

    mdp.render = False
    return overall_reward

def save_all_scores(experiment_name, log_dir, seed, scores):
    print("\rSaving training and validation scores..")
    training_scores_file_name = "{}_{}_training_scores.pkl".format(experiment_name, seed)

    if log_dir:
        training_scores_file_name = os.path.join(log_dir, training_scores_file_name)

    with open(training_scores_file_name, "wb+") as _f:
        pickle.dump(scores, _f)

def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--episodes", type=int, help="# episodes", default=NUM_EPISODES)
    parser.add_argument("--steps", type=int, help="# steps", default=NUM_STEPS)
    parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
    args = parser.parse_args()

    logdir = create_log_dir(args.experiment_name)
    learning_rate = 1e-4

    overall_mdp = GymMDP(env_name="LunarLander-v2", render=args.render)
    ddqn_agent = DQNAgent(state_size=overall_mdp.init_state.features().shape[0], action_size=len(overall_mdp.actions),
                          trained_options=[], seed=args.seed,
                          name="GlobalDDQN", lr=learning_rate, tensor_log=False, use_double_dqn=True)
    ddqn_episode_scores = train(ddqn_agent, overall_mdp, args.episodes, args.steps)
    save_all_scores(args.experiment_name, logdir, args.seed, ddqn_episode_scores)
