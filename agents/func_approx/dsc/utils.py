# Python imports.
import pdb
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import time
import torch
import seaborn as sns
sns.set()
import imageio
from PIL import Image

# Other imports.
from simple_rl.tasks.point_maze.PointMazeStateClass import PointMazeState
from simple_rl.tasks.point_maze.PointMazeMDPClass import PointMazeMDP

class Experience(object):
	def __init__(self, s, a, r, s_prime):
		self.state = s
		self.action = a
		self.reward = r
		self.next_state = s_prime

	def serialize(self):
		return self.state, self.action, self.reward, self.next_state

	def __str__(self):
		return "s: {}, a: {}, r: {}, s': {}".format(self.state, self.action, self.reward, self.next_state)

	def __repr__(self):
		return str(self)

	def __eq__(self, other):
		return isinstance(other, Experience) and self.__dict__ == other.__dict__

	def __ne__(self, other):
		return not self == other

# ---------------
# Plotting utils
# ---------------

def plot_trajectory(trajectory, color='k', marker="o"):
	for i, state in enumerate(trajectory):
		if isinstance(state, PointMazeState):
			x = state.position[0]
			y = state.position[1]
		else:
			x = state[0]
			y = state[1]

		if marker == "x":
			plt.scatter(x, y, c=color, s=100, marker=marker)
		else:
			plt.scatter(x, y, c=color, alpha=float(i) / len(trajectory), marker=marker)

def plot_all_trajectories_in_initiation_data(initiation_data, marker="o"):
	possible_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
					   'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
	for i, trajectory in enumerate(initiation_data):
		color_idx = i % len(possible_colors)
		plot_trajectory(trajectory, color=possible_colors[color_idx], marker=marker)

def get_grid_states():
	ss = []
	for x in np.arange(0., 11., 1.):
		for y in np.arange(0., 11., 1.):
			s = PointMazeState(position=np.array([x, y]), velocity=np.array([0., 0.]),
							   theta=0., theta_dot=0., done=False)
			ss.append(s)
	return ss


def get_initiation_set_values(option):
	values = []
	for x in np.arange(-2., 11., 1.):
		for y in np.arange(-2., 18., 1.):
			s = PointMazeState(position=np.array([x, y]), velocity=np.array([0, 0]),
							   theta=0, theta_dot=0, done=False)
			values.append(option.is_init_true(s))

	return values


def get_values(solver, init_values=False):
	values = []
	for x in np.arange(0., 11., 1.):
		for y in np.arange(0., 11., 1.):
			v = []
			for vx in [-0.01, -0.1, 0., 0.01, 0.1]:
				for vy in [-0.01, -0.1, 0., 0.01, 0.1]:
					for theta in [-90., -45., 0., 45., 90.]:
						for theta_dot in [-1., 0., 1.]:
							s = PointMazeState(position=np.array([x, y]), velocity=np.array([vx, vy]),
											   theta=theta, theta_dot=theta_dot, done=False)

							if not init_values:
								v.append(solver.get_value(s.features()))
							else:
								v.append(solver.is_init_true(s))

			if not init_values:
				values.append(np.mean(v))
			else:
				values.append(np.sum(v))

	return values

def render_sampled_value_function(solver, episode=None, experiment_name=""):
	states = get_grid_states()
	values = get_values(solver)

	x = np.array([state.position[0] for state in states])
	y = np.array([state.position[1] for state in states])
	xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
	xx, yy = np.meshgrid(xi, yi)
	rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
	zz = rbf(xx, yy)
	plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower")
	plt.colorbar()
	name = solver.name if episode is None else solver.name + "_{}_{}".format(experiment_name, episode)
	plt.savefig("value_function_plots/{}/{}_value_function.png".format(experiment_name, name))
	plt.close()

def make_meshgrid(x, y, h=.02):
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	return xx, yy

def plot_one_class_initiation_classifier(option, episode=None, experiment_name=""):
	plt.figure(figsize=(8.0, 5.0))
	X = option.construct_feature_matrix(option.positive_examples)
	X0, X1 = X[:, 0], X[:, 1]
	xx, yy = make_meshgrid(X0, X1)
	Z1 = option.initiation_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z1 = Z1.reshape(xx.shape)
	plt.contour(xx, yy, Z1, levels=[0], linewidths=2, cmap=plt.cm.bone)

	plot_all_trajectories_in_initiation_data(option.positive_examples)

	plt.xlim((-3, 11))
	plt.ylim((-3, 11))

	plt.xlabel("x")
	plt.ylabel("y")
	name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
	plt.savefig("initiation_set_plots/{}/{}_{}_one_class_svm.png".format(experiment_name, name, option.seed))
	plt.close()

def visualize_dqn_replay_buffer(solver, experiment_name=""):
	goal_transitions = list(filter(lambda e: e[2] >= 0 and e[4] == 1, solver.replay_buffer.memory))
	cliff_transitions = list(filter(lambda e: e[2] < 0 and e[4] == 1, solver.replay_buffer.memory))
	non_terminals = list(filter(lambda e: e[4] == 0, solver.replay_buffer.memory))

	goal_x = [e[3][0] for e in goal_transitions]
	goal_y = [e[3][1] for e in goal_transitions]
	cliff_x = [e[3][0] for e in cliff_transitions]
	cliff_y = [e[3][1] for e in cliff_transitions]
	non_term_x = [e[3][0] for e in non_terminals]
	non_term_y = [e[3][1] for e in non_terminals]

	# background_image = imread("pinball_domain.png")

	plt.figure()
	plt.scatter(cliff_x, cliff_y, alpha=0.67, label="cliff")
	plt.scatter(non_term_x, non_term_y, alpha=0.2, label="non_terminal")
	plt.scatter(goal_x, goal_y, alpha=0.67, label="goal")
	# plt.imshow(background_image, zorder=0, alpha=0.5, extent=[0., 1., 1., 0.])

	plt.legend()
	plt.title("# transitions = {}".format(len(solver.replay_buffer)))
	plt.savefig("value_function_plots/{}/{}_replay_buffer_analysis.png".format(experiment_name, solver.name))
	plt.close()

def plot_two_class_classifier(option, episode, experiment_name):
	states = get_grid_states()
	values = get_initiation_set_values(option)

	x = np.array([state.position[0] for state in states])
	y = np.array([state.position[1] for state in states])
	xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
	xx, yy = np.meshgrid(xi, yi)
	rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
	zz = rbf(xx, yy)
	plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower",
			   alpha=0.6, cmap=plt.cm.bwr)
	#plt.colorbar()

	# Plot trajectories
	positive_examples = option.construct_feature_matrix(option.positive_examples)
	negative_examples = option.construct_feature_matrix(option.negative_examples)
	plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", cmap=plt.cm.coolwarm, alpha=0.3)

	if negative_examples.shape[0] > 0:
		plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", cmap=plt.cm.coolwarm,
					alpha=0.3)

	background_image = imageio.imread("point_maze_domain.png")
	plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

	plt.xticks([])
	plt.yticks([])

	name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
	plt.title("{} Initiation Set".format(option.name))
	plt.savefig("initiation_set_plots/{}/{}_initiation_classifier_{}.png".format(experiment_name, name, option.seed))
	plt.close()

def visualize_smdp_updates(global_solver, experiment_name=""):
	smdp_transitions = list(filter(lambda e: isinstance(e.action, int) and e.action > 0, global_solver.replay_buffer.memory))
	negative_transitions = list(filter(lambda e: e.done == 0, smdp_transitions))
	terminal_transitions = list(filter(lambda e: e.done == 1, smdp_transitions))
	positive_start_x = [e.state[0] for e in terminal_transitions]
	positive_start_y = [e.state[1] for e in terminal_transitions]
	positive_end_x = [e.next_state[0] for e in terminal_transitions]
	positive_end_y = [e.next_state[1] for e in terminal_transitions]
	negative_start_x = [e.state[0] for e in negative_transitions]
	negative_start_y = [e.state[1] for e in negative_transitions]
	negative_end_x = [e.next_state[0] for e in negative_transitions]
	negative_end_y = [e.next_state[1] for e in negative_transitions]

	plt.figure(figsize=(8, 5))
	plt.scatter(positive_start_x, positive_start_y, alpha=0.6, label="+s")
	plt.scatter(negative_start_x, negative_start_y, alpha=0.6, label="-s")
	plt.scatter(positive_end_x, positive_end_y, alpha=0.6, label="+s'")
	plt.scatter(negative_end_x, negative_end_y, alpha=0.6, label="-s'")
	plt.legend()
	plt.title("# updates = {}".format(len(smdp_transitions)))
	plt.savefig("value_function_plots/{}/DQN_SMDP_Updates.png".format(experiment_name))
	plt.close()

def visualize_next_state_reward_heat_map(solver, episode=None, experiment_name=""):
	next_states = [experience[3] for experience in solver.replay_buffer.memory]
	rewards = [experience[2] for experience in solver.replay_buffer.memory]
	x = np.array([state[0] for state in next_states])
	y = np.array([state[1] for state in next_states])

	plt.scatter(x, y, None, c=rewards, cmap=plt.cm.coolwarm)
	plt.colorbar()
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Replay Buffer Reward Heat Map")
	plt.xlim((-3, 11))
	plt.ylim((-3, 11))

	name = solver.name if episode is None else solver.name + "_{}_{}".format(experiment_name, episode)
	plt.savefig("value_function_plots/{}/{}_replay_buffer_reward_map.png".format(experiment_name, name))
	plt.close()


def replay_trajectory(trajectory, dir_name):
	colors = ["0 0 0 1", "0 0 1 1", "0 1 0 1", "1 0 0 1", "1 1 0 1", "1 1 1 1", "1 0 1 1", ""]

	for i, (option, state) in enumerate(trajectory):
		color = colors[option % len(colors)]
		mdp = PointMazeMDP(seed=0, render=True, color_str=color)

		qpos = np.copy(state.features()[:3])
		qvel = np.copy(state.features()[3:])

		mdp.env.wrapped_env.set_state(qpos, qvel)
		mdp.env.render()
		img_array = mdp.env.render(mode="rgb_array")
		img = Image.fromarray(img_array)
		img.save("{}/frame{}.png".format(dir_name, i))
