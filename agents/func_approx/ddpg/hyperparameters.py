# Hyperparameters
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.01
LRA = 1e-4
LRC = 1e-3
HIDDEN_1 = 400
HIDDEN_2 = 300

MAX_EPISODES = 50000
MAX_STEPS = 200
GLOBAL_LINEAR_EPS_DECAY = 1e-5  # Decay over 100 thousand transitions
OPTION_LINEAR_EPS_DECAY = 2e-5  # Decay over  50 thousand transitions
PRINT_EVERY = 10
