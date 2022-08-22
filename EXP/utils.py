import numpy as np
import gym
import time
import matplotlib.pyplot as plt
import agent_class

def plot_learning_curve(scores, figure_file):
	x = [i+1 for i in range(len(scores))]
	running_avg = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
	plt.plot(x, running_avg)
	plt.title('Running average of previous 100 scores')
	plt.savefig(figure_file)

class ReplayBuffer():
	def __init__(self, max_size, input_shape, n_actions):
		self.mem_size = max_size
		self.mem_ctr = 0
		self.input_shape = input_shape
		self.n_actions = n_actions
		self.reset()
	
	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_ctr % self.mem_size # wrap around the buffer if ctr gets big enough
		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.new_state_memory[index] = state_
		self.terminal_memory[index] = done

		self.mem_ctr += 1
	
	def reset(self):
		self.state_memory = np.zeros((self.mem_size, *self.input_shape))
		self.new_state_memory = np.zeros((self.mem_size, *self.input_shape))
		self.action_memory = np.zeros((self.mem_size, self.n_actions))
		self.reward_memory = np.zeros(self.mem_size)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	def sample_buffer(self, batch_size): # sample n transitions from buffer
		max_mem = min(self.mem_ctr, self.mem_size) # constrain range to current existing samples

		batch = np.random.choice(max_mem, batch_size) # get indices for samples

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		states_ = self.new_state_memory[batch]
		done = self.terminal_memory[batch]

		return states, actions, rewards, states_, done

def render_games(env_name):
	env = gym.make(env_name)
	agent = agent_class.Agent(alpha=0.0003, beta=0.0003, input_dims=env.observation_space.shape,
					tau=0.005, env=env, action_dim=env.action_space.shape[0]) 
	n_games = 10

	# Load saved model
	agent.load_models()

	for i in range(n_games):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			env.render(mode="human")
			time.sleep(0.01)
			score += reward
			observation = observation_
		print(f"Episode {i}, score: {score}")
	env.close()