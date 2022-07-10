import gym
import numpy as np
import pickle
import os
from imageio import mimsave


class EnvState():
	def __init__(self, sim_state, first_obs, priority=0):
		self.sim_state = sim_state
		self.first_obs = first_obs
		self.priority = priority

class HotStarts(gym.Wrapper):
	def __init__(self, env, save_dir, max_size=10): # TODO: change to save_dir
		super(HotStarts, self).__init__(env)
		self.env = env
		self.state_dir = f"{save_dir}/states"
		self.viz_dir = f"{save_dir}/viz"
		self.max_size = max_size
		self.hot_starts = [] # to be populated by user with starting env states
  
		self.visualizer = Visualizer(self.env, self.viz_dir)

	def track_state(self, obs_):
		if len(self.hot_starts) >= self.max_size:
			return # TODO: prioritize by priority
		sim_state = self.env.sim.get_state()
		env_state = EnvState(sim_state, obs_)
		self.hot_starts.append(env_state)

	def load_states(self):
		for filename in os.listdir(self.state_dir):
			with open(f"{self.state_dir}/{filename}", 'rb') as file:
				env_state = pickle.load(file)
				self.hot_starts.append(env_state)
	
	def save_states(self):
		for i, env_state in enumerate(self.hot_starts):
			filename = f"{self.state_dir}/hot_start_data_{i}"
			with open(filename, 'wb') as file:
				pickle.dump(env_state, file, pickle.HIGHEST_PROTOCOL)

	def get_lowest_priority(self):
		pass

	# agent_policy is the agent's mapping from observations to actions
	def visualize_hot_starts(self, agent_policy):
		for i, hot_start in enumerate(self.hot_starts):
			self.env.reset() # reset needed since step count needs to be reset
			self.env.sim.set_state(hot_start.sim_state)
			first_obs = hot_start.first_obs
			self.visualizer.env_runner(first_obs, agent_policy, i)
   
class Visualizer():
	# TODO: put config details in config.json for easy user editing
	def __init__(self, env, viz_dir, max_steps=500, px=256):
		self.env = env
		self.viz_dir = viz_dir
		self.max_steps = max_steps
		self.px = px
		
	def env_runner(self, first_obs, agent_policy, save_name):
		obs = first_obs
		done = False
		score, steps = 0, 0
		frames = []
		while not done and steps < self.max_steps:
			action = agent_policy(obs) # this call may vary by implementation
			obs_, reward, done, info = self.env.step(action)
			render_img = self.env.render(
				mode="rgb_array",
				width=self.px,
				height=self.px,
			)
			frames.append(render_img)
			score += reward
			steps += 1
			obs = obs_
		print(f"Checkpoint {save_name}, score: {score}, steps: {steps}")
		self.save_vid(frames, save_name)
	
	def save_vid(self, frames, save_name):
		print(f"Saving gif {save_name} to {self.viz_dir}")
		mimsave(f"{self.viz_dir}/hot_start_{save_name}.gif", frames)

# TODO2:
# In the future, we could have env states saved in a dictionary, to make deleting by key easy
# Transition the functionality towards in memory, and saving/loading whole directories
# 
#
