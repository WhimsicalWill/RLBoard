import gym
import numpy as np
import pickle
import os
import math
from imageio import mimsave

class EnvState():
	def __init__(self, sim_state, first_obs):
		self.sim_state = sim_state
		self.first_obs = first_obs

class HotStarts(gym.Wrapper):
	def __init__(self, env, save_dir, max_size=9):
		super(HotStarts, self).__init__(env)
		self.env = env
		self.state_dir = f"{save_dir}/states"
		self.viz_dir = f"{save_dir}/viz"
		self.max_size = max_size
		self.hot_starts = []

		# check that max_size is a perfect square
		side_length = int(math.sqrt(max_size))
		assert(side_length ** 2 == max_size)

		self.visualizer = Visualizer(self.env, self.viz_dir, side_length)

	def track_state(self, obs_):
		if len(self.hot_starts) >= self.max_size:
			return
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

	def visualize_hot_starts(self, agent_policy):
		self.visualizer.reset_frame_collage()
		for i, hot_start in enumerate(self.hot_starts):
			self.env.reset() # reset needed since step count needs to be reset
			self.env.sim.set_state(hot_start.sim_state)
			self.visualizer.env_runner(hot_start.first_obs, agent_policy, i)
		self.visualizer.save_gif()
   
class Visualizer():
	def __init__(self, env, viz_dir, grid_block_width, max_steps=500, px=128, border_thickness=5):
		self.env = env
		self.viz_dir = viz_dir
		self.grid_block_width = grid_block_width
		self.max_steps = max_steps
		self.px = px
		self.border_thickness = border_thickness
		self.num_color_channels = 3

		# calculate how many additional pixels are due to the border
		self.size_offset = border_thickness + grid_block_width * border_thickness
		self.grid_px_width = grid_block_width * px + self.size_offset
		self.frame_collage = np.zeros((max_steps, self.grid_px_width, self.grid_px_width, self.num_color_channels))

	def reset_frame_collage(self):
		self.frame_collage = np.zeros((self.max_steps, self.grid_px_width, self.grid_px_width, self.num_color_channels))

	def env_runner(self, first_obs, agent_policy, hot_start_num):
		obs = first_obs
		done = False
		score, step = 0, 0
		frames = []
		while not done and step < self.max_steps:
			action = agent_policy(obs) # this call may vary by implementation
			obs_, reward, done, info = self.env.step(action)
			render_img = self.env.render(
				mode="rgb_array",
				width=self.px,
				height=self.px,
			)
			frames.append(render_img)
			self.add_to_gif_collage(render_img, step, hot_start_num)
			score += reward
			step += 1
			obs = obs_
	
	def add_to_gif_collage(self, img, step, hot_start_num):
		grid_y_block = hot_start_num // self.grid_block_width
		grid_x_block = hot_start_num % self.grid_block_width
		grid_y_start = grid_y_block * self.px + self.border_thickness * (grid_y_block + 1) 
		grid_x_start = grid_x_block * self.px + self.border_thickness * (grid_x_block + 1)
		self.frame_collage[step, grid_y_start:grid_y_start+self.px, grid_x_start:grid_x_start+self.px, :] = img

	def save_gif(self):
		print(f"Saving gif collage to {self.viz_dir}")
		filename = f"{self.viz_dir}/hot_start_collage.gif"
		mimsave(filename, self.frame_collage.astype(np.uint8))