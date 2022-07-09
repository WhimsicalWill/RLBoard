import mujoco_py
import gym
import pickle

def save_best(env):
	print("new best")
	print(env.sim.get_state())
	with open('state_data.pkl', 'wb') as file:
		pickle.dump(env.sim.get_state(), file, pickle.HIGHEST_PROTOCOL)

def load_best():
	print("loading best")
	with open('state_data.pkl', 'rb') as file:
		state_data = pickle.load(file)
		print(state_data)

env = gym.make('HalfCheetah-v3')

max_episodes = 10
max_time_steps = 300

best_score = float('-inf')
for i in range(max_episodes):
	env.reset()
	# test saving env info
	# print(env.sim.get_state())
	score = 0
	for t in range(max_time_steps):
		render_img = env.render(
			mode="rgb_array",
			width=64,
			height=64,
		)
		# print(rgb_array_large.shape)
		new_state, reward, done, _ = env.step(env.action_space.sample())
		score += reward
		if done:
			break
	print(f"Episode {i}: score {score}")
	if score > best_score:
		best_score = score
		save_best(env)

load_best()


