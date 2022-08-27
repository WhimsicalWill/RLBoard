import gym
import sys
import getopt
import numpy as np
import random
from agent_class import Agent
from utils import plot_learning_curve, render_games, render_games_mujoco
from wrappers import HotStarts

def train(env_name):
	save_dir = 'data'
	curiosity_horizon = 3
	env = gym.make(env_name)
	env = HotStarts(env, save_dir, curiosity_horizon)
	agent = Agent(0.0003, 0.0003, env.observation_space.shape,
					0.005, env, env.action_space.shape[0], curiosity_horizon) 
	total_steps = 3e5
	best_score = env.reward_range[0] # init to smallest possible reward
	scores = []
	steps, episodes = 0, 0
	while steps < total_steps:
		done = False
		score = 0
		episodes += 1
		agent.episode_memory.clear()
		if env.contains_hot_starts() and random.random() < 0.5:
			print("Starting from hot start")
			observation = env.use_hot_start() # sample hot starts uniformly for starting state
			print(f"MAIN: {observation[:2]}")
		else:
			print("Starting from scratch")
			observation = env.reset()
		while not done:
			sim_state = env.get_sim_state()
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			agent.store_transition(observation, action, reward, observation_, done, sim_state)
			# agent.learn(observation_)
			agent.learn()
			score += reward
			steps += 1
			observation = observation_
		if episodes % 5 == 0:
			env.visualize_hot_starts(agent.choose_action)
		scores.append(score)
		avg_score = np.mean(scores[-100:])

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()
		print(f"Episode {episodes}, score: {score}, avg_score: {avg_score}")
	
	env.close()
	filename = f'{env_name}_{episodes}_games'
	figure_file = f'plots/{filename}.png'
	plot_learning_curve(scores, figure_file)

if __name__ == '__main__':
	arg_env_name = 'Hopper-v2'
	arg_render = False
	arg_help = f"{sys.argv[0]} -e <env_name> | use -r to render games from saved policy"

	try:
		opts, args = getopt.getopt(sys.argv[1:], "hre:", ["help", "render", "env_name="])
	except:
		print(arg_help)
		sys.exit(2)

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(arg_help)
			sys.exit(2)
		elif opt in ("-e", "--env_name"):
			arg_env_name = arg
		elif opt in ("-r", "--render"):
			arg_render = True
	
	if arg_render:
		render_games_mujoco(arg_env_name)
	else:
		train(arg_env_name)