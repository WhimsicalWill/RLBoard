import gym
import numpy as np
import wandb
from SAC.agent_class import Agent
from wrappers import HotStarts
# from utils import plot_learning_curve

if __name__ == '__main__':
	wandb.init('RLBoard', name='Ant-run-01')
	env_name = 'Ant-v3'
	save_dir = 'data'
	env = gym.make(env_name)
	env = HotStarts(env, save_dir)
	env.load_states()
	agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, 
					input_dims=env.observation_space.shape,
					tau=0.005, batch_size=256, fc1_dims=256, fc2_dims=256, 
					env=env, action_dim=env.action_space.shape[0])
	wandb.watch(agent.actor, log='all')
	wandb.watch(agent.critic_1, log='all')
	wandb.watch(agent.value, log='all')
 
	n_games = 250
	viz_checkpoints = 10
	viz_interval = n_games // 10

	best_score = env.reward_range[0] # init to smallest possible reward
	score_history = []
	load_checkpoint = False

	if load_checkpoint: # start training from a checkpoint
		agent.load_models()
	
	steps = 0
	for ep_num in range(n_games):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			agent.store_transition(observation, action, reward, observation_, done)
			agent.learn()
			score += reward
			steps += 1
			observation = observation_
		score_history.append(score) # add score to list after episode
		avg_score = np.mean(score_history[-100:]) # average the last 100 episodes

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()
		
		if ep_num % viz_interval == 0:
			print("Visualizing current agent's policy using hot starts")
			env.visualize_hot_starts(agent.choose_action)
		
		print(f"Episode {ep_num}, score: {score}, avg_score: {avg_score}")
		wandb.log({"score": score, "avg_score": avg_score, "episode": ep_num})
