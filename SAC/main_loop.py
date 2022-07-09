import gym
import numpy as np
from agent_class import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('HalfCheetah-v3')
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, 
                    input_dims=env.observation_space.shape,
                    tau=0.005, batch_size=256, fc1_dims=256, fc2_dims=256, 
                    env=env, action_dim=env.action_space.shape[0])
    n_games = 250
    filename = f'InvertedPendulum_scale_{agent.scale}_{n_games}_games'
    figure_file = f'plots/{filename}.png'

    best_score = env.reward_range[0] # init to smallest possible reward
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    
    steps = 0 # for debug
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            if not load_checkpoint: # don't learn when viewing agent checkpoint
                agent.learn()
            score += reward
            steps += 1
            observation = observation_
        score_history.append(score) # add score to list after episode
        avg_score = np.mean(score_history[-100:]) # average the last 100 episodes

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        
        print(f"Episode {i}, score: {score}, avg_score: {avg_score}")

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
