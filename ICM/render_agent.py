import gym
import numpy as np
from agent_class import AgentProcess
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_actions = 2
    input_shape = 4

    agent = AgentProcess(input_shape, n_actions) 
    
    n_games = 10
    name = "RenderTest"
    filename = f'CartPoleRender_{n_games}_games_{name}'
    figure_file = f'plots/{filename}.png'

    # Load saved model
    agent.load_models()

    best_score = env.reward_range[0] # init to smallest possible reward
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, _, _ = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = observation_
        score_history.append(score) # add score to list after episode
        avg_score = np.mean(score_history[-100:]) # average the last 100 episodes
        
        print(f"Episode {i+1}, score: {score}, avg_score: {avg_score}")
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
