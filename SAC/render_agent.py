import gym
import numpy as np
from agent_class import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape,
                    tau=0.001, batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=env.action_space.shape[0])
    n_games = 10
    name = "RenderTest"
    filename = f'LunarLanderRender_alpha_{agent.alpha}_beta_{agent.beta}_{n_games}_games_{name}'
    figure_file = f'plots/{filename}.png'

    # Load saved model
    agent.load_models()

    best_score = env.reward_range[0] # init to smallest possible reward
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = observation_
        score_history.append(score) # add score to list after episode
        avg_score = np.mean(score_history[-100:]) # average the last 100 episodes

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f"Episode {i}, score: {score}, avg_score: {avg_score}")
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
