import numpy  as np
import torch
import gym
from networks import ActorCritic
from agent_class import AgentProcess
from ICM import ICM
from memory import Memory
from utils import plot_learning_curve
    

# we differentiate between an actor_critic and an agent
# with the following paradigm: an AgentProcess has
# an actor_critic (NNs for function approx) and an intrinsic curiosity module (ICM)

def worker(name, input_shape, n_actions, global_ac, 
           optimizer, global_icm, icm_optimizer, env_id, global_idx):
    env = gym.make(env_id)
    local_agent = AgentProcess(input_shape, n_actions, global_ac, optimizer, global_icm, icm_optimizer)

    T_MAX = 20 # update interval
    t_steps = 0
    episode_num = 0
    max_steps = 1e5 # specify max_steps instead of max_episodes for more granular control
    scores = []
    while t_steps < max_steps:
        obs = env.reset()
        score, done, ep_steps = 0, False, 0 # reset episode variables
        episode_num += 1
        while not done:
            action, value, log_prob = local_agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            local_agent.store_transition(reward, value, log_prob, obs, obs_, action)
            score += reward
            ep_steps += 1
            t_steps += 1
            obs = obs_
            if ep_steps % T_MAX == 0 or done: # update networks if needed
                local_agent.learn(obs_, done) # pass in the newest obs for V estimation  
        
        if name == '1':
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print(f"Agent {name}: Episode {episode_num}, {score} score, {ep_steps} steps, {t_steps} cum_steps, avg score: {avg_score}")
    global_ac.save_checkpoint() # save the weights of the global actor_critic after training
    if name == '1': # plot learning curve for agent on first thread
        step_list = [x for x in range(episode_num)]
        plot_learning_curve(step_list, scores, 'A3C_cartpole_final.png')