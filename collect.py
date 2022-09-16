import gym
import random
import numpy as np
from hotstarts import HotStarts

def collect_random_hot_starts(env, amount):
	step, max_steps = 0, 100
	steps_to_save = set(random.sample(range(100), amount))
	
	while step < max_steps:
		env.reset()
		done = False
		while not done:
			action = env.action_space.sample()
			obs_, _, done, _ = env.step(action)
			if step in steps_to_save:
				env.track_state(obs_) # save state of env as a hot start
				steps_to_save.remove(step)
				if not steps_to_save: 
					env.save_states() # save the hot starts to disk
					return
			step += 1


# # viz_hot_starts_random

# import gym
# import numpy as np
# from wrappers import HotStarts

# if __name__ == '__main__':
# 	env_name = 'HalfCheetah-v3'
# 	save_dir = 'data'
# 	env = gym.make(env_name)
# 	env = HotStarts(env, save_dir)

# 	# random mapping from obs -> actions
# 	def random_action(observation):
# 		return env.action_space.sample()
		

# 	# load the hot starts from disk
# 	env.load_states()
# 	env.visualize_hot_starts(random_action)