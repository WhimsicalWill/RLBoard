import random

def collect_random_hot_starts(env, amount):
	step, max_steps = 0, 100
	steps_to_save = set(random.sample(range(max_steps), amount))
	
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