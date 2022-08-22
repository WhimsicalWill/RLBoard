import gym
import numpy as np
import cv2
from collections import deque

class RepeatAction(gym.Wrapper):
    def __init__(self, env=None, n_repeat=4, fire_first=False):
        super(RepeatAction, self).__init__(env)
        self.n_repeat = n_repeat
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.n_repeat):
            obs_, reward, done, info = self.env.step(action)
            total_reward += reward
            if done: # episode might end before n_repeat actions completed
                break
        return obs_, total_reward, done, info

    def reset(self):
        obs = self.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meaning()[1] == 'FIRE'
            obs, _, _, _ = self.env_step(1)
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, new_shape):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1]) # reorder the axes to appease the pytorch gods
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape, dtype=np.float32) # define a simpler observation space

        def observation(self, obs):
            new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            new_frame = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
            new_obs = np.array(new_frame).reshape(self.shape)
            new_obs /= 255.0 # scale so pixels are in interval [0, 1]
            return new_obs

class StackFrames(gym.Wrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        # use np repeat function to specify a new shape for our observation space
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.low.repeat(repeat, axis=0),
                                                dtype=np.float32)
        self.frame_stack = deque(maxlen=repeat)

    def reset(self):
        self.frame_stack.clear()
        obs = self.env.reset()

        # initialize a full stack of the observation seen after resetting
        for _ in range(stack_size):
            self.frame_stack.append(observation)
        return np.array(self.frame_stack).reshape(self.observation_space.low.shape)
    
    def observation(self):
        self.frame_stack.append(observation)
        return np.array(self.frame_stack).reshape(self.observation_space.low.shape) # to be pedantic, reshape

def make_env(env_name, shape=(42, 42, 1), repeat=4):
    env = gym.make(env_name)
    env = RepeatAction(env, repeat)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    return env


