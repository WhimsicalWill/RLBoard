import numpy as np

# Simple memory class used for storing trajectory information
class Memory():
    def __init__(self):
        self.reset()

    def store_transition(self, reward, value, log_prob, state, new_state, action):
        self.value_memory.append(value)
        self.log_prob_memory.append(log_prob)
        self.reward_memory.append(reward)
        self.state_memory.append(state)
        self.new_state_memory.append(new_state)
        self.action_memory.append(action)
    
    def reset(self):
        self.value_memory = []
        self.log_prob_memory = []
        self.reward_memory = []
        self.state_memory = []
        self.new_state_memory = []
        self.action_memory = []

    # get the lists from memory
    def sample_memory(self): # sample n transitions from buffer
        return self.reward_memory, self.value_memory, self.log_prob_memory, \
                self.state_memory, self.new_state_memory, self.action_memory

