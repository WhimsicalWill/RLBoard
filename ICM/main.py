import os
import torch.multiprocessing as mp
from parallel import ParallelEnv

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    mp.set_start_method('spawn')
    global_ep = mp.Value('i', 0)
    env_id = 'CartPole-v1'
    n_threads = 4
    n_actions = 2
    input_shape = 4
    env = ParallelEnv(env_id=env_id, num_threads=n_threads, n_actions=n_actions, 
                      global_idx=global_ep, input_shape=input_shape)