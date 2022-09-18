# RLBoard

Visualize qualitative behavior of RL agents during training.

# Details

Provides a gym wrapper which enables developers to visualize how an RL agent performs from arbitrary environment states. The wrapper saves (sim_state, first_obs) pairs, and can be configured to visualize "hot starts" from arbitrary parts of the environment.

Visualizations are made by creating gif collages of NxN different recordings of an agent playing from various starting states in the environment. The wrapper currently supports Mujoco environments, but extending to other environments is not too difficult. If a hot start visualization terminates early, the cell in the collage corresponding to this hot start becomes black.

![Example Visualization](./assets/hot_start_collage.gif)

# Installation

The easiest way to get started is to create a virtual environment and install the required dependencies with pip

```bash
  git clone https://github.com/WhimsicalWill/RLBoard.git
  cd RLBoard 
  pip install -r requirements.txt
```

# Usage

```bash
	env.track_state(observation)
	env.save_states()
	env.load_states()
	env.visualize_hot_starts(policy)
```

Contained in `hotstarts.py` is a wrapper for gym environments that allows for tracking of states. These states are stored in memory while the program is running, but can be saved/loaded with `env.save_states()` and `env.load_states()`.

To visualize an agent's performance from the collection of hot starts, call the wrapper's `visualize_hot_starts` function, passing in a function mapping from states to actions.

For example, to visualize a random agent:

```bash
	env.visualize_hot_starts(lambda obs: env.action_space.sample())
```

# Starter code

Before visualizing hot starts, they must be collected. `visualize_agent.py` provides an example snippet that collects a random set of hot starts and visualizes the performance of a trained RL agent starting from each hot start.