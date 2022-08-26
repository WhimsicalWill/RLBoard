# RLBoard

Visualize qualitative behavior of RL agents during training.

# Details

Provides a gym wrapper which enables developers to visualize how an RL agent performs from arbitrary environment states. The wrapper saves (sim_state, first_obs) pairs, and can be configured by users to visualize "hot starts" from arbitrary parts of the environment.

Currently, visualizations are made by creating gif collages of NxN different recordings of an agent playing from various starting states in the environment. The wrapper currently supports Mujoco environments, but extending to other environments is not too difficult.

# TODO:

- Include overlay on collage for checkpoint number for each cell
- Make sure all models and tensors on gpu (make faster)
- Reevaluate past hot_starts to recalculate their priorities
- Track the first n transitions from the starting state