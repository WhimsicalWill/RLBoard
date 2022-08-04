# Soft Actor Critic with Hot Starts

Proof of concept of Hot Starts on continuous control task with SAC.

As the model trains, the critic will curate the replay buffer of Hot Starts to exclusively train on after a given random training period. From then on, the actor will train (with probability 1 - eps) on a state sampled from the buffer of hot starts. With small probability (eps), the actor will train from a state sampled uniformly from the starting distribution. This is to encourage diversity and to enable the critic to identify new regions of state space that may benefit the RL agent.

# Initial plan

- Implement SAC on continuous control task
- Add auxiliary loss for curiosity
- Curate the Hot Starts with respect to this curiosity metric
- Maintain a buffer of the top n hot starts measured with respect to curiosity
- Compare the trained model to models trained w/o Hot Starts

# TODO:

- Add ICM loss to agent learn function
- Review ICM networks and tune to problem
- Modify gym wrapper to maintain a PQ of top n priority states