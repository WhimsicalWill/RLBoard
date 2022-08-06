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

# ICM Notes

- Phi is an encoder that embeds states into a latent space representation. phi: S -> L
- Inverse model takes in embeddings of S_(t-1) and S_t and tries to predict the action that was taken in order to make this transition.
- Inverse model has a hidden layer of size `fc_size` and then outputs logits for each action in the action space.
- The forward model takes a (state, action) pair as input, and tries to predict the embedding of the next state.

## Intuition about ICM

There are two components of the ICM loss, namely the inverse_loss and the forward_loss.

For the inverse_loss:

1. First, the model obtains the action prediction from the inverse model which takes two state representations as input.
2. Then, the loss is measured as the distance between the predicted action distribution and the actual action distribution.

The intuition for this loss is that by predicting actions, our embedding space (phi) will capture features relevant to predicting actions. This eliminates the "noisy TV" problem.

For the forward_loss:

1. Feed a state embedding and action to a neural network that predicts the resultant state embedding.
2. Then take the MSE loss between the predicted embedding and the actual embedding.

Since our encoder phi has features that obtain information relevant to predicting the action, we combine an embedding of a state and an action, and feed this to the forward model.

If our embedding represented the entire state space, it may encode features irrelevant to the action we took, e.g. "leaves blowing in the wind". Constraining it to action-related features allows us to focus on the consequences of our actions in the world.

In summary:

The combined loss (inverse_loss + forward_loss) essentially says: predict some aspect of the resultant state embedding as well as possible, under the constraint that the embedding should contain feaatures relevant to predicting the action.

If not for this constraint, many trivial solutions would be possible, such as an encoder that maps all states to zero.


