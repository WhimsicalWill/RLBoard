inverse model drives the representation learning, encoding features that are related to actions
using these states that are encoded with action-related features, we attempt to predict new_states

constraining our feature learning to action-related features solves the "noisy TV problem" of
attending to regions of state space that are intrinsically uncertain. Ex: leaves blowing in wind

transitions that have been seen many times will be predicted reliably, but the result
of taking actions in certain states is uncertain, which is treated as the agent's "curiosity"

return new_states representations, predicted actions, and predicted new_states representations
return phi_new, pi_logits, phi_hat_new