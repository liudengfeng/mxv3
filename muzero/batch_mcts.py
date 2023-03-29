import torch

from .mcts import Node


class BatchMCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observations,
        legal_actions,
        to_plays,
        add_exploration_noise,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        n = len(to_plays)
        roots = [Node(0)] * n
        observations = torch.tensor(observations).to(next(model.parameters()).device)
        (
            root_predicted_values,
            rewards,
            policy_logits,
            hidden_states,
        ) = model.initial_inference(observations)

        for i, root in enumerate(roots):
            root.expand(
                legal_actions[i],
                to_plays[i],
                rewards[i],
                policy_logits[i],
                hidden_states[i],
                debug=True if self.config.debug_mcts else False,
            )

        if add_exploration_noise:
            for i in range(n):
                roots[i].add_exploration_noise(
                    dirichlet_alpha=self.config.root_dirichlet_alpha,
                    exploration_fraction=self.config.root_exploration_fraction,
                )

    def _f(self):
        pass
