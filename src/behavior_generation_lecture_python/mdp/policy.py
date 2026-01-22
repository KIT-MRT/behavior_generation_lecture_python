"""This module contains the CategoricalPolicy implementation."""

from typing import Any, List, Optional, Type

import torch
from torch import nn
from torch.distributions.categorical import Categorical


def multi_layer_perceptron(
    sizes: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Type[nn.Module] = nn.Identity,
):
    """Returns a multi-layer perceptron"""
    mlp = nn.Sequential()
    for i in range(len(sizes) - 1):
        mlp.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            mlp.append(activation())
        else:
            mlp.append(output_activation())
    return mlp


class CategoricalPolicy:
    """A categorical policy parameterized by a neural network."""

    def __init__(
        self, sizes: List[int], actions: List[Any], seed: Optional[int] = None
    ) -> None:
        """Initialize the categorical policy.

        Args:
            sizes: List of layer sizes for the MLP.
            actions: List of available actions.
            seed: Random seed for reproducibility (default: None).
        """
        assert sizes[-1] == len(actions)
        if seed is not None:
            torch.manual_seed(seed)
        self.net = multi_layer_perceptron(sizes=sizes)
        self.actions = actions
        self._actions_tensor = torch.tensor(actions, dtype=torch.long).view(
            len(actions), -1
        )

    def _get_distribution(self, state: torch.Tensor) -> Categorical:
        """Calls the model and returns a categorical distribution over the actions.

        Args:
            state: The current state tensor.

        Returns:
            A categorical distribution over actions.
        """
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Any:
        """Returns an action sample for the given state.

        Args:
            state: The current state tensor.
            deterministic: If True, return the most likely action.

        Returns:
            The selected action.
        """
        policy = self._get_distribution(state)
        if deterministic:
            return self.actions[policy.mode.item()]
        return self.actions[policy.sample().item()]

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Returns the log-probability for taking the action, when being in the given state.

        Args:
            states: Batch of state tensors.
            actions: Batch of action tensors.

        Returns:
            Log-probabilities of the actions.
        """
        return self._get_distribution(states).log_prob(
            self._get_action_id_from_action(actions)
        )

    def _get_action_id_from_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Returns the indices of the passed actions in self.actions.

        Args:
            actions: Batch of action tensors.

        Returns:
            Tensor of action indices.
        """
        reshaped_actions = actions.unsqueeze(1).expand(
            -1, self._actions_tensor.size(0), -1
        )
        reshaped_actions_tensor = self._actions_tensor.unsqueeze(0).expand(
            actions.size(0), -1, -1
        )
        return torch.where(
            torch.all(reshaped_actions == reshaped_actions_tensor, dim=-1)
        )[1]
