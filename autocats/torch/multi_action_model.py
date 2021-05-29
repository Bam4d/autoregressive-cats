from typing import List

from griddly.util.rllib.torch.agents.common import layer_init
from griddly.util.rllib.torch.agents.impala_cnn import ImpalaCNNAgent
from gym.spaces import MultiDiscrete, Discrete, Tuple
from ray.rllib import SampleBatch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn, TensorType
import torch

import numpy as np


class MultiActionAutoregressiveModel(TorchModelV2, nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        custom_model_config = model_config['custom_model_config']
        self._observation_features_module_class = custom_model_config.get('observation_features_class', ImpalaCNNAgent)
        self._observation_features_size = custom_model_config.get('observation_features_size', 256)

        assert isinstance(action_space, Tuple), 'action space is not a tuple. make sure to use the MultiActionEnv wrapper.'
        single_action_space = action_space[0]

        self._action_space_parts = None

        if isinstance(single_action_space, Discrete):
            self._num_action_logits = single_action_space.n
            self._action_space_parts = [self._num_action_logits]
        elif isinstance(single_action_space, MultiDiscrete):
            self._num_action_logits = np.sum(single_action_space.nvec)
            self._action_space_parts = [*single_action_space.nvec]
        else:
            raise RuntimeError('Can only be used with discrete and multi-discrete action spaces')

        #assert self._observation_features_size == num_outputs, 'bruh.'

        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Create the observation features network (By default use IMPALA CNN)
        self._observation_features_module = self._observation_features_module_class(
            obs_space,
            action_space,
            self._observation_features_size,
            model_config,
            name
        )

        # Action embedding network
        self._action_embedding_module = nn.Sequential(
            nn.Linear(self._num_action_logits, 256),
            nn.ReLU(),
            nn.Linear(256, self._observation_features_size),
            nn.ReLU(),
        )

        # Actor head
        self._action_module = nn.Sequential(
            nn.Linear(self._observation_features_size*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            layer_init(nn.Linear(128, self._num_action_logits), std=0.01)
        )


    def embed_action(self, action):
        # One-hot encode the action selection
        batch_size = action.shape[0]
        one_hot_actions = torch.zeros([batch_size, self._num_action_logits]).to(action.device)
        offset = 0
        for i, num_logits in enumerate(self._action_space_parts):
            oh_idxs = (offset+action[:, i]).type(torch.LongTensor)
            one_hot_actions[torch.arange(batch_size), oh_idxs] = 1
            offset+=num_logits

        return self._action_embedding_module(one_hot_actions)

    def observation_features_module(self, input_dict, state, seq_lens):
        observation_features, state_out = self._observation_features_module(input_dict, state, seq_lens)
        self._value = self._observation_features_module.value_function()

        return observation_features, state_out

    def action_module(self, observation_features, embedded_action=None):
        if embedded_action is None:
            embedded_action = torch.zeros_like(observation_features).to(observation_features.device)
        action_module_input = torch.cat([observation_features, embedded_action], dim=1)
        return self._action_module(action_module_input)

    def from_batch(self, train_batch, is_training=True):

        input_dict = train_batch.copy()
        input_dict["is_training"] = is_training
        states = []
        i = 0
        while "state_in_{}".format(i) in input_dict:
            states.append(input_dict["state_in_{}".format(i)])
            i += 1
        return self.observation_features_module(input_dict, states, input_dict.get("seq_lens"))

    def forward(self, input_dict, state, seq_lens):
        # Just do the state embedding here, actions are decoded as part of the distribution
        raise NotImplementedError

    def value_function(self):
        """
        This is V(s) depending on whatever the last state was.
        """
        return self._value