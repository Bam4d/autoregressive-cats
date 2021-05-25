import torch
from ray.rllib.models import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, TorchMultiCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class MultiActionAutoregressiveDistribution(ActionDistribution):

    def _split_multi_actions(self, multi_actions):
        #self.model.num_action_parts

        return []

    def deterministic_sample(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, multi_actions):
        # Assume that the self.inputs here are logits from some already sampled distribution
        return TorchMultiCategorical(self.inputs).logp(multi_actions)

    def logp_and_entropy(self, multi_actions):
        # Assume here that the inputs are from model (i.e feature vector from observations)
        actions = self._split_multi_actions(multi_actions)

        embedded_action = None
        logp_list = []
        entropy_list = []
        for a in range(self.actions_per_step):
            if a != 0:
                embedded_action = self.model.embed_action(actions[a])

            logits = self.model.action_module(self.inputs, embedded_action)
            cat = TorchCategorical(logits)
            logp_list.append(cat.logp(actions[0]))
            entropy_list.append(cat.entropy())

        logp = torch.sum(torch.stack(logp_list))
        entropy = torch.sum(torch.stack(entropy_list))
        return logp, entropy

    def sampled_action_logp(self):
        raise NotImplementedError


    # def entropy(self):
    #     a1_dist = self._a1_distribution()
    #     a2_dist = self._a2_distribution(a1_dist.sample())
    #     return a1_dist.entropy() + a2_dist.entropy()
    #
    # def kl(self, other):
    #     a1_dist = self._a1_distribution()
    #     a1_terms = a1_dist.kl(other._a1_distribution())
    #
    #     a1 = a1_dist.sample()
    #     a2_terms = self._a2_distribution(a1).kl(other._a2_distribution(a1))
    #     return a1_terms + a2_terms

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 16  # controls model output feature vector size