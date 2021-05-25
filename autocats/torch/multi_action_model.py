from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn



class MultiActionAutoregressiveImpalaCNNModel(TorchModelV2, nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):

        assert 'model_class' in model_config, '"model_class" must be set in model_config'

        self.actions_per_step = model_config.get('actions_per_step', 1)
        self.action_embedding_size = model_config.get('action_embedding_size', 32)

        action_model_action_space = action_space
        action_model_num_outputs = num_outputs // self._actions_per_step

        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)


        # Action embedding network
        self.embedding = nn.Sequential(

        )



    def forward(self, input_dict, state, seq_lens):
        # Just do the state embedding here, actions are decoded as part of the distribution


    def value_function(self):