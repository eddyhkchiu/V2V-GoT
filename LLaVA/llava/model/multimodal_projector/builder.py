import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))

        # MY_DEBUG
        # only change the input size of mm_projector
        #modules = [nn.Linear(1024, config.hidden_size)]
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]

        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

# MY_CODE
# create another projector for scene level feature
def build_scene_vision_projector(config, delay_load=False, **kwargs):

    #mm_scene_projector_input_size = 1024
    #mm_scene_projector_input_size = 320
    #mm_scene_projector_input_size = 3072
    mm_scene_projector_input_size = config.mm_scene_projector_input_size
    print('mm_scene_projector_input_size: ', mm_scene_projector_input_size)
    #assert False

    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(mm_scene_projector_input_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))

        # MY_DEBUG
        # only change the input size of mm_projector
        #modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        modules = [nn.Linear(mm_scene_projector_input_size, config.hidden_size)]

        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
