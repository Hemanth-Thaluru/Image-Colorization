import torch
import torch.nn as nn

from .base_color import *

from .model_architecture import *

def colorize_model(pretrained=True):
    model = ModelStructure()
    if(pretrained):
        local_path = 'model.pth'
        state_dict = torch.load(local_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model
