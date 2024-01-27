from __future__ import print_function
from enum import Enum
import torch
from torch import nn
from models.ViT_CCT import StochasticClassifier, ViT_CCT, BN_3D, BN_Pixels
from utils import dino_utils
from configs.Configurations import Configurations
import torch.nn.functional as F
from torch import Tensor as T


class Supervised_learning_with_prefixes(nn.Module):
    def __init__(self, configs: Configurations):
        super().__init__()
        
        self.configs = configs
        self.backbone = configs.get_the_model()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(configs.configs_model.attention_dropout_rate)
        self.prefixes = configs.prefixes_base_task
        
    def forward(self, samples):
        prefixes = self.prepare_prefixes(samples)
        res = self.backbone(samples, prefixes)
        return res
    
    def prepare_prefixes(self, samples):
        prefixes_list = []
        for _ in range(len(samples)):
            prefixes_list.append(self.configs.prefixes_base_task)
        prefixes = torch.stack(prefixes_list)
        prefixes = self.dropout(prefixes)
        prefixes = prefixes.permute([1, 2, 0, 3, 4, 5])
        return prefixes
    
    def reset_backbone(self):
        self.configs.reset_backbone()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.zero_grad()
    
    def parameters(self) -> list:
        params = [self.prefixes]
        return params
