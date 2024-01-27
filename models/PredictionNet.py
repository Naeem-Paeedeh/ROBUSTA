import torch
from torch import nn
from models.MLP_V import MLP_V

# @inproceedings{Xue2020OneShotIC,
#   title={One-Shot Image Classification by Learning to Restore Prototypes},
#   author={Wanqi Xue and Wei Wang},
#   booktitle={AAAI Conference on Artificial Intelligence},
#   year={2020}
# }


class PredictionNet(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 n_layers: int,
                 size_hidden_layer: int,
                 bias: bool,
                 dropout_rate: float,
                 use_real_residual_connections: bool,
                 use_stochastic_classifier: bool,
                 device,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.size_hidden_layer = size_hidden_layer
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.device = device
        self._is_ready = False
        self.use_real_residual_connections = use_real_residual_connections
        self.use_stochastic_classifier = use_stochastic_classifier

        self.mlp = MLP_V(size_first_layer=embed_dim,
                         size_hidden_layer=size_hidden_layer,
                         size_last_layer=embed_dim,
                         bias=bias,
                         use_stochastic_classifier=use_stochastic_classifier,
                         n_layers=self.n_layers,
                         dropout_rate=dropout_rate,
                         device=device
                         )

    def training_is_finished(self):
        self._is_ready = True

    def _is_ready(self):
        return self._is_ready
    
    def forward(self, prototype: torch.Tensor) -> torch.Tensor:
        if not self.training and not self._is_ready:
            return prototype
        
        calibrated = self.mlp(prototype)

        if self.use_real_residual_connections:
            return prototype + calibrated
        else:   # The paper's approach!
            if self.training:
                return calibrated
            else:
                res = 0.5 * (prototype + calibrated)   # Residual connection!
                return res
