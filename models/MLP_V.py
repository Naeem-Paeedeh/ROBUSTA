from torch import nn
from models.ViT_CCT import StochasticClassifier


class MLP_V(nn.Module):
    """MLP with arbitrary number of layers"""
    def __init__(self,
                 #  configs: Configurations,
                 size_first_layer: int,
                 size_hidden_layer: int,
                 size_last_layer: int,
                 n_layers: int,
                 device,
                 bias: bool,
                 dropout_rate: float,
                 use_stochastic_classifier: bool = False,
                 temperature_stochastic_classifier: float = 16):
        """An MLP with a  number of layers.

        Args:
            size_first_layer (int): The number of activations in the first layer
            size_hidden_layer (int): The number of neurons in hidden layers
            size_last_layer (int): The output dimension
            n_layers (int): The number of layers.
            device (_type_): CPU or GPU
            dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__()

        self.dropout_rate = dropout_rate
        device = device     # configs.device
        self.n_layers = n_layers
        self.use_stochastic_classifier = use_stochastic_classifier
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.activation = nn.ReLU()

        assert self.n_layers >= 1

        if self.n_layers > 1:
            self.input_layer = nn.Linear(size_first_layer, size_hidden_layer, bias=bias, device=device)

            self.hidden_layers = nn.ModuleList([
                nn.Linear(size_hidden_layer, size_hidden_layer, bias=bias, device=device) for _ in range(self.n_layers - 2)
            ])

            if use_stochastic_classifier:
                self.output_layer = StochasticClassifier(device, size_hidden_layer, size_last_layer, temperature_stochastic_classifier)  # configs.configs_arch.temperature_stochastic_classifier
            else:
                self.output_layer = nn.Linear(size_hidden_layer, size_last_layer, bias=bias, device=device)
        else:
            # When we want to define a linear layer
            if use_stochastic_classifier:
                self.input_layer = StochasticClassifier(device, size_first_layer, size_last_layer, temperature_stochastic_classifier)    # configs.configs_arch.temperature_stochastic_classifier
            else:
                self.input_layer = nn.Linear(size_first_layer, size_last_layer, bias=bias, device=device)

    def forward(self, x, stochatic: bool = True):
        if self.n_layers > 1:
            x = self.input_layer(x)

            for layer in self.hidden_layers:
                if self.dropout_rate > 0.0:
                    x = self.dropout(x)
                x = self.activation(x)
                x = layer(x)

            if self.dropout_rate > 0.0:
                x = self.dropout(x)
            x = self.activation(x)
            if self.use_stochastic_classifier:
                x = self.output_layer(x, stochatic)
            else:
                x = self.output_layer(x)
        else:
            if self.use_stochastic_classifier:
                x = self.input_layer(x, stochatic)
            else:
                x = self.input_layer(x)

        return x
