# We used some codes from TVT, CDTrans, and STARTUP papers, and the
# timm (https://github.com/rwightman/pytorch-image-models), https://github.com/lucidrains/vit-pytorch repositories,
# and https://github.com/asrafulashiq/dynamic-cdfsl.

# The implementations of both ViT and CCT are merged.

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import Tensor as T
import torch.nn as nn
from torch.nn import Dropout, Linear
import einops
from utils.pos_embed import get_2d_sincos_pos_embed
from configs.configs_model import ConfigurationModel
import utils.dino_utils as dino_utils
from enum import Enum
from utils.shared import find_patterns_sequencially


class PositionalEmbeddingType(Enum):
    Learnable = 1
    SinCos = 2


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    @staticmethod
    def drop_path(x, drop_prob: float = 0., training: bool = False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# @misc{website,
# 	author	={Yin, Zeyuan},
# 	title	={zeyuanyin (Zeyuan)},
# 	year	={2023},
# 	url	={https://github.com/zeyuanyin/BN-ViT}
# 	}
class BN_3D(nn.BatchNorm1d):    # It uses the embed_dim for BN instead of n_channels
    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.shape
        x = x.reshape(-1, embed_dim)
        x = super().forward(x)   # Apply BatchNorm1d
        x = x.reshape(batch_size, sequence_length, embed_dim)
        return x
    

class BN_Pixels(nn.BatchNorm1d):    # It uses the embed_dim for BN instead of n_channels
    def forward(self, x):
        n_samples, num_channels, image_size, _ = x.shape
        if n_samples == 1:
            return x
        x = x.reshape(n_samples, -1)
        x = super().forward(x)   # Apply BatchNorm1d
        x = x.reshape(n_samples, num_channels, image_size, image_size)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self,
                 configs_model: ConfigurationModel, 
                 use_proj_dino: bool,
                 use_BatchNorm_for_patch_embeddings: bool,
                 use_BatchNorm_for_patch_embeddings_for_local_patches: bool,
                 image_size: int,
                 local_patch_size: int,
                 in_channels: int,
                 in_planes: int = 64):  # in_channels=3
        super().__init__()
        self.configs_model = configs_model
        self.use_proj_dino = use_proj_dino
        self.image_size = image_size
        self.local_patch_size = local_patch_size
        self.in_channels = in_channels
        self.in_planes = in_planes
        self.num_patches = None
        self.proj = None
        self.proj_dino = None

        if configs_model.net_type.is_cct():
            my_list = self.create_cct_proj(self.image_size, use_BatchNorm=use_BatchNorm_for_patch_embeddings)        # self.configs_model.use_BatchNorm
            self.proj_image_size = nn.Sequential(*my_list)

            if use_proj_dino:
                my_list = self.create_cct_proj(local_patch_size, use_BatchNorm=use_BatchNorm_for_patch_embeddings_for_local_patches)       # self.configs_model.use_BatchNorm
                self.proj_dino = nn.Sequential(*my_list)
        else:
            self.proj = nn.Conv2d(in_channels=in_channels,
                                  out_channels=configs_model.embed_dim,
                                  kernel_size=(configs_model.patch_size, configs_model.patch_size),
                                  stride=configs_model.stride,
                                  padding=configs_model.padding)

        # For both CCT and ViT
        self.num_patches = self.forward(torch.zeros((1, in_channels, self.image_size, self.image_size))).shape[1]
        self.apply(self.init_weight)

    def create_cct_proj(self, image_size, use_BatchNorm: bool):
        my_list = []

        image_size_temp = image_size

        n_filter_list = [self.in_channels] + \
                        [self.in_planes for _ in range(self.configs_model.n_conv_layers - 1)] + \
                        [self.configs_model.embed_dim]

        for i in range(self.configs_model.n_conv_layers):
            if use_BatchNorm:
                my_list.append(BN_Pixels(n_filter_list[i] * image_size_temp * image_size_temp))

            my_list.append(nn.Conv2d(in_channels=n_filter_list[i],
                                     out_channels=n_filter_list[i + 1],
                                     kernel_size=(self.configs_model.patch_size),
                                     stride=self.configs_model.stride,
                                     padding=self.configs_model.padding,
                                     bias=False))
            
            if use_BatchNorm:
                # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                image_size_temp = int(np.floor(((image_size_temp + 2 * self.configs_model.padding[0] - (self.configs_model.patch_size - 1) - 1)) / self.configs_model.stride[0] + 1))

                my_list.append(BN_Pixels(n_filter_list[i + 1] * image_size_temp * image_size_temp))

            my_list.append(nn.ReLU())

            my_list.append(nn.MaxPool2d(kernel_size=self.configs_model.pooling_kernel_size,
                                        stride=self.configs_model.pooling_stride,
                                        padding=self.configs_model.pooling_padding))
            
            if use_BatchNorm:
                # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                image_size_temp = int(np.floor(((image_size_temp + 2 * self.configs_model.pooling_padding - (self.configs_model.pooling_kernel_size - 1) - 1)) / self.configs_model.pooling_stride + 1))
        
        return my_list

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x: T) -> T:
        # x.shape is (n_samples, num_channels, image_size, image_size)
        _, _, image_size, _ = x.shape
        if image_size == self.image_size:
            x = self.proj_image_size(x)  # -> batch_size x embed_dim x n_patches_dim_1 x n_patches_dim_2
        elif image_size == self.local_patch_size:
            x = self.proj_dino(x)
        else:
            self.configs_model.logger.exception("Not Implemented!")
            raise NotImplementedError
        x = x.flatten(2)  # -> batch_size x embed_dim x n_patches
        x = x.transpose(-1, -2)  # -> batch_size x n_patches x embed_dim
        return x  # batch_size x n_patches x embed_dim
    

# @article{Yao2021LeveragingBN,
#   title={Leveraging Batch Normalization for Vision Transformers},
#   author={Zhuliang Yao and Yue Cao and Yutong Lin and Ze Liu and Zheng Zhang and Han Hu},
#   journal={2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
#   year={2021},
#   pages={413-422}
# }
class StochasticClassifier(nn.Module):
    def __init__(self, device, num_features: int, total_classes: int, temperature: float):
        super().__init__()
        self.device = device
        self.shape = total_classes, num_features
        self.mu = nn.Parameter(0.01 * torch.randn(*self.shape, requires_grad=True, device=self.device))
        self.sigma = nn.Parameter(torch.zeros(*self.shape, requires_grad=True, device=self.device))   # each rotation have individual variance here
        self.num_features = num_features
        self.total_classes = total_classes
        self.temperature = temperature

    def forward(self, x, stochastic=True):
        mu = self.mu
        sigma = self.sigma

        if stochastic:
            sigma = F.softplus(sigma - 4)                                   # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn(*self.shape, device=self.device) + mu
        else:
            weight = mu
        
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        score = F.linear(x, weight)
        score = score * self.temperature

        return score
    

class CosineClassifier(nn.Module):
    def __init__(self, device, num_features: int, total_classes: int, temperature: float):
        super().__init__()
        self.device = device
        self.weights = nn.Parameter(0.01 * torch.randn(total_classes, num_features, requires_grad=True, device=self.device))
        self.num_features = num_features
        self.total_classes = total_classes
        self.temperature = temperature

    def forward(self, x):
        weights = F.normalize(self.weights, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        score = x @ weights.T
        score = score * self.temperature

        return score


class MLP(nn.Module):
    def __init__(self, configs_model: ConfigurationModel):
        super().__init__()
        self.configs_model = configs_model

        self.fc1 = Linear(configs_model.embed_dim,
                          configs_model.transformer_mlp_dim)
        
        self.fc2 = Linear(configs_model.transformer_mlp_dim,
                          configs_model.embed_dim)

        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(configs_model.dropout_rate)
        self.bn1 = BN_3D(configs_model.transformer_mlp_dim) if configs_model.use_BatchNorm else None
        self.bn2 = BN_3D(configs_model.transformer_mlp_dim) if configs_model.use_BatchNorm else None
        self.bn3 = BN_3D(configs_model.embed_dim) if configs_model.use_BatchNorm else None

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, embed_dim)
        x = self.fc1(x)
        if self.configs_model.use_BatchNorm:
            x = self.bn1(x)
        x = self.act_fn(x)
        if self.configs_model.use_BatchNorm:
            x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.configs_model.use_BatchNorm:
            x = self.bn3(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, configs_model: ConfigurationModel):
        super().__init__()
        self.configs_model = configs_model
        self.num_heads = configs_model.transformer_num_heads
        self.embed_dim = configs_model.embed_dim
        self.head_dim = configs_model.head_dim

        self.scale = self.head_dim ** -0.5

        self.query = None
        self.key = None
        self.value = None
        self.query_cross = None
        self.key_cross = None
        self.value_cross = None

        self.query = nn.Linear(self.embed_dim, self.embed_dim, bias=configs_model.qkv_bias)
        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=configs_model.qkv_bias)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=configs_model.qkv_bias)

        self.attn_dropout = nn.Dropout(configs_model.attention_dropout_rate)

        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(configs_model.dropout_rate)
        )

        self.proj_cross = None

    def transpose_for_scores(self, x: T) -> T:
        # x.shape is (batch_size, seq_length, embed_dim) for ViT
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        # new shape is (batch_size, seq_length, self.num_heads, self.head_dim) for ViT
        x = x.view(*new_x_shape)
        # The shape of returned tensor is (batch_size, self.num_heads, seq_length, self.head_dim) for ViT
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, prefix: T = None) -> T:
        # x.shape is (batch_size, seq_length, embed_dim) for ViT
        num_samples, num_patches_and_cls, num_channels = x.shape

        q = self.query(x)   # q.shape after the linear layer is (batch_size, seq_length, embed_dim)
        k = self.key(x)     # k.shape after the linear layer is (batch_size, seq_length, embed_dim)
        v = self.value(x)   # v.shape after the linear layer is (batch_size, seq_length, embed_dim)

        # Their shapes after theese -> (batch_size, self.num_heads, seq_length, self.head_dim)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        if prefix is not None:
            # Their shape before concatenation -> (batch_size, self.num_heads, seq_length, self.head_dim)
            k = torch.cat([prefix[0], k], dim=2)
            v = torch.cat([prefix[1], v], dim=2)

        attn_prob = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)

        attn_prob = self.attn_dropout(attn_prob)

        x = (attn_prob @ v).transpose(1, 2).reshape(num_samples, num_patches_and_cls, num_channels)

        # Projection and dropout
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, configs_model: ConfigurationModel, drop_path_rate=0.1):
        super().__init__()
        self.config = configs_model
        self.embed_dim = configs_model.embed_dim
        self.norm_input = BN_3D(self.embed_dim) if configs_model.use_BatchNorm else nn.LayerNorm(self.embed_dim)
        self.attn = Attention(configs_model)
        self.drop_path = \
            DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm_mlp = BN_3D(self.embed_dim) if configs_model.use_BatchNorm else nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(configs_model)

    def forward(self, z: T, prefixes: T = None):
        # z.shape is (batch_size, sequence_length, embed_dim)
        z_norm = self.norm_input(z)
        attn = self.attn(z_norm, prefixes)
        # Residual connections, dropouts, norms, additions
        z_hat = z + self.drop_path(attn)
        z = z_hat + self.drop_path(self.mlp(self.norm_mlp(z_hat)))
        return z


# ViT or CCT for Domain Adaptation with Quadruple Transformer Blocks
class ViT_CCT(nn.Module):
    """
    These are the specifications of the vanilla ViT, CCT and quadruple transformer blocks for Adapter.
    """
    def __init__(self,
                 configs_model: ConfigurationModel,
                 use_proj_dino: bool,
                 use_BatchNorm_for_patch_embeddings: bool,
                 use_BatchNorm_for_patch_embeddings_for_local_patches: bool,
                 image_size: int,           # The image size may be different when we train the model with DINO
                 local_patch_size: int,     # For DINO
                 in_channels: int = 3,
                 pos_embedding_type: PositionalEmbeddingType = PositionalEmbeddingType.Learnable):
        super().__init__()

        self.configs_model = configs_model
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_embedding = PatchEmbedding(configs_model=configs_model,
                                              use_proj_dino=use_proj_dino,
                                              use_BatchNorm_for_patch_embeddings=use_BatchNorm_for_patch_embeddings,
                                              use_BatchNorm_for_patch_embeddings_for_local_patches=use_BatchNorm_for_patch_embeddings_for_local_patches,
                                              image_size=image_size,
                                              local_patch_size=local_patch_size,
                                              in_channels=in_channels
                                              )
        self.seq_length = self.patch_embedding.num_patches

        if configs_model.net_type.is_vit():
            self.cls_token = nn.Parameter(torch.zeros(1, 1, configs_model.embed_dim))
            self.seq_length += 1
        else:
            self.attention_pool = nn.Linear(configs_model.embed_dim, 1)

        self.dropout = nn.Dropout(configs_model.dropout_rate)
        self.num_layers = configs_model.num_layers

        # We follow the CCT implementation
        dpr = [x.item() for x in torch.linspace(0, configs_model.stochastic_depth, configs_model.num_layers)]

        self.blocks = nn.ModuleList([Block(configs_model, drop_path_rate=dpr[i])
                                     for i in range(configs_model.num_layers)])
        self.norm = BN_3D(configs_model.embed_dim) if configs_model.use_BatchNorm else nn.LayerNorm(configs_model.embed_dim)

        self.pos_embedding = None
        self.pos_embedding_type = pos_embedding_type

        self.initialize_positional_embeddings()

    def initialize_positional_embeddings(self):
        if self.pos_embedding_type is PositionalEmbeddingType.Learnable:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.seq_length, self.configs_model.embed_dim),
                requires_grad=True)
        elif self.pos_embedding_type is PositionalEmbeddingType.SinCos:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.seq_length, self.configs_model.embed_dim),
                requires_grad=False)

            # Positional embedding for the backbone
            pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1],
                                                int(self.patch_embedding.num_patches ** .5),
                                                cls_token=self.configs_model.net_type.is_vit())
            self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            msg = "Error: Unknown positional embedding type!"
            self.configs_model.logger.exception(msg)
            raise Exception(msg)

    def interpolate_pos_embedding(self, x, w, h):
        if x.shape[1] == self.pos_embedding.shape[1] and w == h:
            return self.pos_embedding

        dim = x.shape[-1]

        if self.configs_model.net_type.is_cct():
            N = self.pos_embedding.shape[1]
            patch_pos_embed = self.pos_embedding
            scale = math.sqrt(x.shape[1] / self.pos_embedding.shape[1])
        else:   # For ViT
            N = self.pos_embedding.shape[1] - 1
            class_pos_embed = self.pos_embedding[:, 0]
            patch_pos_embed = self.pos_embedding[:, 1:]
            scale = math.sqrt((x.shape[1] - 1) / (self.pos_embedding.shape[1] - 1))

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(scale, scale),
            mode='bicubic',
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if self.configs_model.net_type.is_cct():
            return patch_pos_embed
        else:   # For ViT
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: T, prefixes: torch.Tensor = None, prompts: torch.Tensor = None):     # Configs is a Configurations object
        # x.shape is (n_samples, num_channels, image_size, image_size)
        n_samples, _, w, h = x.shape
        assert w == h

        x = self.patch_embedding(x)     # x.shape = [n_samples, sequence_length, embed_dim]

        if self.configs_model.net_type.is_vit():
            # Replicating cls_token for all samples
            cls_tokens = einops.repeat(self.cls_token, '1 1 d -> n 1 d', n=n_samples)
            x = torch.cat((cls_tokens, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)

        # (n_samples, 1 + n_patches for ViT or sequence length, embed_dim)
        # add positional encoding to each token
        x += self.interpolate_pos_embedding(x, w, h)

        if prompts is not None:
            x = torch.cat([prompts, x], dim=1)

        x = self.dropout(x)

        for i, block in enumerate(self.blocks):
            prefix_for_this_layer = prefixes[i] if prefixes is not None and i < len(prefixes) else None
            x = block(x, prefix_for_this_layer)

        # x.shape = (batch_size, sequence_length, embed_dim)
        x = self.norm(x)

        if self.configs_model.net_type.is_cct():
            x = (F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2) @ x).squeeze(-2)
        else:
            x = x[:, 0]     # cls_token

        return x


class DINOHead(nn.Module):
    def __init__(self, in_dim,
                 out_dim,
                 use_bn=False,
                 norm_last_layer=True,
                 nlayers=3,
                 hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            dino_utils.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def freeze_the_first_layers(configs, model: ViT_CCT):
    """This method only permits the last half blocks of our network to be trained.

    Args:
        model (_type_): _description_
    """
    number_of_the_first_layers_to_be_frozen = configs.configs_arch.number_of_the_first_layers_to_be_frozen
    if number_of_the_first_layers_to_be_frozen == 0:
        return
    
    num_layers = 0

    # 1- We find the number of layers
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        res = find_patterns_sequencially(name, [r'blocks\.\d+\.', r'\d+'])
        if res is not None:
            layer_num = int(res)
            if layer_num > num_layers:
                num_layers = layer_num

    # 2- We disable the requires_grad of the parameters of the first half
    if number_of_the_first_layers_to_be_frozen > num_layers:
        configs.logger.warning(f"number_of_the_first_layers_to_be_frozen is {number_of_the_first_layers_to_be_frozen} while our model has {num_layers} layers!")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        res = find_patterns_sequencially(name, [r'blocks\.\d+\.', r'\d+'])
        if res is not None:     # It is a parameter of a block
            layer_num = int(res)
            if layer_num < number_of_the_first_layers_to_be_frozen:
                param.requires_grad = False
        else:
            param.requires_grad = False
