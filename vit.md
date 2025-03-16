# Table of Contents
- vit_pytorch/pit.py
- vit_pytorch/simple_vit.py
- vit_pytorch/cct.py
- vit_pytorch/simple_vit_with_qk_norm.py
- vit_pytorch/vit_for_small_dataset.py
- vit_pytorch/na_vit_nested_tensor_3d.py
- vit_pytorch/mae.py
- vit_pytorch/simple_vit_3d.py
- vit_pytorch/recorder.py
- vit_pytorch/dino.py
- vit_pytorch/vit_with_patch_merger.py
- vit_pytorch/cvt.py
- vit_pytorch/rvt.py
- vit_pytorch/simple_vit_with_value_residual.py
- vit_pytorch/mp3.py
- vit_pytorch/nest.py
- vit_pytorch/sep_vit.py
- vit_pytorch/xcit.py
- vit_pytorch/mobile_vit.py
- vit_pytorch/simple_vit_1d.py
- vit_pytorch/extractor.py
- vit_pytorch/levit.py
- vit_pytorch/simple_uvit.py
- vit_pytorch/simple_vit_with_fft.py
- vit_pytorch/mpp.py
- vit_pytorch/es_vit.py
- vit_pytorch/parallel_vit.py
- vit_pytorch/max_vit.py
- vit_pytorch/twins_svt.py
- vit_pytorch/cross_vit.py
- vit_pytorch/simple_vit_with_hyper_connections.py
- vit_pytorch/simple_flash_attn_vit.py
- vit_pytorch/vit_with_patch_dropout.py
- vit_pytorch/__init__.py
- vit_pytorch/simmim.py
- vit_pytorch/ats_vit.py
- vit_pytorch/vit.py
- vit_pytorch/cait.py
- vit_pytorch/simple_flash_attn_vit_3d.py
- vit_pytorch/t2t.py
- vit_pytorch/na_vit_nested_tensor.py
- vit_pytorch/na_vit.py
- vit_pytorch/deepvit.py
- vit_pytorch/regionvit.py
- vit_pytorch/distill.py
- vit_pytorch/simple_vit_with_register_tokens.py
- vit_pytorch/normalized_vit.py
- vit_pytorch/vivit.py
- vit_pytorch/crossformer.py
- vit_pytorch/vit_3d.py
- vit_pytorch/efficient.py
- vit_pytorch/local_vit.py
- vit_pytorch/max_vit_with_registers.py
- vit_pytorch/vit_1d.py
- vit_pytorch/cct_3d.py
- vit_pytorch/look_vit.py
- vit_pytorch/scalable_vit.py
- vit_pytorch/learnable_memory_vit.py
- vit_pytorch/simple_vit_with_patch_dropout.py

## File: vit_pytorch/pit.py

- Extension: .py
- Language: python
- Size: 5666 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from math import sqrt

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# depthwise convolution, for pooling

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_out, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

# pooling layer

class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(dim, dim * 2, kernel_size = 3, stride = 2, padding = 1)
        self.cls_ff = nn.Linear(dim, dim * 2)

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h = int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')

        return torch.cat((cls_token, tokens), dim = 1)

# main class

class PiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        channels = 3
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(depth, tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'
        heads = cast_tuple(heads, len(depth))

        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            nn.Unfold(kernel_size = patch_size, stride = patch_size // 2),
            Rearrange('b c n -> b n c'),
            nn.Linear(patch_dim, dim)
        )

        output_size = conv_output_size(image_size, patch_size, patch_size // 2)
        num_patches = output_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        layers = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)
            
            layers.append(Transformer(dim, layer_depth, layer_heads, dim_head, mlp_dim, dropout))

            if not_last:
                layers.append(Pool(dim))
                dim *= 2

        self.layers = nn.Sequential(*layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.layers(x)

        return self.mlp_head(x[:, 0])

```

## File: vit_pytorch/simple_vit.py

- Extension: .py
- Language: python
- Size: 3834 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

```

## File: vit_pytorch/cct.py

- Extension: .py
- Language: python
- Size: 12001 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# CCT Models

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']


def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = default(stride, max(1, (kernel_size // 2) - 1))
    padding = default(padding, max(1, (kernel_size // 2)))

    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)

# positional

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

# modules

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(chan_in, chan_out,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return rearrange(self.conv_layers(x), 'b c h w -> b (h w) c')

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        assert positional_embedding in {'sine', 'learnable', 'none'}

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                               requires_grad=True)
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        else:
            self.positional_emb = nn.Parameter(sinusoidal_embedding(sequence_length, embedding_dim),
                                               requires_grad=False)

        self.dropout = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        b = x.shape[0]

        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_token, x), dim=1)

        if exists(self.positional_emb):
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim = 1), x)
        else:
            x = x[:, 0]

        return self.fc(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# CCT Main model

class CCT(nn.Module):
    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        n_input_channels=3,
        n_conv_layers=1,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        *args, **kwargs
    ):
        super().__init__()
        img_height, img_width = pair(img_size)

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_height,
                                                           width=img_width),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)

```

## File: vit_pytorch/simple_vit_with_qk_norm.py

- Extension: .py
- Language: python
- Size: 4548 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper

# in latest tweet, seem to claim more stable training at higher learning rates
# unsure if this has taken off within Brain, or it has some hidden drawback

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim) / self.scale)

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.LayerNorm(dim)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

```

## File: vit_pytorch/vit_for_small_dataset.py

- Extension: .py
- Language: python
- Size: 4829 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/na_vit_nested_tensor_3d.py

- Extension: .py
- Language: python
- Size: 11035 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from __future__ import annotations

from typing import List
from functools import partial

import torch
import packaging.version as pkg_version

from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.nested import nested_tensor

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# feedforward

def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim, bias = False),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qk_norm = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias = False)

        dim_inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_queries = nn.Linear(dim, dim_inner, bias = False)
        self.to_keys = nn.Linear(dim, dim_inner, bias = False)
        self.to_values = nn.Linear(dim, dim_inner, bias = False)

        # in the paper, they employ qk rmsnorm, a way to stabilize attention
        # will use layernorm in place of rmsnorm, which has been shown to work in certain papers. requires l2norm on non-ragged dimension to be supported in nested tensors

        self.query_norm = nn.LayerNorm(dim_head, bias = False) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(dim_head, bias = False) if qk_norm else nn.Identity()

        self.dropout = dropout

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self, 
        x,
        context: Tensor | None = None
    ):

        x = self.norm(x)

        # for attention pooling, one query pooling to entire sequence

        context = default(context, x)

        # queries, keys, values

        query = self.to_queries(x)
        key = self.to_keys(context)
        value = self.to_values(context)

        # split heads

        def split_heads(t):
            return t.unflatten(-1, (self.heads, self.dim_head))

        def transpose_head_seq(t):
            return t.transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))

        # qk norm for attention stability

        query = self.query_norm(query)
        key = self.key_norm(key)

        query, key, value = map(transpose_head_seq, (query, key, value))

        # attention

        out = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p = self.dropout if self.training else 0.
        )

        # merge heads

        out = out.transpose(1, 2).flatten(-2)

        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., qk_norm = True):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, qk_norm = qk_norm),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = nn.LayerNorm(dim, bias = False)

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class NaViT(Module):
    def __init__(
        self,
        *,
        image_size,
        max_frames,
        patch_size,
        frame_patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        num_registers = 4,
        qk_rmsnorm = True,
        token_dropout_prob: float | None = None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        if pkg_version.parse(torch.__version__) < pkg_version.parse('2.5'):
            print('nested tensor NaViT was tested on pytorch 2.5')

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.token_dropout_prob = token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'
        assert divisible_by(max_frames, frame_patch_size)

        patch_frame_dim, patch_height_dim, patch_width_dim = (max_frames // frame_patch_size), (image_height // patch_size), (image_width // patch_size)

        patch_dim = channels * (patch_size ** 2) * frame_patch_size

        self.channels = channels
        self.patch_size = patch_size
        self.to_patches = Rearrange('c (f pf) (h p1) (w p2) -> f h w (c p1 p2 pf)', p1 = patch_size, p2 = patch_size, pf = frame_patch_size)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embed_frame = nn.Parameter(torch.zeros(patch_frame_dim, dim))
        self.pos_embed_height = nn.Parameter(torch.zeros(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.zeros(patch_width_dim, dim))

        # register tokens

        self.register_tokens = nn.Parameter(torch.zeros(num_registers, dim))

        nn.init.normal_(self.pos_embed_frame, std = 0.02)
        nn.init.normal_(self.pos_embed_height, std = 0.02)
        nn.init.normal_(self.pos_embed_width, std = 0.02)
        nn.init.normal_(self.register_tokens, std = 0.02)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qk_rmsnorm)

        # final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, bias = False),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        volumes: List[Tensor], # different resolution images / CT scans
    ):
        batch, device = len(volumes), self.device
        arange = partial(torch.arange, device = device)

        assert all([volume.ndim == 4 and volume.shape[0] == self.channels for volume in volumes]), f'all volumes must have {self.channels} channels and number of dimensions of {self.channels} (channels, frame, height, width)'

        all_patches = [self.to_patches(volume) for volume in volumes]

        # prepare factorized positional embedding height width indices

        positions = []

        for patches in all_patches:
            patch_frame, patch_height, patch_width = patches.shape[:3]
            fhw_indices = torch.stack(torch.meshgrid((arange(patch_frame), arange(patch_height), arange(patch_width)), indexing = 'ij'), dim = -1)
            fhw_indices = rearrange(fhw_indices, 'f h w c -> (f h w) c')

            positions.append(fhw_indices)

        # need the sizes to compute token dropout + positional embedding

        tokens = [rearrange(patches, 'f h w d -> (f h w) d') for patches in all_patches]

        # handle token dropout

        seq_lens = torch.tensor([i.shape[0] for i in tokens], device = device)

        if self.training and self.token_dropout_prob > 0:

            keep_seq_lens = ((1. - self.token_dropout_prob) * seq_lens).int().clamp(min = 1)

            kept_tokens = []
            kept_positions = []

            for one_image_tokens, one_image_positions, seq_len, num_keep in zip(tokens, positions, seq_lens, keep_seq_lens):
                keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                one_image_kept_tokens = one_image_tokens[keep_indices]
                one_image_kept_positions = one_image_positions[keep_indices]

                kept_tokens.append(one_image_kept_tokens)
                kept_positions.append(one_image_kept_positions)

            tokens, positions, seq_lens = kept_tokens, kept_positions, keep_seq_lens

        # add all height and width factorized positions


        frame_indices, height_indices, width_indices = torch.cat(positions).unbind(dim = -1)
        frame_embed, height_embed, width_embed = self.pos_embed_frame[frame_indices], self.pos_embed_height[height_indices], self.pos_embed_width[width_indices]

        pos_embed = frame_embed + height_embed + width_embed

        tokens = torch.cat(tokens)

        # linear projection to patch embeddings

        tokens = self.to_patch_embedding(tokens)

        # absolute positions

        tokens = tokens + pos_embed

        # add register tokens

        tokens = tokens.split(seq_lens.tolist())

        tokens = [torch.cat((self.register_tokens, one_tokens)) for one_tokens in tokens]

        # use nested tensor for transformers and save on padding computation

        tokens = nested_tensor(tokens, layout = torch.jagged, device = device)

        # embedding dropout

        tokens = self.dropout(tokens)

        # transformer

        tokens = self.transformer(tokens)

        # attention pooling
        # will use a jagged tensor for queries, as SDPA requires all inputs to be jagged, or not

        attn_pool_queries = [rearrange(self.attn_pool_queries, '... -> 1 ...')] * batch

        attn_pool_queries = nested_tensor(attn_pool_queries, layout = torch.jagged)

        pooled = self.attn_pool(attn_pool_queries, tokens)

        # back to unjagged

        logits = torch.stack(pooled.unbind())

        logits = rearrange(logits, 'b 1 d -> b d')

        logits = self.to_latent(logits)

        return self.mlp_head(logits)

# quick test

if __name__ == '__main__':

    # works for torch 2.5

    v = NaViT(
        image_size = 256,
        max_frames = 8,
        patch_size = 32,
        frame_patch_size = 2,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.,
        emb_dropout = 0.,
        token_dropout_prob = 0.1
    )

    # 5 volumetric data (videos or CT scans) of different resolutions - List[Tensor]

    volumes = [
        torch.randn(3, 2, 256, 256), torch.randn(3, 8, 128, 128),
        torch.randn(3, 4, 128, 256), torch.randn(3, 2, 256, 128),
        torch.randn(3, 4, 64, 256)
    ]

    assert v(volumes).shape == (5, 1000)

    v(volumes).sum().backward()

```

## File: vit_pytorch/mae.py

- Extension: .py
- Language: python
- Size: 4094 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from vit_pytorch.vit import Transformer

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss

```

## File: vit_pytorch/simple_vit_3d.py

- Extension: .py
- Language: python
- Size: 4344 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by the frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, video):
        *_, h, w, dtype = *video.shape, video.dtype

        x = self.to_patch_embedding(video)
        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

```

## File: vit_pytorch/recorder.py

- Extension: .py
- Language: python
- Size: 1772 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from functools import wraps
import torch
from torch import nn

from vit_pytorch.vit import Attention

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    def __init__(self, vit, device = None):
        super().__init__()
        self.vit = vit

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.vit.transformer, Attention)
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))

        attns = torch.stack(recordings, dim = 1) if len(recordings) > 0 else None
        return pred, attns

```

## File: vit_pytorch/dino.py

- Extension: .py
- Language: python
- Size: 9662 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import copy
import random
from functools import wraps, partial

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# helper functions

def exists(val):
    return val is not None

def default(val, default):
    return val if exists(val) else default

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss function # (algorithm 1 in the paper)

def loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)
    return - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
        return x / norm

class MLP(nn.Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size = 256):
        super().__init__()

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend([
                nn.Linear(layer_dim_in, layer_dim_out),
                nn.GELU() if not is_last else nn.Identity()
            ])

        self.net = nn.Sequential(
            *layers,
            L2Norm(),
            nn.Linear(hidden_size, dim_out)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, output_dim, projection_hidden_size, projection_num_layers, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_hidden_size = projection_hidden_size
        self.projection_num_layers = projection_num_layers
        self.output_dim = output_dim

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.flatten(1)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    def get_embedding(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        embed = self.get_embedding(x)
        if not return_projection:
            return embed

        projector = self._get_projector(embed)
        return projector(embed), embed

# main class

class Dino(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_hidden_size = 256,
        num_classes_K = 65336,
        projection_layers = 4,
        student_temp = 0.9,
        teacher_temp = 0.04,
        local_upper_crop_scale = 0.4,
        global_lower_crop_scale = 0.5,
        moving_average_decay = 0.9,
        center_moving_average_decay = 0.9,
        augment_fn = None,
        augment_fn2 = None
    ):
        super().__init__()
        self.net = net

        # default BYOL augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)

        # local and global crops

        self.local_crop = T.RandomResizedCrop((image_size, image_size), scale = (0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))

        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = hidden_layer)

        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)

        self.register_buffer('teacher_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_centers',  torch.zeros(1, num_classes_K))

        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.last_teacher_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        student_temp = None,
        teacher_temp = None
    ):
        if return_embedding:
            return self.student_encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        student_proj_one, _ = self.student_encoder(local_image_one)
        student_proj_two, _ = self.student_encoder(local_image_two)

        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one, _ = teacher_encoder(global_image_one)
            teacher_proj_two, _ = teacher_encoder(global_image_two)

        loss_fn_ = partial(
            loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_centers
        )

        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim = 0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

        loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2
        return loss

```

## File: vit_pytorch/vit_with_patch_merger.py

- Extension: .py
- Language: python
- Size: 4810 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val ,d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# patch merger class

class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim = -1)
        return torch.matmul(attn, x)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., patch_merge_layer = None, patch_merge_num_tokens = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        self.patch_merge_layer_index = default(patch_merge_layer, depth // 2) - 1 # default to mid-way through transformer, as shown in paper
        self.patch_merger = PatchMerger(dim = dim, num_tokens_out = patch_merge_num_tokens)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x

            if index == self.patch_merge_layer_index:
                x = self.patch_merger(x)

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, patch_merge_layer = None, patch_merge_num_tokens = 8, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, patch_merge_layer, patch_merge_num_tokens)

        self.mlp_head = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        return self.mlp_head(x)

```

## File: vit_pytorch/cvt.py

- Extension: .py
- Language: python
- Size: 5935 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CvT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_emb_kernel = 7,
        s1_emb_stride = 4,
        s1_proj_kernel = 3,
        s1_kv_proj_stride = 2,
        s1_heads = 1,
        s1_depth = 1,
        s1_mlp_mult = 4,
        s2_emb_dim = 192,
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.,
        channels = 3
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = channels
        layers = []

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
            ))

            dim = config['emb_dim']

        self.layers = nn.Sequential(*layers)

        self.to_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        latents = self.layers(x)
        return self.to_logits(latents)

```

## File: vit_pytorch/rvt.py

- Extension: .py
- Language: python
- Size: 7731 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from math import sqrt, pi, log

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.amp import autocast

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# rotary embeddings

@autocast('cuda', enabled = False)
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    @autocast('cuda', enabled = False)
    def forward(self, x):
        device, dtype, n = x.device, x.dtype, int(sqrt(x.shape[-2]))

        seq = torch.linspace(-1., 1., steps = n, device = device)
        seq = seq.unsqueeze(-1)

        scales = self.scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        seq = seq * scales * pi

        x_sinu = repeat(seq, 'i d -> i j d', j = n)
        y_sinu = repeat(seq, 'j d -> i j d', i = n)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

# helper classes

class SpatialConv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel, bias = False):
        super().__init__()
        self.conv = DepthWiseConv2d(dim_in, dim_out, kernel, padding = kernel // 2, bias = False)
        self.cls_proj = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x, fmap_dims):
        cls_token, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (h w) d -> b d h w', **fmap_dims)
        x = self.conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        cls_token = self.cls_proj(cls_token)
        return torch.cat((cls_token, x), dim = 1)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., use_glu = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
            GEGLU() if use_glu else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_rotary = True, use_ds_conv = True, conv_query_kernel = 5):
        super().__init__()
        inner_dim = dim_head *  heads
        self.use_rotary = use_rotary
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.use_ds_conv = use_ds_conv

        self.to_q = SpatialConv(dim, inner_dim, conv_query_kernel, bias = False) if use_ds_conv else nn.Linear(dim, inner_dim, bias = False)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, pos_emb, fmap_dims):
        b, n, _, h = *x.shape, self.heads

        to_q_kwargs = {'fmap_dims': fmap_dims} if self.use_ds_conv else {}

        x = self.norm(x)

        q = self.to_q(x, **to_q_kwargs)

        qkv = (q, *self.to_kv(x).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        if self.use_rotary:
            # apply 2d rotary embeddings to queries and keys, excluding CLS tokens

            sin, cos = pos_emb
            dim_rotary = sin.shape[-1]

            (q_cls, q), (k_cls, k) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k))

            # handle the case where rotary dimension < head dimension

            (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
            q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
            q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))

            # concat back the CLS tokens

            q = torch.cat((q_cls, q), dim = 1)
            k = torch.cat((k_cls, k), dim = 1)

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, image_size, dropout = 0., use_rotary = True, use_ds_conv = True, use_glu = True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = AxialRotaryEmbedding(dim_head, max_freq = image_size)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, use_rotary = use_rotary, use_ds_conv = use_ds_conv),
                FeedForward(dim, mlp_dim, dropout = dropout, use_glu = use_glu)
            ]))
    def forward(self, x, fmap_dims):
        pos_emb = self.pos_emb(x[:, 1:])

        for attn, ff in self.layers:
            x = attn(x, pos_emb = pos_emb, fmap_dims = fmap_dims) + x
            x = ff(x) + x
        return x

# Rotary Vision Transformer

class RvT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., use_rotary = True, use_ds_conv = True, use_glu = True):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, image_size, dropout, use_rotary, use_ds_conv, use_glu)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        b, _, h, w, p = *img.shape, self.patch_size

        x = self.to_patch_embedding(img)
        n = x.shape[1]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        fmap_dims = {'h': h // p, 'w': w // p}
        x = self.transformer(x, fmap_dims = fmap_dims)

        return self.mlp_head(x[:, 0])

```

## File: vit_pytorch/simple_vit_with_value_residual.py

- Extension: .py
- Language: python
- Size: 4748 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

def FeedForward(dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, learned_value_residual_mix = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.to_residual_mix = nn.Sequential(
            nn.Linear(dim, heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        ) if learned_value_residual_mix else (lambda _: 0.5)

    def forward(self, x, value_residual = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if exists(value_residual):
            mix = self.to_residual_mix(x)
            v = v * mix + value_residual * (1. - mix)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), v

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        for i in range(depth):
            is_first = i == 0
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, learned_value_residual_mix = not is_first),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        value_residual = None

        for attn, ff in self.layers:

            attn_out, values = attn(x, value_residual = value_residual)
            value_residual = default(value_residual, values)

            x = attn_out + x
            x = ff(x) + x

        return self.norm(x)

class SimpleViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# quick test

if __name__ == '__main__':
    v = SimpleViT(
        num_classes = 1000,
        image_size = 256,
        patch_size = 8,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
    )

    images = torch.randn(2, 3, 256, 256)

    logits = v(images)

```

## File: vit_pytorch/mp3.py

- Extension: .py
- Language: python
- Size: 6160 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# positional embedding

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# (cross)attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)

        context = self.norm(context) if exists(context) else x

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, num_classes, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.dim = dim
        self.num_patches = num_patches

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# Masked Position Prediction Pre-Training

class MP3(nn.Module):
    def __init__(self, vit: ViT, masking_ratio):
        super().__init__()
        self.vit = vit

        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        dim = vit.dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, vit.num_patches)
        )

    def forward(self, img):
        device = img.device
        tokens = self.vit.to_patch_embedding(img)
        tokens = rearrange(tokens, 'b ... d -> b (...) d')

        batch, num_patches, *_ = tokens.shape

        # Masking
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens_unmasked = tokens[batch_range, unmasked_indices]

        attended_tokens = self.vit.transformer(tokens, tokens_unmasked)
        logits = rearrange(self.mlp_head(attended_tokens), 'b n d -> (b n) d')
        
        # Define labels
        labels = repeat(torch.arange(num_patches, device = device), 'n -> (b n)', b = batch)
        loss = F.cross_entropy(logits, labels)

        return loss

```

## File: vit_pytorch/nest.py

- Extension: .py
- Language: python
- Size: 5837 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# helpers

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class NesT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across hierarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            LayerNorm(patch_dim),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
            LayerNorm(layer_dims[0])
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)

        self.layers = nn.ModuleList([])

        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))


        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape

        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)

        return self.mlp_head(x)

```

## File: vit_pytorch/sep_vit.py

- Extension: .py
- Language: python
- Size: 8857 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class OverlappingPatchEmbed(nn.Module):
    def __init__(self, dim_in, dim_out, stride = 2):
        super().__init__()
        kernel_size = stride * 2 - 1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride = stride, padding = padding)

    def forward(self, x):
        return self.conv(x)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class DSSA(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        inner_dim = dim_head * heads

        self.norm = ChanLayerNorm(dim)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)

        # window tokens

        self.window_tokens = nn.Parameter(torch.randn(dim))

        # prenorm and non-linearity for window tokens
        # then projection to queries and keys for window tokens

        self.window_tokens_to_qk = nn.Sequential(
            nn.LayerNorm(dim_head),
            nn.GELU(),
            Rearrange('b h n c -> b (h c) n'),
            nn.Conv1d(inner_dim, inner_dim * 2, 1),
            Rearrange('b (h c) n -> b h n c', h = heads),
        )

        # window attention

        self.window_attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        einstein notation

        b - batch
        c - channels
        w1 - window size (height)
        w2 - also window size (width)
        i - sequence dimension (source)
        j - sequence dimension (target dimension to be reduced)
        h - heads
        x - height of feature map divided by window size
        y - width of feature map divided by window size
        """

        batch, height, width, heads, wsz = x.shape[0], *x.shape[-2:], self.heads, self.window_size
        assert (height % wsz) == 0 and (width % wsz) == 0, f'height {height} and width {width} must be divisible by window size {wsz}'
        num_windows = (height // wsz) * (width // wsz)

        x = self.norm(x)

        # fold in windows for "depthwise" attention - not sure why it is named depthwise when it is just "windowed" attention

        x = rearrange(x, 'b c (h w1) (w w2) -> (b h w) c (w1 w2)', w1 = wsz, w2 = wsz)

        # add windowing tokens

        w = repeat(self.window_tokens, 'c -> b c 1', b = x.shape[0])
        x = torch.cat((w, x), dim = -1)

        # project for queries, keys, value

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # scale

        q = q * self.scale

        # similarity

        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention

        attn = self.attend(dots)

        # aggregate values

        out = torch.matmul(attn, v)

        # split out windowed tokens

        window_tokens, windowed_fmaps = out[:, :, 0], out[:, :, 1:]

        # early return if there is only 1 window

        if num_windows == 1:
            fmap = rearrange(windowed_fmaps, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
            return self.to_out(fmap)

        # carry out the pointwise attention, the main novelty in the paper

        window_tokens = rearrange(window_tokens, '(b x y) h d -> b h (x y) d', x = height // wsz, y = width // wsz)
        windowed_fmaps = rearrange(windowed_fmaps, '(b x y) h n d -> b h (x y) n d', x = height // wsz, y = width // wsz)

        # windowed queries and keys (preceded by prenorm activation)

        w_q, w_k = self.window_tokens_to_qk(window_tokens).chunk(2, dim = -1)

        # scale

        w_q = w_q * self.scale

        # similarities

        w_dots = einsum('b h i d, b h j d -> b h i j', w_q, w_k)

        w_attn = self.window_attend(w_dots)

        # aggregate the feature maps from the "depthwise" attention step (the most interesting part of the paper, one i haven't seen before)

        aggregated_windowed_fmap = einsum('b h i j, b h j w d -> b h i w d', w_attn, windowed_fmaps)

        # fold back the windows and then combine heads for aggregation

        fmap = rearrange(aggregated_windowed_fmap, 'b h (x y) (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz, y = width // wsz, w1 = wsz, w2 = wsz)
        return self.to_out(fmap)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        dropout = 0.,
        norm_output = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                DSSA(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout),
            ]))

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class SepViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        window_size = 7,
        dim_head = 32,
        ff_mult = 4,
        channels = 3,
        dropout = 0.
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (channels, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        strides = (4, *((2,) * (num_stages - 1)))

        hyperparams_per_stage = [heads, window_size]
        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        self.layers = nn.ModuleList([])

        for ind, ((layer_dim_in, layer_dim), layer_depth, layer_stride, layer_heads, layer_window_size) in enumerate(zip(dim_pairs, depth, strides, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            self.layers.append(nn.ModuleList([
                OverlappingPatchEmbed(layer_dim_in, layer_dim, stride = layer_stride),
                PEG(layer_dim),
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_mult = ff_mult, dropout = dropout, norm_output = not is_last),
            ]))

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):

        for ope, peg, transformer in self.layers:
            x = ope(x)
            x = peg(x)
            x = transformer(x)

        return self.mlp_head(x)

```

## File: vit_pytorch/xcit.py

- Extension: .py
- Language: python
- Size: 8539 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from random import randrange

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 > depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.fn = fn
        self.scale = nn.Parameter(torch.full((dim,), init_eps))

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        h = self.heads

        x = self.norm(x)
        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class XCAttention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.heads
        x, ps = pack_one(x, 'b * d')

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        sim = einsum('b h i n, b h j n -> b h i j', q, k) * self.temperature.exp()

        attn = self.attend(sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j n -> b h i n', attn, v)
        out = rearrange(out, 'b h d n -> b n (h d)')

        out = unpack_one(out, ps, 'b * d')
        return self.to_out(out)

class LocalPatchInteraction(Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        assert (kernel_size % 2) == 1
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(dim, dim, kernel_size, padding = padding, groups = dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size, padding = padding, groups = dim),
            Rearrange('b c h w -> b h w c'),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            layer = ind + 1
            self.layers.append(ModuleList([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = layer),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = layer)
            ]))

    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x

        return x

class XCATransformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size = 3, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            layer = ind + 1
            self.layers.append(ModuleList([
                LayerScale(dim, XCAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = layer),
                LayerScale(dim, LocalPatchInteraction(dim, local_patch_kernel_size), depth = layer),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = layer)
            ]))

    def forward(self, x):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for cross_covariance_attn, local_patch_interaction, ff in layers:
            x = cross_covariance_attn(x) + x
            x = local_patch_interaction(x) + x
            x = ff(x) + x

        return x

class XCiT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        local_patch_kernel_size = 3,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.xcit_transformer = XCATransformer(dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size, dropout, layer_dropout)

        self.final_norm = nn.LayerNorm(dim)

        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)

        x, ps = pack_one(x, 'b * d')

        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        x = unpack_one(x, ps, 'b * d')

        x = self.dropout(x)

        x = self.xcit_transformer(x)

        x = self.final_norm(x)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)

        x = rearrange(x, 'b ... d -> b (...) d')
        cls_tokens = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(cls_tokens[:, 0])

```

## File: vit_pytorch/mobile_vit.py

- Extension: .py
- Language: python
- Size: 7675 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Reduce

# helpers

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(dims[1], depths[1], channels[7],
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], depths[2], channels[9],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1_bn(channels[-2], last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)

```

## File: vit_pytorch/simple_vit_1d.py

- Extension: .py
- Language: python
- Size: 3696 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()

        assert seq_len % patch_size == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, series):
        *_, n, dtype = *series.shape, series.dtype

        x = self.to_patch_embedding(series)
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

if __name__ == '__main__':

    v = SimpleViT(
        seq_len = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    time_series = torch.randn(4, 3, 256)
    logits = v(time_series) # (4, 1000)

```

## File: vit_pytorch/extractor.py

- Extension: .py
- Language: python
- Size: 2478 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

def exists(val):
    return val is not None

def identity(t):
    return t

def clone_and_detach(t):
    return t.clone().detach()

def apply_tuple_or_single(fn, val):
    if isinstance(val, tuple):
        return tuple(map(fn, val))
    return fn(val)

class Extractor(nn.Module):
    def __init__(
        self,
        vit,
        device = None,
        layer = None,
        layer_name = 'transformer',
        layer_save_input = False,
        return_embeddings_only = False,
        detach = True
    ):
        super().__init__()
        self.vit = vit

        self.data = None
        self.latents = None
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

        self.layer = layer
        self.layer_name = layer_name
        self.layer_save_input = layer_save_input # whether to save input or output of layer
        self.return_embeddings_only = return_embeddings_only

        self.detach_fn = clone_and_detach if detach else identity

    def _hook(self, _, inputs, output):
        layer_output = inputs if self.layer_save_input else output
        self.latents = apply_tuple_or_single(self.detach_fn, layer_output)

    def _register_hook(self):
        if not exists(self.layer):
            assert hasattr(self.vit, self.layer_name), 'layer whose output to take as embedding not found in vision transformer'
            layer = getattr(self.vit, self.layer_name)
        else:
            layer = self.layer

        handle = layer.register_forward_hook(self._hook)
        self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        del self.latents
        self.latents = None

    def forward(
        self,
        img,
        return_embeddings_only = False
    ):
        assert not self.ejected, 'extractor has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        target_device = self.device if exists(self.device) else img.device
        latents = apply_tuple_or_single(lambda t: t.to(target_device), self.latents)

        if return_embeddings_only or self.return_embeddings_only:
            return latents

        return pred, latents

```

## File: vit_pytorch/levit.py

- Extension: .py
- Language: python
- Size: 6474 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from math import ceil

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))

def always(val):
    return lambda *args, **kwargs: val

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, fmap_size, heads = 8, dim_key = 32, dim_value = 64, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        inner_dim_key = dim_key *  heads
        inner_dim_value = dim_value *  heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        self.to_q = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False), nn.BatchNorm2d(inner_dim_key))
        self.to_k = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, bias = False), nn.BatchNorm2d(inner_dim_key))
        self.to_v = nn.Sequential(nn.Conv2d(dim, inner_dim_value, 1, bias = False), nn.BatchNorm2d(inner_dim_value))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        self.to_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm,
            nn.Dropout(dropout)
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step = (2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range, indexing = 'ij'), dim = -1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range, indexing = 'ij'), dim = -1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        qkv = (q, self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult = 2, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and dim == dim_out

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, fmap_size = fmap_size, heads = heads, dim_key = dim_key, dim_value = dim_value, dropout = dropout, downsample = downsample, dim_out = dim_out),
                FeedForward(dim_out, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x) + attn_res
            x = ff(x) + x
        return x

class LeViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_mult,
        stages = 3,
        dim_key = 32,
        dim_value = 64,
        dropout = 0.,
        num_distill_classes = None
    ):
        super().__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2, padding = 1),
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1),
            nn.Conv2d(64, 128, 3, stride = 2, padding = 1),
            nn.Conv2d(128, dims[0], 3, stride = 2, padding = 1)
        )

        fmap_size = image_size // (2 ** 4)
        layers = []

        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == (stages - 1)
            layers.append(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))

            if not is_last:
                next_dim = dims[ind + 1]
                layers.append(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out = next_dim, downsample = True))
                fmap_size = ceil(fmap_size / 2)

        self.backbone = nn.Sequential(*layers)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...')
        )

        self.distill_head = nn.Linear(dim, num_distill_classes) if exists(num_distill_classes) else always(None)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.conv_embedding(img)

        x = self.backbone(x)        

        x = self.pool(x)

        out = self.mlp_head(x)
        distill = self.distill_head(x)

        if exists(distill):
            return out, distill

        return out

```

## File: vit_pytorch/simple_uvit.py

- Extension: .py
- Language: python
- Size: 5126 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert divisible_by(dim, 4), "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

def FeedForward(dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )    

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for layer in range(1, depth + 1):
            latter_half = layer >= (depth / 2 + 1)

            self.layers.append(nn.ModuleList([
                nn.Linear(dim * 2, dim) if latter_half else None,
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):

        skips = []

        for ind, (combine_skip, attn, ff) in enumerate(self.layers):
            layer = ind + 1
            first_half = layer <= (self.depth / 2)

            if first_half:
                skips.append(x)

            if exists(combine_skip):
                skip = skips.pop()
                skip_and_x = torch.cat((skip, x), dim = -1)
                x = combine_skip(skip_and_x)

            x = attn(x) + x
            x = ff(x) + x

        assert len(skips) == 0

        return self.norm(x)

class SimpleUViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, num_register_tokens = 4, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim
        )

        self.register_buffer('pos_embedding', pos_embedding, persistent = False)

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch, device = img.shape[0], img.device

        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.type(x.dtype)

        r = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        x, ps = pack([x, r], 'b * d')

        x = self.transformer(x)

        x, _ = unpack(x, ps, 'b * d')

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# quick test on odd number of layers

if __name__ == '__main__':

    v = SimpleUViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 7,
        heads = 16,
        mlp_dim = 2048
    ).cuda()

    img = torch.randn(2, 3, 256, 256).cuda()

    preds = v(img)
    assert preds.shape == (2, 1000)

```

## File: vit_pytorch/simple_vit_with_fft.py

- Extension: .py
- Language: python
- Size: 5256 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch.fft import fft2
from torch import nn

from einops import rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, freq_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        freq_patch_height, freq_patch_width = pair(freq_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_height % freq_patch_height == 0 and image_width % freq_patch_width == 0, 'Image dimensions must be divisible by the freq patch size.'

        patch_dim = channels * patch_height * patch_width
        freq_patch_dim = channels * 2 * freq_patch_height * freq_patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.to_freq_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) ri -> b (h w) (p1 p2 ri c)", p1 = freq_patch_height, p2 = freq_patch_width),
            nn.LayerNorm(freq_patch_dim),
            nn.Linear(freq_patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        )

        self.freq_pos_embedding = posemb_sincos_2d(
            h = image_height // freq_patch_height,
            w = image_width // freq_patch_width,
            dim = dim
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device, dtype = img.device, img.dtype

        x = self.to_patch_embedding(img)
        freqs = torch.view_as_real(fft2(img))

        f = self.to_freq_embedding(freqs)

        x += self.pos_embedding.to(device, dtype = dtype)
        f += self.freq_pos_embedding.to(device, dtype = dtype)

        x, ps = pack((f, x), 'b * d')

        x = self.transformer(x)

        _, x = unpack(x, ps, 'b * d')
        x = reduce(x, 'b n d -> b d', 'mean')

        x = self.to_latent(x)
        return self.linear_head(x)

if __name__ == '__main__':
    vit = SimpleViT(
        num_classes = 1000,
        image_size = 256,
        patch_size = 8,
        freq_patch_size = 8,
        dim = 1024,
        depth = 1,
        heads = 8,
        mlp_dim = 2048,
    )

    images = torch.randn(8, 3, 256, 256)

    logits = vit(images)

```

## File: vit_pytorch/mpp.py

- Extension: .py
- Language: python
- Size: 5895 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import math

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

# helpers

def exists(val):
    return val is not None

def prob_mask_like(t, prob):
    batch, seq_length, _ = t.shape
    return torch.zeros((batch, seq_length)).float().uniform_(0, 1) < prob

def get_mask_subset_with_prob(patched_input, prob):
    batch, seq_len, _, device = *patched_input.shape, patched_input.device
    max_masked = math.ceil(prob * seq_len)

    rand = torch.rand((batch, seq_len), device=device)
    _, sampled_indices = rand.topk(max_masked, dim=-1)

    new_mask = torch.zeros((batch, seq_len), device=device)
    new_mask.scatter_(1, sampled_indices, 1)
    return new_mask.bool()


# mpp loss


class MPPLoss(nn.Module):
    def __init__(
        self,
        patch_size,
        channels,
        output_channel_bits,
        max_pixel_val,
        mean,
        std
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.output_channel_bits = output_channel_bits
        self.max_pixel_val = max_pixel_val

        self.mean = torch.tensor(mean).view(-1, 1, 1) if mean else None
        self.std = torch.tensor(std).view(-1, 1, 1) if std else None

    def forward(self, predicted_patches, target, mask):
        p, c, mpv, bits, device = self.patch_size, self.channels, self.max_pixel_val, self.output_channel_bits, target.device
        bin_size = mpv / (2 ** bits)

        # un-normalize input
        if exists(self.mean) and exists(self.std):
            target = target * self.std + self.mean

        # reshape target to patches
        target = target.clamp(max = mpv) # clamp just in case
        avg_target = reduce(target, 'b c (h p1) (w p2) -> b (h w) c', 'mean', p1 = p, p2 = p).contiguous()

        channel_bins = torch.arange(bin_size, mpv, bin_size, device = device)
        discretized_target = torch.bucketize(avg_target, channel_bins)

        bin_mask = (2 ** bits) ** torch.arange(0, c, device = device).long()
        bin_mask = rearrange(bin_mask, 'c -> () () c')

        target_label = torch.sum(bin_mask * discretized_target, dim = -1)

        loss = F.cross_entropy(predicted_patches[mask], target_label[mask])
        return loss


# main class


class MPP(nn.Module):
    def __init__(
        self,
        transformer,
        patch_size,
        dim,
        output_channel_bits=3,
        channels=3,
        max_pixel_val=1.0,
        mask_prob=0.15,
        replace_prob=0.5,
        random_patch_prob=0.5,
        mean=None,
        std=None
    ):
        super().__init__()
        self.transformer = transformer
        self.loss = MPPLoss(patch_size, channels, output_channel_bits,
                            max_pixel_val, mean, std)

        # extract patching function
        self.patch_to_emb = nn.Sequential(transformer.to_patch_embedding[1:])

        # output transformation
        self.to_bits = nn.Linear(dim, 2**(output_channel_bits * channels))

        # vit related dimensions
        self.patch_size = patch_size

        # mpp related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_patch_prob = random_patch_prob

        # token ids
        self.mask_token = nn.Parameter(torch.randn(1, 1, channels * patch_size ** 2))

    def forward(self, input, **kwargs):
        transformer = self.transformer
        # clone original image for loss
        img = input.clone().detach()

        # reshape raw image to patches
        p = self.patch_size
        input = rearrange(input,
                          'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=p,
                          p2=p)

        mask = get_mask_subset_with_prob(input, self.mask_prob)

        # mask input with mask patches with probability of `replace_prob` (keep patches the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mpp
        if self.random_patch_prob > 0:
            random_patch_sampling_prob = self.random_patch_prob / (
                1 - self.replace_prob)
            random_patch_prob = prob_mask_like(input,
                                               random_patch_sampling_prob).to(mask.device)

            bool_random_patch_prob = mask * (random_patch_prob == True)
            random_patches = torch.randint(0,
                                           input.shape[1],
                                           (input.shape[0], input.shape[1]),
                                           device=input.device)
            randomized_input = masked_input[
                torch.arange(masked_input.shape[0]).unsqueeze(-1),
                random_patches]
            masked_input[bool_random_patch_prob] = randomized_input[
                bool_random_patch_prob]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob).to(mask.device)
        bool_mask_replace = (mask * replace_prob) == True
        masked_input[bool_mask_replace] = self.mask_token

        # linear embedding of patches
        masked_input = self.patch_to_emb(masked_input)

        # add cls token to input sequence
        b, n, _ = masked_input.shape
        cls_tokens = repeat(transformer.cls_token, '() n d -> b n d', b=b)
        masked_input = torch.cat((cls_tokens, masked_input), dim=1)

        # add positional embeddings to input
        masked_input += transformer.pos_embedding[:, :(n + 1)]
        masked_input = transformer.dropout(masked_input)

        # get generator output and get mpp loss
        masked_input = transformer.transformer(masked_input, **kwargs)
        cls_logits = self.to_bits(masked_input)
        logits = cls_logits[:, 1:, :]

        mpp_loss = self.loss(logits, img, mask)

        return mpp_loss

```

## File: vit_pytorch/es_vit.py

- Extension: .py
- Language: python
- Size: 12657 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import copy
import random
from functools import wraps, partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms as T

from einops import rearrange, reduce, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, default):
    return val if exists(val) else default

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# tensor related helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

# loss function # (algorithm 1 in the paper)

def view_loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)
    return - (teacher_probs * log(student_probs, eps)).sum(dim = -1).mean()

def region_loss_fn(
    teacher_logits,
    student_logits,
    teacher_latent,
    student_latent,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):
    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits - centers) / teacher_temp).softmax(dim = -1)

    sim_matrix = einsum('b i d, b j d -> b i j', student_latent, teacher_latent)
    sim_indices = sim_matrix.max(dim = -1).indices
    sim_indices = repeat(sim_indices, 'b n -> b n k', k = teacher_probs.shape[-1])
    max_sim_teacher_probs = teacher_probs.gather(1, sim_indices)

    return - (max_sim_teacher_probs * log(student_probs, eps)).sum(dim = -1).mean()

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        return F.normalize(x, dim = 1, eps = eps)

class MLP(nn.Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size = 256):
        super().__init__()

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend([
                nn.Linear(layer_dim_in, layer_dim_out),
                nn.GELU() if not is_last else nn.Identity()
            ])

        self.net = nn.Sequential(
            *layers,
            L2Norm(),
            nn.Linear(hidden_size, dim_out)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, output_dim, projection_hidden_size, projection_num_layers, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.view_projector = None
        self.region_projector = None
        self.projection_hidden_size = projection_hidden_size
        self.projection_num_layers = projection_num_layers
        self.output_dim = output_dim

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('view_projector')
    def _get_view_projector(self, hidden):
        dim = hidden.shape[1]
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    @singleton('region_projector')
    def _get_region_projector(self, hidden):
        dim = hidden.shape[1]
        projector = MLP(dim, self.output_dim, self.projection_num_layers, self.projection_hidden_size)
        return projector.to(hidden)

    def get_embedding(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        region_latents = self.get_embedding(x)
        global_latent = reduce(region_latents, 'b c h w -> b c', 'mean')

        if not return_projection:
            return global_latent, region_latents

        view_projector = self._get_view_projector(global_latent)
        region_projector = self._get_region_projector(region_latents)

        region_latents = rearrange(region_latents, 'b c h w -> b (h w) c')

        return view_projector(global_latent), region_projector(region_latents), region_latents

# main class

class EsViTTrainer(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer = -2,
        projection_hidden_size = 256,
        num_classes_K = 65336,
        projection_layers = 4,
        student_temp = 0.9,
        teacher_temp = 0.04,
        local_upper_crop_scale = 0.4,
        global_lower_crop_scale = 0.5,
        moving_average_decay = 0.9,
        center_moving_average_decay = 0.9,
        augment_fn = None,
        augment_fn2 = None
    ):
        super().__init__()
        self.net = net

        # default BYOL augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)

        # local and global crops

        self.local_crop = T.RandomResizedCrop((image_size, image_size), scale = (0.05, local_upper_crop_scale))
        self.global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))

        self.student_encoder = NetWrapper(net, num_classes_K, projection_hidden_size, projection_layers, layer = hidden_layer)

        self.teacher_encoder = None
        self.teacher_ema_updater = EMA(moving_average_decay)

        self.register_buffer('teacher_view_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_view_centers',  torch.zeros(1, num_classes_K))

        self.register_buffer('teacher_region_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_region_centers',  torch.zeros(1, num_classes_K))

        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=device))

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

        new_teacher_view_centers = self.teacher_centering_ema_updater.update_average(self.teacher_view_centers, self.last_teacher_view_centers)
        self.teacher_view_centers.copy_(new_teacher_view_centers)

        new_teacher_region_centers = self.teacher_centering_ema_updater.update_average(self.teacher_region_centers, self.last_teacher_region_centers)
        self.teacher_region_centers.copy_(new_teacher_region_centers)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        student_temp = None,
        teacher_temp = None
    ):
        if return_embedding:
            return self.student_encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        student_view_proj_one, student_region_proj_one, student_latent_one = self.student_encoder(local_image_one)
        student_view_proj_two, student_region_proj_two, student_latent_two = self.student_encoder(local_image_two)

        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_view_proj_one, teacher_region_proj_one, teacher_latent_one = teacher_encoder(global_image_one)
            teacher_view_proj_two, teacher_region_proj_two, teacher_latent_two = teacher_encoder(global_image_two)

        view_loss_fn_ = partial(
            view_loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_view_centers
        )

        region_loss_fn_ = partial(
            region_loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_region_centers
        )

        # calculate view-level loss

        teacher_view_logits_avg = torch.cat((teacher_view_proj_one, teacher_view_proj_two)).mean(dim = 0)
        self.last_teacher_view_centers.copy_(teacher_view_logits_avg)

        teacher_region_logits_avg = torch.cat((teacher_region_proj_one, teacher_region_proj_two)).mean(dim = (0, 1))
        self.last_teacher_region_centers.copy_(teacher_region_logits_avg)

        view_loss = (view_loss_fn_(teacher_view_proj_one, student_view_proj_two) \
                   + view_loss_fn_(teacher_view_proj_two, student_view_proj_one)) / 2

        # calculate region-level loss

        region_loss = (region_loss_fn_(teacher_region_proj_one, student_region_proj_two, teacher_latent_one, student_latent_two) \
                     + region_loss_fn_(teacher_region_proj_two, student_region_proj_one, teacher_latent_two, student_latent_one)) / 2

        return (view_loss + region_loss) / 2

```

## File: vit_pytorch/parallel_vit.py

- Extension: .py
- Language: python
- Size: 4526 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum([fn(x) for fn in self.fns])

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_parallel_branches = 2, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        attn_block = lambda: Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        ff_block = lambda: FeedForward(dim, mlp_dim, dropout = dropout)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Parallel(*[attn_block() for _ in range(num_parallel_branches)]),
                Parallel(*[ff_block() for _ in range(num_parallel_branches)]),
            ]))

    def forward(self, x):
        for attns, ffs in self.layers:
            x = attns(x) + x
            x = ffs(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', num_parallel_branches = 2, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_parallel_branches, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/max_vit.py

- Extension: .py
- Language: python
- Size: 8315 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class Residual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    Residual(layer_dim, Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                self.layers.append(block)

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.conv_stem(x)

        for stage in self.layers:
            x = stage(x)

        return self.mlp_head(x)

```

## File: vit_pytorch/twins_svt.py

- Extension: .py
- Language: python
- Size: 7966 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods

def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PatchEmbedding(nn.Module):
    def __init__(self, *, dim, dim_out, patch_size):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            LayerNorm(patch_size ** 2 * dim),
            nn.Conv2d(patch_size ** 2 * dim, dim_out, 1),
            LayerNorm(dim_out)
        )

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = p, p2 = p)
        return self.proj(fmap)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1))

    def forward(self, x):
        return self.proj(x)

class LocalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., patch_size = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, fmap):
        fmap = self.norm(fmap)

        shape, p = fmap.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        x, y = map(lambda t: t // p, (x, y))

        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1 = p, p2 = p)

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = - 1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h = h, x = x, y = y, p1 = p, p2 = p)
        return self.to_out(out)

class GlobalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., k = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, k, stride = k, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads = 8, dim_head = 64, mlp_mult = 4, local_patch_size = 7, global_k = 7, dropout = 0., has_local = True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LocalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, patch_size = local_patch_size)) if has_local else nn.Identity(),
                Residual(FeedForward(dim, mlp_mult, dropout = dropout)) if has_local else nn.Identity(),
                Residual(GlobalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, k = global_k)),
                Residual(FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    def forward(self, x):
        for local_attn, ff1, global_attn, ff2 in self.layers:
            x = local_attn(x)
            x = ff1(x)
            x = global_attn(x)
            x = ff2(x)
        return x

class TwinsSVT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_patch_size = 4,
        s1_local_patch_size = 7,
        s1_global_k = 7,
        s1_depth = 1,
        s2_emb_dim = 128,
        s2_patch_size = 2,
        s2_local_patch_size = 7,
        s2_global_k = 7,
        s2_depth = 1,
        s3_emb_dim = 256,
        s3_patch_size = 2,
        s3_local_patch_size = 7,
        s3_global_k = 7,
        s3_depth = 5,
        s4_emb_dim = 512,
        s4_patch_size = 2,
        s4_local_patch_size = 7,
        s4_global_k = 7,
        s4_depth = 4,
        peg_kernel_size = 3,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 3
        layers = []

        for prefix in ('s1', 's2', 's3', 's4'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's4'

            dim_next = config['emb_dim']

            layers.append(nn.Sequential(
                PatchEmbedding(dim = dim, dim_out = dim_next, patch_size = config['patch_size']),
                Transformer(dim = dim_next, depth = 1, local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last),
                PEG(dim = dim_next, kernel_size = peg_kernel_size),
                Transformer(dim = dim_next, depth = config['depth'],  local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last)
            ))

            dim = dim_next

        self.layers = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

```

## File: vit_pytorch/cross_vit.py

- Extension: .py
- Language: python
- Size: 9072 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder, for small and large patches

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                ProjectInOut(lg_dim, sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# patch-based image to token embedder

class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.,
        channels = 3
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# cross ViT class

class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, channels= channels, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder(dim = lg_dim, channels = channels, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits

```

## File: vit_pytorch/simple_vit_with_hyper_connections.py

- Extension: .py
- Language: python
- Size: 7177 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
"""
ViT + Hyper-Connections + Register Tokens
https://arxiv.org/abs/2409.19606
"""

import torch
from torch import nn, tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange

# b - batch, h - heads, n - sequence, e - expansion rate / residual streams, d - feature dimension

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# hyper connections

class HyperConnection(Module):
    def __init__(
        self,
        dim,
        num_residual_streams,
        layer_index
    ):
        """ Appendix J - Algorithm 2, Dynamic only """
        super().__init__()

        self.norm = nn.LayerNorm(dim, bias = False)

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        init_alpha0 = torch.zeros((num_residual_streams, 1))
        init_alpha0[layer_index % num_residual_streams, 0] = 1.

        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(num_residual_streams)], dim = 1))

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, num_residual_streams + 1))
        self.dynamic_alpha_scale = nn.Parameter(tensor(1e-2))
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
        self.dynamic_beta_scale = nn.Parameter(tensor(1e-2))

    def width_connection(self, residuals):
        normed = self.norm(residuals)

        wc_weight = (normed @ self.dynamic_alpha_fn).tanh()
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        dc_weight = (normed @ self.dynamic_beta_fn).tanh()
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        # width connection
        mix_h = einsum(alpha, residuals, '... e1 e2, ... e1 d -> ... e2 d')

        branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]

        return branch_input, residuals, beta

    def depth_connection(
        self,
        branch_output,
        residuals,
        beta
    ):
        return einsum(branch_output, beta, "b n d, b n e -> b n e d") + residuals

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_residual_streams):
        super().__init__()

        self.num_residual_streams = num_residual_streams

        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for layer_index in range(depth):
            self.layers.append(nn.ModuleList([
                HyperConnection(dim, num_residual_streams, layer_index),
                Attention(dim, heads = heads, dim_head = dim_head),
                HyperConnection(dim, num_residual_streams, layer_index),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):

        x = repeat(x, 'b n d -> b n e d', e = self.num_residual_streams)

        for attn_hyper_conn, attn, ff_hyper_conn, ff in self.layers:

            x, attn_res, beta = attn_hyper_conn.width_connection(x)

            x = attn(x)

            x = attn_hyper_conn.depth_connection(x, attn_res, beta)

            x, ff_res, beta = ff_hyper_conn.width_connection(x)

            x = ff(x)

            x = ff_hyper_conn.depth_connection(x, ff_res, beta)

        x = reduce(x, 'b n e d -> b n d', 'sum')

        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, num_residual_streams, num_register_tokens = 4, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, num_residual_streams)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch, device = img.shape[0], img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(x)

        r = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        x, ps = pack([x, r], 'b * d')

        x = self.transformer(x)

        x, _ = unpack(x, ps, 'b * d')

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# main

if __name__ == '__main__':
    vit = SimpleViT(
        num_classes = 1000,
        image_size = 256,
        patch_size = 8,
        dim = 1024,
        depth = 12,
        heads = 8,
        mlp_dim = 2048,
        num_residual_streams = 8
    )

    images = torch.randn(3, 3, 256, 256)

    logits = vit(images)

```

## File: vit_pytorch/simple_flash_attn_vit.py

- Extension: .py
- Language: python
- Size: 5691 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from collections import namedtuple
from packaging import version

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# constants

Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# main class

class Attend(nn.Module):
    def __init__(self, use_flash = False):
        super().__init__()
        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # flash attention - https://arxiv.org/abs/2205.14135
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_flash = True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = Attend(use_flash = use_flash)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_flash):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, use_flash = use_flash),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, use_flash = True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, use_flash)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

```

## File: vit_pytorch/vit_with_patch_dropout.py

- Extension: .py
- Language: python
- Size: 4786 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., patch_dropout = 0.25):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.patch_dropout = PatchDropout(patch_dropout)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding

        x = self.patch_dropout(x)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/__init__.py

- Extension: .py
- Language: python
- Size: 144 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from vit_pytorch.vit import ViT
from vit_pytorch.simple_vit import SimpleViT

from vit_pytorch.mae import MAE
from vit_pytorch.dino import Dino

```

## File: vit_pytorch/simmim.py

- Extension: .py
- Language: python
- Size: 2709 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

class SimMIM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        masking_ratio = 0.5
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # simple linear head

        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
        self.to_pixels = nn.Linear(encoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # for indexing purposes

        batch_range = torch.arange(batch, device = device)[:, None]

        # get positions

        pos_emb = self.encoder.pos_embedding[:, 1:(num_patches + 1)]

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens = tokens + pos_emb

        # prepare mask tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_patches)
        mask_tokens = mask_tokens + pos_emb

        # calculate of patches needed to be masked, and get positions (indices) to be masked

        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(batch, num_patches, device = device).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((batch, num_patches), device = device).scatter_(-1, masked_indices, 1).bool()

        # mask tokens

        tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)

        # attend with vision transformer

        encoded = self.encoder.transformer(tokens)

        # get the masked tokens

        encoded_mask_tokens = encoded[batch_range, masked_indices]

        # small linear projection for predicted pixel values

        pred_pixel_values = self.to_pixels(encoded_mask_tokens)

        # get the masked patches for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # calculate reconstruction loss

        recon_loss = F.l1_loss(pred_pixel_values, masked_patches) / num_masked
        return recon_loss

```

## File: vit_pytorch/ats_vit.py

- Extension: .py
- Language: python
- Size: 10095 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# adaptive token sampling functions and classes

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def sample_gumbel(shape, device, dtype, eps = 1e-6):
    u = torch.empty(shape, device = device, dtype = dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

class AdaptiveTokenSampling(nn.Module):
    def __init__(self, output_num_tokens, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.output_num_tokens = output_num_tokens

    def forward(self, attn, value, mask):
        heads, output_num_tokens, eps, device, dtype = attn.shape[1], self.output_num_tokens, self.eps, attn.device, attn.dtype

        # first get the attention values for CLS token to all other tokens

        cls_attn = attn[..., 0, 1:]

        # calculate the norms of the values, for weighting the scores, as described in the paper

        value_norms = value[..., 1:, :].norm(dim = -1)

        # weigh the attention scores by the norm of the values, sum across all heads

        cls_attn = einsum('b h n, b h n -> b n', cls_attn, value_norms)

        # normalize to 1

        normed_cls_attn = cls_attn / (cls_attn.sum(dim = -1, keepdim = True) + eps)

        # instead of using inverse transform sampling, going to invert the softmax and use gumbel-max sampling instead

        pseudo_logits = log(normed_cls_attn)

        # mask out pseudo logits for gumbel-max sampling

        mask_without_cls = mask[:, 1:]
        mask_value = -torch.finfo(attn.dtype).max / 2
        pseudo_logits = pseudo_logits.masked_fill(~mask_without_cls, mask_value)

        # expand k times, k being the adaptive sampling number

        pseudo_logits = repeat(pseudo_logits, 'b n -> b k n', k = output_num_tokens)
        pseudo_logits = pseudo_logits + sample_gumbel(pseudo_logits.shape, device = device, dtype = dtype)

        # gumble-max and add one to reserve 0 for padding / mask

        sampled_token_ids = pseudo_logits.argmax(dim = -1) + 1

        # calculate unique using torch.unique and then pad the sequence from the right

        unique_sampled_token_ids_list = [torch.unique(t, sorted = True) for t in torch.unbind(sampled_token_ids)]
        unique_sampled_token_ids = pad_sequence(unique_sampled_token_ids_list, batch_first = True)

        # calculate the new mask, based on the padding

        new_mask = unique_sampled_token_ids != 0

        # CLS token never gets masked out (gets a value of True)

        new_mask = F.pad(new_mask, (1, 0), value = True)

        # prepend a 0 token id to keep the CLS attention scores

        unique_sampled_token_ids = F.pad(unique_sampled_token_ids, (1, 0), value = 0)
        expanded_unique_sampled_token_ids = repeat(unique_sampled_token_ids, 'b n -> b h n', h = heads)

        # gather the new attention scores

        new_attn = batched_index_select(attn, expanded_unique_sampled_token_ids, dim = 2)

        # return the sampled attention scores, new mask (denoting padding), as well as the sampled token indices (for the residual)
        return new_attn, new_mask, unique_sampled_token_ids

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., output_num_tokens = None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.output_num_tokens = output_num_tokens
        self.ats = AdaptiveTokenSampling(output_num_tokens) if exists(output_num_tokens) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, *, mask):
        num_tokens = x.shape[1]

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(mask):
            dots_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~dots_mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        sampled_token_ids = None

        # if adaptive token sampling is enabled
        # and number of tokens is greater than the number of output tokens
        if exists(self.output_num_tokens) and (num_tokens - 1) > self.output_num_tokens:
            attn, mask, sampled_token_ids = self.ats(attn, v, mask = mask)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), mask, sampled_token_ids

class Transformer(nn.Module):
    def __init__(self, dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        assert len(max_tokens_per_depth) == depth, 'max_tokens_per_depth must be a tuple of length that is equal to the depth of the transformer'
        assert sorted(max_tokens_per_depth, reverse = True) == list(max_tokens_per_depth), 'max_tokens_per_depth must be in decreasing order'
        assert min(max_tokens_per_depth) > 0, 'max_tokens_per_depth must have at least 1 token at any layer'

        self.layers = nn.ModuleList([])
        for _, output_num_tokens in zip(range(depth), max_tokens_per_depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, output_num_tokens = output_num_tokens, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        b, n, device = *x.shape[:2], x.device

        # use mask to keep track of the paddings when sampling tokens
        # as the duplicates (when sampling) are just removed, as mentioned in the paper
        mask = torch.ones((b, n), device = device, dtype = torch.bool)

        token_ids = torch.arange(n, device = device)
        token_ids = repeat(token_ids, 'n -> b n', b = b)

        for attn, ff in self.layers:
            attn_out, mask, sampled_token_ids = attn(x, mask = mask)

            # when token sampling, one needs to then gather the residual tokens with the sampled token ids
            if exists(sampled_token_ids):
                x = batched_index_select(x, sampled_token_ids, dim = 1)
                token_ids = batched_index_select(token_ids, sampled_token_ids, dim = 1)

            x = x + attn_out

            x = ff(x) + x

        return x, token_ids

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, max_tokens_per_depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, max_tokens_per_depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, return_sampled_token_ids = False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, token_ids = self.transformer(x)

        logits = self.mlp_head(x[:, 0])

        if return_sampled_token_ids:
            # remove CLS token and decrement by 1 to make -1 the padding
            token_ids = token_ids[:, 1:] - 1
            return logits, token_ids

        return logits

```

## File: vit_pytorch/vit.py

- Extension: .py
- Language: python
- Size: 4115 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/cait.py

- Extension: .py
- Language: python
- Size: 5671 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax

        attn = self.attend(dots)
        attn = self.dropout(attn)

        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = ind + 1),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = ind + 1)
            ]))
    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

class CaiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0])

```

## File: vit_pytorch/simple_flash_attn_vit_3d.py

- Extension: .py
- Language: python
- Size: 5563 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from packaging import version
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange

# constants

Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)

# main class

class Attend(Module):
    def __init__(self, use_flash = False, config: Config = Config(True, True, True)):
        super().__init__()
        self.config = config
        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

    def flash_attn(self, q, k, v):
        # flash attention - https://arxiv.org/abs/2205.14135
        
        with torch.backends.cuda.sdp_kernel(**self.config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_flash = True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = Attend(use_flash = use_flash)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_flash):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, use_flash = use_flash),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class SimpleViT(Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, use_flash_attn = True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by the frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, use_flash_attn)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, video):
        *_, h, w, dtype = *video.shape, video.dtype

        x = self.to_patch_embedding(video)
        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

```

## File: vit_pytorch/t2t.py

- Extension: .py
- Language: python
- Size: 2888 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import math
import torch
from torch import nn

from vit_pytorch.vit import Transformer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

# main class

class T2TViT(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth = None, heads = None, mlp_dim = None, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., transformer = None, t2t_layers = ((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = channels
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)

            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim = layer_dim, heads = 1, depth = 1, dim_head = layer_dim, mlp_dim = layer_dim, dropout = dropout) if not is_last else nn.Identity(),
            ])

        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/na_vit_nested_tensor.py

- Extension: .py
- Language: python
- Size: 9936 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from __future__ import annotations

from typing import List
from functools import partial

import torch
import packaging.version as pkg_version

from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.nested import nested_tensor

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# feedforward

def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim, bias = False),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qk_norm = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias = False)

        dim_inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_queries = nn.Linear(dim, dim_inner, bias = False)
        self.to_keys = nn.Linear(dim, dim_inner, bias = False)
        self.to_values = nn.Linear(dim, dim_inner, bias = False)

        # in the paper, they employ qk rmsnorm, a way to stabilize attention
        # will use layernorm in place of rmsnorm, which has been shown to work in certain papers. requires l2norm on non-ragged dimension to be supported in nested tensors

        self.query_norm = nn.LayerNorm(dim_head, bias = False) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(dim_head, bias = False) if qk_norm else nn.Identity()

        self.dropout = dropout

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self, 
        x,
        context: Tensor | None = None
    ):
        x = self.norm(x)

        # for attention pooling, one query pooling to entire sequence

        context = default(context, x)

        # queries, keys, values

        query = self.to_queries(x)
        key = self.to_keys(context)
        value = self.to_values(context)

        # split heads

        def split_heads(t):
            return t.unflatten(-1, (self.heads, self.dim_head))

        def transpose_head_seq(t):
            return t.transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))

        # qk norm for attention stability

        query = self.query_norm(query)
        key = self.key_norm(key)

        query, key, value = map(transpose_head_seq, (query, key, value))

        # attention

        out = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p = self.dropout if self.training else 0.
        )

        # merge heads

        out = out.transpose(1, 2).flatten(-2)

        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., qk_norm = True):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, qk_norm = qk_norm),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = nn.LayerNorm(dim, bias = False)

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class NaViT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        qk_rmsnorm = True,
        token_dropout_prob: float | None = None
    ):
        super().__init__()

        if pkg_version.parse(torch.__version__) < pkg_version.parse('2.5'):
            print('nested tensor NaViT was tested on pytorch 2.5')


        image_height, image_width = pair(image_size)

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.token_dropout_prob = token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size
        self.to_patches = Rearrange('c (h p1) (w p2) -> h w (c p1 p2)', p1 = patch_size, p2 = patch_size)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qk_rmsnorm)

        # final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, bias = False),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        images: List[Tensor], # different resolution images
    ):
        batch, device = len(images), self.device
        arange = partial(torch.arange, device = device)

        assert all([image.ndim == 3 and image.shape[0] == self.channels for image in images]), f'all images must have {self.channels} channels and number of dimensions of 3 (channels, height, width)'

        all_patches = [self.to_patches(image) for image in images]

        # prepare factorized positional embedding height width indices

        positions = []

        for patches in all_patches:
            patch_height, patch_width = patches.shape[:2]
            hw_indices = torch.stack(torch.meshgrid((arange(patch_height), arange(patch_width)), indexing = 'ij'), dim = -1)
            hw_indices = rearrange(hw_indices, 'h w c -> (h w) c')
            positions.append(hw_indices)

        # need the sizes to compute token dropout + positional embedding

        tokens = [rearrange(patches, 'h w d -> (h w) d') for patches in all_patches]

        # handle token dropout

        seq_lens = torch.tensor([i.shape[0] for i in tokens], device = device)

        if self.training and self.token_dropout_prob > 0:

            keep_seq_lens = ((1. - self.token_dropout_prob) * seq_lens).int().clamp(min = 1)

            kept_tokens = []
            kept_positions = []

            for one_image_tokens, one_image_positions, seq_len, num_keep in zip(tokens, positions, seq_lens, keep_seq_lens):
                keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                one_image_kept_tokens = one_image_tokens[keep_indices]
                one_image_kept_positions = one_image_positions[keep_indices]

                kept_tokens.append(one_image_kept_tokens)
                kept_positions.append(one_image_kept_positions)

            tokens, positions, seq_lens = kept_tokens, kept_positions, keep_seq_lens

        # add all height and width factorized positions

        height_indices, width_indices = torch.cat(positions).unbind(dim = -1)
        height_embed, width_embed = self.pos_embed_height[height_indices], self.pos_embed_width[width_indices]

        pos_embed = height_embed + width_embed

        # use nested tensor for transformers and save on padding computation

        tokens = torch.cat(tokens)

        # linear projection to patch embeddings

        tokens = self.to_patch_embedding(tokens)

        # absolute positions

        tokens = tokens + pos_embed

        tokens = nested_tensor(tokens.split(seq_lens.tolist()), layout = torch.jagged, device = device)

        # embedding dropout

        tokens = self.dropout(tokens)

        # transformer

        tokens = self.transformer(tokens)

        # attention pooling
        # will use a jagged tensor for queries, as SDPA requires all inputs to be jagged, or not

        attn_pool_queries = [rearrange(self.attn_pool_queries, '... -> 1 ...')] * batch

        attn_pool_queries = nested_tensor(attn_pool_queries, layout = torch.jagged)

        pooled = self.attn_pool(attn_pool_queries, tokens)

        # back to unjagged

        logits = torch.stack(pooled.unbind())

        logits = rearrange(logits, 'b 1 d -> b d')

        logits = self.to_latent(logits)

        return self.mlp_head(logits)

# quick test

if __name__ == '__main__':

    v = NaViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.,
        emb_dropout = 0.,
        token_dropout_prob = 0.1
    )

    # 5 images of different resolutions - List[Tensor]

    images = [
        torch.randn(3, 256, 256), torch.randn(3, 128, 128),
        torch.randn(3, 128, 256), torch.randn(3, 256, 128),
        torch.randn(3, 64, 256)
    ]

    assert v(images).shape == (5, 1000)

    v(images).sum().backward()

```

## File: vit_pytorch/na_vit.py

- Extension: .py
- Language: python
- Size: 12590 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from __future__ import annotations

from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# auto grouping images

def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = (ph * pw)
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

# normalization
# they use layernorm without bias, something that pytorch does not offer

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper

class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# feedforward

def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None
    ):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        attn_mask = None
    ):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x

        return self.norm(x)

class NaViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., token_dropout_prob = None):
        super().__init__()
        image_height, image_width = pair(image_size)

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. <= token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        batched_images: List[Tensor] | List[List[Tensor]], # assume different resolution images already grouped correctly
        group_images = False,
        group_max_seq_len = 2048
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout) and self.training

        arange = partial(torch.arange, device = device)
        pad_sequence = partial(orig_pad_sequence, batch_first = True)

        # auto pack if specified

        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images,
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout if self.training else None,
                max_seq_len = group_max_seq_len
            )

        # if List[Tensor] is not grouped -> List[List[Tensor]]

        if torch.is_tensor(batched_images[0]):
            batched_images = [batched_images]

        # process images into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device = device, dtype = torch.long)

            for image_id, image in enumerate(images):
                assert image.ndim ==3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

                ph, pw = map(lambda dim: dim // p, image_dims)

                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)

                pos = rearrange(pos, 'h w c -> (h w) c')
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p)

                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id)
                sequences.append(seq)
                positions.append(pos)

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim = 0))
            batched_positions.append(torch.cat(positions, dim = 0))

        # derive key padding mask

        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)
        seq_arange = arange(lengths.amax().item())
        key_pad_mask = rearrange(seq_arange, 'n -> 1 n') < rearrange(lengths, 'b -> b 1')

        # derive attention mask, and combine with key padding mask from above

        batched_image_ids = pad_sequence(batched_image_ids)
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')

        # combine patched images as well as the patched width / height positions for 2d positional embedding

        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        # need to know how many images for final attention pooling

        num_images = torch.tensor(num_images, device = device, dtype = torch.long)        

        # to patches

        x = self.to_patch_embedding(patches)        

        # factorized 2d absolute positional embedding

        h_indices, w_indices = patch_positions.unbind(dim = -1)

        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]

        x = x + h_pos + w_pos

        # embed dropout

        x = self.dropout(x)

        # attention

        x = self.transformer(x, attn_mask = attn_mask)

        # do attention pooling at the end

        max_queries = num_images.amax().item()

        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])

        # attention pool mask

        image_id_arange = arange(max_queries)

        attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')

        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')

        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')

        # attention pool

        x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries

        x = rearrange(x, 'b n d -> (b n) d')

        # each batch element may not have same amount of images

        is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
        is_images = rearrange(is_images, 'b n -> (b n)')

        x = x[is_images]

        # project out to logits

        x = self.to_latent(x)

        return self.mlp_head(x)

```

## File: vit_pytorch/deepvit.py

- Extension: .py
- Language: python
- Size: 4282 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DeepViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/regionvit.py

- Extension: .py
- Language: python
- Size: 9515 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def divisible_by(val, d):
    return (val % d) == 0

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# transformer classes

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim, 1)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, rel_pos_bias = None):
        h = self.heads

        # prenorm

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add relative positional bias for local tokens

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class R2LTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        window_size,
        depth = 4,
        heads = 4,
        dim_head = 32,
        attn_dropout = 0.,
        ff_dropout = 0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.window_size = window_size
        rel_positions = 2 * window_size - 1
        self.local_rel_pos_bias = nn.Embedding(rel_positions ** 2, heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout)
            ]))

    def forward(self, local_tokens, region_tokens):
        device = local_tokens.device
        lh, lw = local_tokens.shape[-2:]
        rh, rw = region_tokens.shape[-2:]
        window_size_h, window_size_w = lh // rh, lw // rw

        local_tokens = rearrange(local_tokens, 'b c h w -> b (h w) c')
        region_tokens = rearrange(region_tokens, 'b c h w -> b (h w) c')

        # calculate local relative positional bias
        
        h_range = torch.arange(window_size_h, device = device)
        w_range = torch.arange(window_size_w, device = device)

        grid_x, grid_y = torch.meshgrid(h_range, w_range, indexing = 'ij')
        grid = torch.stack((grid_x, grid_y))
        grid = rearrange(grid, 'c h w -> c (h w)')
        grid = (grid[:, :, None] - grid[:, None, :]) + (self.window_size - 1)
        bias_indices = (grid * torch.tensor([1, self.window_size * 2 - 1], device = device)[:, None, None]).sum(dim = 0)
        rel_pos_bias = self.local_rel_pos_bias(bias_indices)
        rel_pos_bias = rearrange(rel_pos_bias, 'i j h -> () h i j')
        rel_pos_bias = F.pad(rel_pos_bias, (1, 0, 1, 0), value = 0)

        # go through r2l transformer layers

        for attn, ff in self.layers:
            region_tokens = attn(region_tokens) + region_tokens

            # concat region tokens to local tokens

            local_tokens = rearrange(local_tokens, 'b (h w) d -> b h w d', h = lh)
            local_tokens = rearrange(local_tokens, 'b (h p1) (w p2) d -> (b h w) (p1 p2) d', p1 = window_size_h, p2 = window_size_w)
            region_tokens = rearrange(region_tokens, 'b n d -> (b n) () d')

            # do self attention on local tokens, along with its regional token

            region_and_local_tokens = torch.cat((region_tokens, local_tokens), dim = 1)
            region_and_local_tokens = attn(region_and_local_tokens, rel_pos_bias = rel_pos_bias) + region_and_local_tokens

            # feedforward

            region_and_local_tokens = ff(region_and_local_tokens) + region_and_local_tokens

            # split back local and regional tokens

            region_tokens, local_tokens = region_and_local_tokens[:, :1], region_and_local_tokens[:, 1:]
            local_tokens = rearrange(local_tokens, '(b h w) (p1 p2) d -> b (h p1 w p2) d', h = lh // window_size_h, w = lw // window_size_w, p1 = window_size_h)
            region_tokens = rearrange(region_tokens, '(b n) () d -> b n d', n = rh * rw)

        local_tokens = rearrange(local_tokens, 'b (h w) c -> b c h w', h = lh, w = lw)
        region_tokens = rearrange(region_tokens, 'b (h w) c -> b c h w', h = rh, w = rw)
        return local_tokens, region_tokens

# classes

class RegionViT(nn.Module):
    def __init__(
        self,
        *,
        dim = (64, 128, 256, 512),
        depth = (2, 2, 8, 2),
        window_size = 7,
        num_classes = 1000,
        tokenize_local_3_conv = False,
        local_patch_size = 4,
        use_peg = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        channels = 3,
    ):
        super().__init__()
        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        assert len(dim) == 4, 'dim needs to be a single value or a tuple of length 4'
        assert len(depth) == 4, 'depth needs to be a single value or a tuple of length 4'

        self.local_patch_size = local_patch_size

        region_patch_size = local_patch_size * window_size
        self.region_patch_size = local_patch_size * window_size

        init_dim, *_, last_dim = dim

        # local and region encoders

        if tokenize_local_3_conv:
            self.local_encoder = nn.Sequential(
                nn.Conv2d(3, init_dim, 3, 2, 1),
                ChanLayerNorm(init_dim),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, 3, 2, 1),
                ChanLayerNorm(init_dim),
                nn.GELU(),
                nn.Conv2d(init_dim, init_dim, 3, 1, 1)
            )
        else:
            self.local_encoder = nn.Conv2d(3, init_dim, 8, 4, 3)

        self.region_encoder = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = region_patch_size, p2 = region_patch_size),
            nn.Conv2d((region_patch_size ** 2) * channels, init_dim, 1)
        )

        # layers

        current_dim = init_dim
        self.layers = nn.ModuleList([])

        for ind, dim, num_layers in zip(range(4), dim, depth):
            not_first = ind != 0
            need_downsample = not_first
            need_peg = not_first and use_peg

            self.layers.append(nn.ModuleList([
                Downsample(current_dim, dim) if need_downsample else nn.Identity(),
                PEG(dim) if need_peg else nn.Identity(),
                R2LTransformer(dim, depth = num_layers, window_size = window_size, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
            ]))

            current_dim = dim

        # final logits

        self.to_logits = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, x):
        *_, h, w = x.shape
        assert divisible_by(h, self.region_patch_size) and divisible_by(w, self.region_patch_size), 'height and width must be divisible by region patch size'
        assert divisible_by(h, self.local_patch_size) and divisible_by(w, self.local_patch_size), 'height and width must be divisible by local patch size'

        local_tokens = self.local_encoder(x)
        region_tokens = self.region_encoder(x)

        for down, peg, transformer in self.layers:
            local_tokens, region_tokens = down(local_tokens), down(region_tokens)
            local_tokens = peg(local_tokens)
            local_tokens, region_tokens = transformer(local_tokens, region_tokens)

        return self.to_logits(region_tokens)

```

## File: vit_pytorch/distill.py

- Extension: .py
- Language: python
- Size: 4561 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from vit_pytorch.vit import ViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.efficient import ViT as EfficientViT

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class DistillMixin:
    def forward(self, img, distill_token = None):
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]

        if distilling:
            distill_tokens = repeat(distill_token, '1 n d -> b n d', b = b)
            x = torch.cat((x, distill_tokens), dim = 1)

        x = self._attend(x)

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        out = self.mlp_head(x)

        if distilling:
            return out, distill_tokens

        return out

class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableT2TViT(DistillMixin, T2TViT):
    def __init__(self, *args, **kwargs):
        super(DistillableT2TViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = T2TViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableEfficientViT(DistillMixin, EfficientViT):
    def __init__(self, *args, **kwargs):
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = EfficientViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        return self.transformer(x)

# knowledge distillation wrapper

class DistillWrapper(Module):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature = 1.,
        alpha = 0.5,
        hard = False,
        mlp_layernorm = False
    ):
        super().__init__()
        assert (isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT))) , 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim) if mlp_layernorm else nn.Identity(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):

        alpha = default(alpha, self.alpha)
        T = default(temperature, self.temperature)

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim = -1),
                F.softmax(teacher_logits / T, dim = -1).detach(),
            reduction = 'batchmean')
            distill_loss *= T ** 2

        else:
            teacher_labels = teacher_logits.argmax(dim = -1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        return loss * (1 - alpha) + distill_loss * alpha

```

## File: vit_pytorch/simple_vit_with_register_tokens.py

- Extension: .py
- Language: python
- Size: 4219 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
"""
    Vision Transformers Need Registers
    https://arxiv.org/abs/2309.16588
"""

import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, num_register_tokens = 4, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch, device = img.shape[0], img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        r = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        x, ps = pack([x, r], 'b * d')

        x = self.transformer(x)

        x, _ = unpack(x, ps, 'b * d')

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

```

## File: vit_pytorch/normalized_vit.py

- Extension: .py
- Language: python
- Size: 6993 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

# for use with parametrize

class L2Norm(Module):
    def __init__(self, dim = -1):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return l2norm(t, dim = self.dim)

class NormLinear(Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim_in = True
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias = False)

        parametrize.register_parametrization(
            self.linear,
            'weight',
            L2Norm(dim = -1 if norm_dim_in else 0)
        )

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x)

# attention and feedforward

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.to_q = NormLinear(dim, dim_inner)
        self.to_k = NormLinear(dim, dim_inner)
        self.to_v = NormLinear(dim, dim_inner)

        self.dropout = dropout

        self.q_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * (dim_head ** 0.25))
        self.k_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * (dim_head ** 0.25))

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in = False)

    def forward(
        self,
        x
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # query key rmsnorm

        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        # scale is 1., as scaling factor is moved to s_qk (dk ^ 0.25) - eq. 16

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.,
            scale = 1.
        )

        out = self.merge_heads(out)
        return self.to_out(out)

class FeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        dim_inner,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim_inner * 2 / 3)

        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = NormLinear(dim, dim_inner)
        self.to_gate = NormLinear(dim, dim_inner)

        self.hidden_scale = nn.Parameter(torch.ones(dim_inner))
        self.gate_scale = nn.Parameter(torch.ones(dim_inner))

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in = False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale
        gate = gate * self.gate_scale * (self.dim ** 0.5)

        hidden = F.silu(gate) * hidden

        hidden = self.dropout(hidden)
        return self.to_out(hidden)

# classes

class nViT(Module):
    """ https://arxiv.org/abs/2410.01131 """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout = 0.,
        channels = 3,
        dim_head = 64,
        residual_lerp_scale_init = None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)
        num_patches = patch_height_dim * patch_width_dim

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            NormLinear(patch_dim, dim, norm_dim_in = False),
        )

        self.abs_pos_emb = NormLinear(dim, num_patches)

        residual_lerp_scale_init = default(residual_lerp_scale_init, 1. / depth)

        # layers

        self.dim = dim
        self.scale = dim ** 0.5

        self.layers = ModuleList([])
        self.residual_lerp_scales = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, dim_inner = mlp_dim, dropout = dropout),
            ]))

            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init / self.scale),
                nn.Parameter(torch.ones(dim) * residual_lerp_scale_init / self.scale),
            ]))

        self.logit_scale = nn.Parameter(torch.ones(num_classes))

        self.to_pred = NormLinear(dim, num_classes)

    @torch.no_grad()
    def norm_weights_(self):
        for module in self.modules():
            if not isinstance(module, NormLinear):
                continue

            normed = module.weight
            original = module.linear.parametrizations.weight.original

            original.copy_(normed)

    def forward(self, images):
        device = images.device

        tokens = self.to_patch_embedding(images)

        seq_len = tokens.shape[-2]
        pos_emb = self.abs_pos_emb.weight[torch.arange(seq_len, device = device)]

        tokens = l2norm(tokens + pos_emb)

        for (attn, ff), (attn_alpha, ff_alpha) in zip(self.layers, self.residual_lerp_scales):

            attn_out = l2norm(attn(tokens))
            tokens = l2norm(tokens.lerp(attn_out, attn_alpha * self.scale))

            ff_out = l2norm(ff(tokens))
            tokens = l2norm(tokens.lerp(ff_out, ff_alpha * self.scale))

        pooled = reduce(tokens, 'b n d -> b d', 'mean')

        logits = self.to_pred(pooled)
        logits = logits * self.logit_scale * self.scale

        return logits

# quick test

if __name__ == '__main__':

    v = nViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
    )

    img = torch.randn(4, 3, 256, 256)
    logits = v(img) # (4, 1000)
    assert logits.shape == (4, 1000)

```

## File: vit_pytorch/vivit.py

- Extension: .py
- Language: python
- Size: 7599 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class FactorizedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        b, f, n, _ = x.shape
        for spatial_attn, temporal_attn, ff in self.layers:
            x = rearrange(x, 'b f n d -> (b f) n d')
            x = spatial_attn(x) + x
            x = rearrange(x, '(b f) n d -> (b n) f d', b=b, f=f)
            x = temporal_attn(x) + x
            x = ff(x) + x
            x = rearrange(x, '(b n) f d -> b f n d', b=b, n=n)

        return self.norm(x)

class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        variant = 'factorized_encoder',
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = (frames // frame_patch_size)

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        if variant == 'factorized_encoder':
            self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
            self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
            self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)
        elif variant == 'factorized_self_attention':
            assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for factorized self-attention'
            self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        self.variant = variant

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape

        x = x + self.pos_embedding[:, :f, :n]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)

        if self.variant == 'factorized_encoder':
            x = rearrange(x, 'b f n d -> (b f) n d')

            # attend across space

            x = self.spatial_transformer(x)
            x = rearrange(x, '(b f) n d -> b f n d', b = b)

            # excise out the spatial cls tokens or average pool for temporal attention

            x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

            # append temporal CLS tokens

            if exists(self.temporal_cls_token):
                temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

                x = torch.cat((temporal_cls_tokens, x), dim = 1)
            

            # attend across time

            x = self.temporal_transformer(x)

            # excise out temporal cls token or average pool

            x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        elif self.variant == 'factorized_self_attention':
            x = self.factorized_transformer(x)
            x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/crossformer.py

- Extension: .py
- Language: python
- Size: 8347 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

# helpers

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# cross embed layer

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_sizes,
        stride = 2
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

# dynamic positional bias

def DynamicPositionBias(dim):
    return nn.Sequential(
        nn.Linear(2, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.LayerNorm(dim),
        nn.ReLU(),
        nn.Linear(dim, 1),
        Rearrange('... () -> ...')
    )

# transformer classes

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Conv2d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv2d(dim * mult, dim, 1)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        attn_type,
        window_size,
        dim_head = 32,
        dropout = 0.
    ):
        super().__init__()
        assert attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

        # positions

        self.dpb = DynamicPositionBias(dim // 4)

        # calculate and store indices for retrieving bias

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        *_, height, width, heads, wsz, device = *x.shape, self.heads, self.window_size, x.device

        # prenorm

        x = self.norm(x)

        # rearrange for short or long distance attention

        if self.attn_type == 'short':
            x = rearrange(x, 'b d (h s1) (w s2) -> (b h w) d s1 s2', s1 = wsz, s2 = wsz)
        elif self.attn_type == 'long':
            x = rearrange(x, 'b d (l1 h) (l2 w) -> (b h w) d l1 l2', l1 = wsz, l2 = wsz)

        # queries / keys / values

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add dynamic positional bias

        pos = torch.arange(-wsz, wsz + 1, device = device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        rel_pos = rearrange(rel_pos, 'c i j -> (i j) c')
        biases = self.dpb(rel_pos.float())
        rel_pos_bias = biases[self.rel_pos_indices]

        sim = sim + rel_pos_bias

        # attend

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # merge heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = wsz, y = wsz)
        out = self.to_out(out)

        # rearrange back for long or short distance attention

        if self.attn_type == 'short':
            out = rearrange(out, '(b h w) d s1 s2 -> b d (h s1) (w s2)', h = height // wsz, w = width // wsz)
        elif self.attn_type == 'long':
            out = rearrange(out, '(b h w) d l1 l2 -> b d (l1 h) (l2 w)', h = height // wsz, w = width // wsz)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        local_window_size,
        global_window_size,
        depth = 4,
        dim_head = 32,
        attn_dropout = 0.,
        ff_dropout = 0.,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, attn_type = 'short', window_size = local_window_size, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
                Attention(dim, attn_type = 'long', window_size = global_window_size, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout)
            ]))

    def forward(self, x):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x

# classes

class CrossFormer(nn.Module):
    def __init__(
        self,
        *,
        dim = (64, 128, 256, 512),
        depth = (2, 2, 8, 2),
        global_window_size = (8, 4, 2, 1),
        local_window_size = 7,
        cross_embed_kernel_sizes = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
        cross_embed_strides = (4, 2, 2, 2),
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        channels = 3
    ):
        super().__init__()

        dim = cast_tuple(dim, 4)
        depth = cast_tuple(depth, 4)
        global_window_size = cast_tuple(global_window_size, 4)
        local_window_size = cast_tuple(local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        # dimensions

        last_dim = dim[-1]
        dims = [channels, *dim]
        dim_in_and_out = tuple(zip(dims[:-1], dims[1:]))

        # layers

        self.layers = nn.ModuleList([])

        for (dim_in, dim_out), layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim_in_and_out, depth, global_window_size, local_window_size, cross_embed_kernel_sizes, cross_embed_strides):
            self.layers.append(nn.ModuleList([
                CrossEmbedLayer(dim_in, dim_out, cel_kernel_sizes, stride = cel_stride),
                Transformer(dim_out, local_window_size = local_wsz, global_window_size = global_wsz, depth = layers, attn_dropout = attn_dropout, ff_dropout = ff_dropout)
            ]))

        # final logits

        self.to_logits = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, x):
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)

        return self.to_logits(x)

```

## File: vit_pytorch/vit_3d.py

- Extension: .py
- Language: python
- Size: 4351 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/efficient.py

- Extension: .py
- Language: python
- Size: 1813 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        image_size_h, image_size_w = pair(image_size)
        assert image_size_h % patch_size == 0 and image_size_w % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size_h // patch_size) * (image_size_w // patch_size)
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

```

## File: vit_pytorch/local_vit.py

- Extension: .py
- Language: python
- Size: 4883 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from math import sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class ExcludeCLS(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        cls_token, x = x[:, :1], x[:, 1:]
        x = self.fn(x, **kwargs)
        return torch.cat((cls_token, x), dim = 1)

# feed forward related classes

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.Hardswish(),
            DepthWiseConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        h = w = int(sqrt(x.shape[-2]))
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                ExcludeCLS(Residual(FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# main class

class LocalViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        return self.mlp_head(x[:, 0])

```

## File: vit_pytorch/max_vit_with_registers.py

- Extension: .py
- Language: python
- Size: 9892 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(dim * mult)
    return Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    )

# MBConv

class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        num_registers = 1
    ):
        super().__init__()
        assert num_registers > 0
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        num_rel_pos_bias = (2 * window_size - 1) ** 2

        self.rel_pos_bias = nn.Embedding(num_rel_pos_bias + 1, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        rel_pos_indices = F.pad(rel_pos_indices, (num_registers, 0, num_registers, 0), value = num_rel_pos_bias)
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        device, h, bias_indices = x.device, self.heads, self.rel_pos_indices

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(bias_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads out

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MaxViT(Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3,
        num_register_tokens = 4
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'
        assert num_register_tokens > 0

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # window size

        self.window_size = window_size

        self.register_tokens = nn.ParameterList([])

        # iterate through stages

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                conv = MBConv(
                    stage_dim_in,
                    layer_dim,
                    downsample = is_first,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                )

                block_attn = Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)
                block_ff = FeedForward(dim = layer_dim, dropout = dropout)

                grid_attn = Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = window_size, num_registers = num_register_tokens)
                grid_ff = FeedForward(dim = layer_dim, dropout = dropout)

                register_tokens = nn.Parameter(torch.randn(num_register_tokens, layer_dim))

                self.layers.append(ModuleList([
                    conv,
                    ModuleList([block_attn, block_ff]),
                    ModuleList([grid_attn, grid_ff])
                ]))

                self.register_tokens.append(register_tokens)

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        b, w = x.shape[0], self.window_size

        x = self.conv_stem(x)

        for (conv, (block_attn, block_ff), (grid_attn, grid_ff)), register_tokens in zip(self.layers, self.register_tokens):
            x = conv(x)

            # block-like attention

            x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)

            # prepare register tokens

            r = repeat(register_tokens, 'n d -> b x y n d', b = b, x = x.shape[1],y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = block_attn(x) + x
            x = block_ff(x) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')

            r = unpack_one(r, register_batch_ps, '* n d')

            # grid-like attention

            x = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)

            # prepare register tokens

            r = reduce(r, 'b x y n d -> b n d', 'mean')
            r = repeat(r, 'b n d -> b x y n d', x = x.shape[1], y = x.shape[2])
            r, register_batch_ps = pack_one(r, '* n d')

            x, window_ps = pack_one(x, 'b x y * d')
            x, batch_ps  = pack_one(x, '* n d')
            x, register_ps = pack([r, x], 'b * d')

            x = grid_attn(x) + x

            r, x = unpack(x, register_ps, 'b * d')

            x = grid_ff(x) + x

            x = unpack_one(x, batch_ps, '* n d')
            x = unpack_one(x, window_ps, 'b x y * d')
            x = rearrange(x, 'b x y w1 w2 d -> b d (w1 x) (w2 y)')

        return self.mlp_head(x)

```

## File: vit_pytorch/vit_1d.py

- Extension: .py
- Language: python
- Size: 3880 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_head(cls_tokens)

if __name__ == '__main__':

    v = ViT(
        seq_len = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    time_series = torch.randn(4, 3, 256)
    logits = v(time_series) # (4, 1000)

```

## File: vit_pytorch/cct_3d.py

- Extension: .py
- Language: python
- Size: 12590 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# CCT Models

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']


def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = default(stride, max(1, (kernel_size // 2) - 1))
    padding = default(padding, max(1, (kernel_size // 2)))

    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)

# positional

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

# modules

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // self.heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q = q * self.scale

        attn = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        return self.proj_drop(self.proj(x))


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output

class Tokenizer(nn.Module):
    def __init__(
        self,
        frame_kernel_size,
        kernel_size,
        stride,
        padding,
        frame_stride=1,
        frame_padding=None,
        frame_pooling_stride=1,
        frame_pooling_kernel_size=1,
        frame_pooling_padding=None,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=3,
        n_output_channels=64,
        in_planes=64,
        activation=None,
        max_pool=True,
        conv_bias=False
    ):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        if frame_padding is None:
            frame_padding = frame_kernel_size // 2

        if frame_pooling_padding is None:
            frame_pooling_padding = frame_pooling_kernel_size // 2

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(chan_in, chan_out,
                          kernel_size=(frame_kernel_size, kernel_size, kernel_size),
                          stride=(frame_stride, stride, stride),
                          padding=(frame_padding, padding, padding), bias=conv_bias),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool3d(kernel_size=(frame_pooling_kernel_size, pooling_kernel_size, pooling_kernel_size),
                             stride=(frame_pooling_stride, pooling_stride, pooling_stride),
                             padding=(frame_pooling_padding, pooling_padding, pooling_padding)) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, frames=8, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, frames, height, width))).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        return rearrange(x, 'b c f h w -> b (f h w) c')

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        seq_pool=True,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout_rate=0.1,
        attention_dropout=0.1,
        stochastic_depth_rate=0.1,
        positional_embedding='sine',
        sequence_length=None,
        *args, **kwargs
    ):
        super().__init__()
        assert positional_embedding in {'sine', 'learnable', 'none'}

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
            nn.init.trunc_normal_(self.positional_emb, std = 0.2)
        else:
            self.register_buffer('positional_emb', sinusoidal_embedding(sequence_length, embedding_dim))

        self.dropout = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b = x.shape[0]

        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_token, x), dim=1)

        if exists(self.positional_emb):
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim = 1), x)
        else:
            x = x[:, 0]

        return self.fc(x)

# CCT Main model

class CCT(nn.Module):
    def __init__(
        self,
        img_size=224,
        num_frames=8,
        embedding_dim=768,
        n_input_channels=3,
        n_conv_layers=1,
        frame_stride=1,
        frame_kernel_size=3,
        frame_padding=None,
        frame_pooling_kernel_size=1,
        frame_pooling_stride=1,
        frame_pooling_padding=None,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        *args, **kwargs
    ):
        super().__init__()
        img_height, img_width = pair(img_size)

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            frame_stride=frame_stride,
            frame_kernel_size=frame_kernel_size,
            frame_padding=frame_padding,
            frame_pooling_stride=frame_pooling_stride,
            frame_pooling_kernel_size=frame_pooling_kernel_size,
            frame_pooling_padding=frame_pooling_padding,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False
        )

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(
                n_channels=n_input_channels,
                frames=num_frames,
                height=img_height,
                width=img_width
            ),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)

```

## File: vit_pytorch/look_vit.py

- Extension: .py
- Language: python
- Size: 8537 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

# simple vit sinusoidal pos emb

def posemb_sincos_2d(t, temperature = 10000):
    h, w, d, device = *t.shape[1:], t.device
    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (d % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(d // 4, device = device) / (d // 4 - 1)
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pos = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)

    return pos.float()

# bias-less layernorm with unit offset trick (discovered by Ohad Rubin)

class LayerNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        normed = self.ln(x)
        return normed * (self.gamma + 1)

# mlp

def MLP(dim, factor = 4, dropout = 0.):
    hidden_dim = int(dim * factor)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        cross_attend = False,
        reuse_attention = False
    ):
        super().__init__()
        inner_dim = dim_head *  heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.reuse_attention = reuse_attention
        self.cross_attend = cross_attend

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.norm = LayerNorm(dim) if not reuse_attention else nn.Identity()
        self.norm_context = LayerNorm(dim) if cross_attend else nn.Identity()

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False) if not reuse_attention else None
        self.to_k = nn.Linear(dim, inner_dim, bias = False) if not reuse_attention else None
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        return_qk_sim = False,
        qk_sim = None
    ):
        x = self.norm(x)

        assert not (exists(context) ^ self.cross_attend)

        if self.cross_attend:
            context = self.norm_context(context)
        else:
            context = x

        v = self.to_v(context)
        v = self.split_heads(v)

        if not self.reuse_attention:
            qk = (self.to_q(x), self.to_k(context))
            q, k = tuple(self.split_heads(t) for t in qk)

            q = q * self.scale
            qk_sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        else:
            assert exists(qk_sim), 'qk sim matrix must be passed in for reusing previous attention'

        attn = self.attend(qk_sim)
        attn = self.dropout(attn)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = self.to_out(out)

        if not return_qk_sim:
            return out

        return out, qk_sim

# LookViT

class LookViT(Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        num_classes,
        depth = 3,
        patch_size = 16,
        heads = 8,
        mlp_factor = 4,
        dim_head = 64,
        highres_patch_size = 12,
        highres_mlp_factor = 4,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        patch_conv_kernel_size = 7,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert divisible_by(image_size, highres_patch_size)
        assert divisible_by(image_size, patch_size)
        assert patch_size > highres_patch_size, 'patch size of the main vision transformer should be smaller than the highres patch sizes (that does the `lookup`)'
        assert not divisible_by(patch_conv_kernel_size, 2)

        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size

        kernel_size = patch_conv_kernel_size
        patch_dim = (highres_patch_size * highres_patch_size) * channels

        self.to_patches = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = highres_patch_size, p2 = highres_patch_size),
            nn.Conv2d(patch_dim, dim, kernel_size, padding = kernel_size // 2),
            Rearrange('b c h w -> b h w c'),
            LayerNorm(dim),
        )

        # absolute positions

        num_patches = (image_size // highres_patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))

        # lookvit blocks

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                MLP(dim = dim, factor = mlp_factor, dropout = dropout),
                Attention(dim = dim, dim_head = cross_attn_dim_head, heads = cross_attn_heads, dropout = dropout, cross_attend = True),
                Attention(dim = dim, dim_head = cross_attn_dim_head, heads = cross_attn_heads, dropout = dropout, cross_attend = True, reuse_attention = True),
                LayerNorm(dim),
                MLP(dim = dim, factor = highres_mlp_factor, dropout = dropout)
            ]))

        self.layers = layers

        self.norm = LayerNorm(dim)
        self.highres_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, num_classes, bias = False)

    def forward(self, img):
        assert img.shape[-2:] == (self.image_size, self.image_size)

        # to patch tokens and positions

        highres_tokens = self.to_patches(img)
        size = highres_tokens.shape[-2]

        pos_emb = posemb_sincos_2d(highres_tokens)
        highres_tokens = highres_tokens + rearrange(pos_emb, '(h w) d -> h w d', h = size)

        tokens = F.interpolate(
            rearrange(highres_tokens, 'b h w d -> b d h w'),
            img.shape[-1] // self.patch_size,
            mode = 'bilinear'
        )

        tokens = rearrange(tokens, 'b c h w -> b (h w) c')
        highres_tokens = rearrange(highres_tokens, 'b h w c -> b (h w) c')

        # attention and feedforwards

        for attn, mlp, lookup_cross_attn, highres_attn, highres_norm, highres_mlp in self.layers:

            # main tokens cross attends (lookup) on the high res tokens

            lookup_out, qk_sim = lookup_cross_attn(tokens, highres_tokens, return_qk_sim = True)  # return attention as they reuse the attention matrix
            tokens = lookup_out + tokens

            tokens = attn(tokens) + tokens
            tokens = mlp(tokens) + tokens

            # attention-reuse

            qk_sim = rearrange(qk_sim, 'b h i j -> b h j i') # transpose for reverse cross attention

            highres_tokens = highres_attn(highres_tokens, tokens, qk_sim = qk_sim) + highres_tokens
            highres_tokens = highres_norm(highres_tokens)

            highres_tokens = highres_mlp(highres_tokens) + highres_tokens

        # to logits

        tokens = self.norm(tokens)
        highres_tokens = self.highres_norm(highres_tokens)

        tokens = reduce(tokens, 'b n d -> b d', 'mean')
        highres_tokens = reduce(highres_tokens, 'b n d -> b d', 'mean')

        return self.to_logits(tokens + highres_tokens)

# main

if __name__ == '__main__':
    v = LookViT(
        image_size = 256,
        num_classes = 1000,
        dim = 512,
        depth = 2,
        heads = 8,
        dim_head = 64,
        patch_size = 32,
        highres_patch_size = 8,
        highres_mlp_factor = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        dropout = 0.1
    ).cuda()

    img = torch.randn(2, 3, 256, 256).cuda()
    pred = v(img)

    assert pred.shape == (2, 1000)

```

## File: vit_pytorch/scalable_vit.py

- Extension: .py
- Language: python
- Size: 9585 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
from functools import partial
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 3, stride = 2, padding = 1)

    def forward(self, x):
        return self.conv(x)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1)

    def forward(self, x):
        return self.proj(x) + x

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor = 4, dropout = 0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# attention

class ScalableSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_key = 32,
        dim_value = 32,
        dropout = 0.,
        reduction_factor = 1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = ChanLayerNorm(dim)
        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, reduction_factor, stride = reduction_factor, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, reduction_factor, stride = reduction_factor, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        height, width, heads = *x.shape[-2:], self.heads

        x = self.norm(x)

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # similarity

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attention

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # aggregate values

        out = torch.matmul(attn, v)

        # merge back heads

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = height, y = width)
        return self.to_out(out)

class InteractiveWindowedSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        heads = 8,
        dim_key = 32,
        dim_value = 32,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_key ** -0.5
        self.window_size = window_size
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.norm = ChanLayerNorm(dim)
        self.local_interactive_module = nn.Conv2d(dim_value * heads, dim_value * heads, 3, padding = 1)

        self.to_q = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_key * heads, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value * heads, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(dim_value * heads, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        height, width, heads, wsz = *x.shape[-2:], self.heads, self.window_size

        x = self.norm(x)

        wsz_h, wsz_w = default(wsz, height), default(wsz, width)
        assert (height % wsz_h) == 0 and (width % wsz_w) == 0, f'height ({height}) or width ({width}) of feature map is not divisible by the window size ({wsz_h}, {wsz_w})'

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # get output of LIM

        local_out = self.local_interactive_module(v)

        # divide into window (and split out heads) for efficient self attention

        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d', h = heads, w1 = wsz_h, w2 = wsz_w), (q, k, v))

        # similarity

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # attention

        attn = self.attend(dots)
        attn = self.dropout(attn)

        # aggregate values

        out = torch.matmul(attn, v)

        # reshape the windows back to full feature map (and merge heads)

        out = rearrange(out, '(b x y) h (w1 w2) d -> b (h d) (x w1) (y w2)', x = height // wsz_h, y = width // wsz_w, w1 = wsz_h, w2 = wsz_w)

        # add LIM output 

        out = out + local_out

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        ff_expansion_factor = 4,
        dropout = 0.,
        ssa_dim_key = 32,
        ssa_dim_value = 32,
        ssa_reduction_factor = 1,
        iwsa_dim_key = 32,
        iwsa_dim_value = 32,
        iwsa_window_size = None,
        norm_output = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for ind in range(depth):
            is_first = ind == 0

            self.layers.append(nn.ModuleList([
                ScalableSelfAttention(dim, heads = heads, dim_key = ssa_dim_key, dim_value = ssa_dim_value, reduction_factor = ssa_reduction_factor, dropout = dropout),
                FeedForward(dim, expansion_factor = ff_expansion_factor, dropout = dropout),
                PEG(dim) if is_first else None,
                FeedForward(dim, expansion_factor = ff_expansion_factor, dropout = dropout),
                InteractiveWindowedSelfAttention(dim, heads = heads, dim_key = iwsa_dim_key, dim_value = iwsa_dim_value, window_size = iwsa_window_size, dropout = dropout)
            ]))

        self.norm = ChanLayerNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for ssa, ff1, peg, iwsa, ff2 in self.layers:
            x = ssa(x) + x
            x = ff1(x) + x

            if exists(peg):
                x = peg(x)

            x = iwsa(x) + x
            x = ff2(x) + x

        return self.norm(x)

class ScalableViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        reduction_factor,
        window_size = None,
        iwsa_dim_key = 32,
        iwsa_dim_value = 32,
        ssa_dim_key = 32,
        ssa_dim_value = 32,
        ff_expansion_factor = 4,
        channels = 3,
        dropout = 0.
    ):
        super().__init__()
        self.to_patches = nn.Conv2d(channels, dim, 7, stride = 4, padding = 3)

        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        num_stages = len(depth)
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))

        hyperparams_per_stage = [
            heads,
            ssa_dim_key,
            ssa_dim_value,
            reduction_factor,
            iwsa_dim_key,
            iwsa_dim_value,
            window_size,
        ]

        hyperparams_per_stage = list(map(partial(cast_tuple, length = num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        self.layers = nn.ModuleList([])

        for ind, (layer_dim, layer_depth, layer_heads, layer_ssa_dim_key, layer_ssa_dim_value, layer_ssa_reduction_factor, layer_iwsa_dim_key, layer_iwsa_dim_value, layer_window_size) in enumerate(zip(dims, depth, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            self.layers.append(nn.ModuleList([
                Transformer(dim = layer_dim, depth = layer_depth, heads = layer_heads, ff_expansion_factor = ff_expansion_factor, dropout = dropout, ssa_dim_key = layer_ssa_dim_key, ssa_dim_value = layer_ssa_dim_value, ssa_reduction_factor = layer_ssa_reduction_factor, iwsa_dim_key = layer_iwsa_dim_key, iwsa_dim_value = layer_iwsa_dim_value, iwsa_window_size = layer_window_size, norm_output = not is_last),
                Downsample(layer_dim, layer_dim * 2) if not is_last else None
            ]))

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, img):
        x = self.to_patches(img)

        for transformer, downsample in self.layers:
            x = transformer(x)

            if exists(downsample):
                x = downsample(x)

        return self.mlp_head(x)

```

## File: vit_pytorch/learnable_memory_vit.py

- Extension: .py
- Language: python
- Size: 7308 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# controlling freezing of layers

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask = None, memories = None):
        x = self.norm(x)

        x_kv = x # input for key / values projection

        if exists(memories):
            # add memories to key / values if it is passed in
            memories = repeat(memories, 'n d -> b n d', b = x.shape[0]) if memories.ndim == 2 else memories
            x_kv = torch.cat((x_kv, memories), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, attn_mask = None, memories = None):
        for ind, (attn, ff) in enumerate(self.layers):
            layer_memories = memories[ind] if exists(memories) else None

            x = attn(x, attn_mask = attn_mask, memories = layer_memories) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def img_to_tokens(self, img):
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = x.shape[0])
        x = torch.cat((cls_tokens, x), dim = 1)

        x += self.pos_embedding
        x = self.dropout(x)
        return x

    def forward(self, img):
        x = self.img_to_tokens(img)        

        x = self.transformer(x)

        cls_tokens = x[:, 0]
        return self.mlp_head(cls_tokens)

# adapter with learnable memories per layer, memory CLS token, and learnable adapter head

class Adapter(nn.Module):
    def __init__(
        self,
        *,
        vit,
        num_memories_per_layer = 10,
        num_classes = 2,   
    ):
        super().__init__()
        assert isinstance(vit, ViT)

        # extract some model variables needed

        dim = vit.cls_token.shape[-1]
        layers = len(vit.transformer.layers)
        num_patches = vit.pos_embedding.shape[-2]

        self.vit = vit

        # freeze ViT backbone - only memories will be finetuned

        freeze_all_layers_(vit)

        # learnable parameters

        self.memory_cls_token = nn.Parameter(torch.randn(dim))
        self.memories_per_layer = nn.Parameter(torch.randn(layers, num_memories_per_layer, dim))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        # specialized attention mask to preserve the output of the original ViT
        # it allows the memory CLS token to attend to all other tokens (and the learnable memory layer tokens), but not vice versa        

        attn_mask = torch.ones((num_patches, num_patches), dtype = torch.bool)
        attn_mask = F.pad(attn_mask, (1, num_memories_per_layer), value = False)  # main tokens cannot attend to learnable memories per layer
        attn_mask = F.pad(attn_mask, (0, 0, 1, 0), value = True)                  # memory CLS token can attend to everything
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, img):
        b = img.shape[0]

        tokens = self.vit.img_to_tokens(img)

        # add task specific memory tokens

        memory_cls_tokens = repeat(self.memory_cls_token, 'd -> b 1 d', b = b)
        tokens = torch.cat((memory_cls_tokens, tokens), dim = 1)        

        # pass memories along with image tokens through transformer for attending

        out = self.vit.transformer(tokens, memories = self.memories_per_layer, attn_mask = self.attn_mask)

        # extract memory CLS tokens

        memory_cls_tokens = out[:, 0]

        # pass through task specific adapter head

        return self.mlp_head(memory_cls_tokens)

```

## File: vit_pytorch/simple_vit_with_patch_dropout.py

- Extension: .py
- Language: python
- Size: 4652 bytes
- Created: 2025-03-17 07:13:37
- Modified: 2025-03-17 07:13:37

### Code

```python
import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, patch_dropout = 0.5):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.patch_dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

```

