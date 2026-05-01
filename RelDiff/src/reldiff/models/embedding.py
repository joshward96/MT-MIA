import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# https://github.com/yandex-research/rtdl-num-embeddings


def normalize_emb(emb, dim):
    return F.normalize(emb, dim=dim, eps=1e-20)


class PositionalEmbedder(nn.Module):
    """
    Positional embedding layer for encoding continuous features.
    Adapted from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py#L61
    """

    def __init__(self, dim, num_features, trainable=False, freq_init_scale=0.01):
        super().__init__()
        assert (dim % 2) == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(
            torch.randn(1, num_features, self.half_dim), requires_grad=trainable
        )
        self.sigma = freq_init_scale
        bound = self.sigma * 3
        nn.init.trunc_normal_(self.weights, 0.0, self.sigma, a=-bound, b=bound)

    def forward(self, x):
        x = rearrange(x, "b f -> b f 1")
        freqs = x * self.weights * 2 * torch.pi
        fourier = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fourier


class NLinear(nn.Module):
    """N separate linear layers for N separate features
    adapted from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py#L61
    x has typically 3 dimensions: (batch, features, embedding dim)
    """

    def __init__(self, in_dim, out_dim, n):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(n, out_dim))
        d_in_rsqrt = 1 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x):
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        x += self.bias
        return x


class ContEmbedder(nn.Module):
    """
    Embedding layer for continuous features that utilizes Fourier features.
    """

    def __init__(self, dim, num_features, freq_init_scale=0.01):
        super().__init__()
        assert (dim % 2) == 0
        self.pos_emb = PositionalEmbedder(
            2 * dim, num_features, trainable=True, freq_init_scale=freq_init_scale
        )
        self.nlinear = NLinear(2 * dim, dim, num_features)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.pos_emb(x)
        x = self.nlinear(x)
        return self.act(x)


class FourierTokenizer(nn.Module):
    def __init__(
        self,
        d_numerical,
        categories,
        d_token,
        bias,
        freq_init_scale=0.01,
    ):
        super().__init__()
        self.dim = torch.tensor(d_token)
        self.n_tokens = d_numerical + 1
        self.bias = bias
        if categories is not None:
            category_offsets = torch.tensor([0] + list(categories[:-1])).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.cat_weight = nn.Parameter(torch.Tensor(sum(categories), d_token))
            nn.init.kaiming_uniform_(self.cat_weight, a=math.sqrt(5))

            if bias:
                self.cat_bias = nn.Parameter(torch.zeros(len(categories), d_token))

            self.n_tokens += sum(categories)

        # take [CLS] token into account
        self.num_emb = ContEmbedder(
            d_token, d_numerical + 1, freq_init_scale=freq_init_scale
        )

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )

        x = self.num_emb(x_num)

        if x_cat is not None:
            for i, (start, end) in enumerate(
                zip(
                    self.category_offsets,
                    torch.cat(
                        [
                            self.category_offsets[1:],
                            torch.tensor([x_cat.shape[1]], device=x_cat.device),
                        ]
                    ),
                )
            ):
                # NOTE: This could be done more efficiently if we are not using
                # one-hot encodings. (Only applicable without learnable noise schedule.)
                x_category = (
                    x_cat[:, start:end].unsqueeze(1) @ self.cat_weight[start:end][None]
                )
                if self.bias:
                    x_category += self.cat_bias[i][None]
                x_category = normalize_emb(x_category, dim=2) * self.dim.sqrt()
                x = torch.cat(
                    [
                        x,
                        x_category,
                    ],
                    dim=1,
                )

        return x
