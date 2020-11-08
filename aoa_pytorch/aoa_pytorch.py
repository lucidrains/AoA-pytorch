import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class AttentionOnAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        aoa_dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)

        self.aoa = nn.Sequential(
            nn.Linear(2 * inner_dim, 2 * dim),
            nn.GLU(),
            nn.Dropout(aoa_dropout)
        )

    def forward(self, x, context = None):
        h = self.heads

        q_ = self.to_q(x)

        context = default(context, x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q_, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # attention
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # weighted average of values
        attn_out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # concat heads
        out = rearrange(attn_out, 'b h n d -> b n (h d)', h = h)

        # attention on attention
        out = self.aoa(torch.cat((out, q_), dim = -1))
        return out
