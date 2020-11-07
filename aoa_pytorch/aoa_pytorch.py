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
        heads = 8
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_out = nn.Linear(dim_head, dim_head, bias = False)
        self.attn_out = nn.Linear(dim_head, dim_head, bias = False)
        self.out_bias = nn.Parameter(torch.zeros(1, 1, dim_head))

        self.q_gate = nn.Linear(dim_head, dim_head, bias = False)
        self.attn_gate = nn.Linear(dim_head, dim_head, bias = False)
        self.gate_bias = nn.Parameter(torch.zeros(1, 1, dim_head))

    def forward(self, x, context = None):
        h = self.heads

        q = self.to_q(x)

        context = default(context, x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # attention
        attn = dots.softmax(dim = -1)
        attn_out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # as described in equations (2) and (3) in paper
        I = self.q_out(q) + self.attn_out(attn_out) + self.out_bias
        G = self.q_gate(q) + self.attn_gate(attn_out) + self.gate_bias

        # attention on attention
        out = I * G.sigmoid()

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return out
