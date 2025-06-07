"""
ref:
https://www.53ai.com/news/qianyanjishu/1844.html
"""
import torch
from torch import nn, Tensor
from math import sqrt

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.norm_factor = 1/sqrt(self.depth)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)


    def split_heads(self, x:Tensor):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.depth).transpose(1,2)
    
    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask=None):
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        QK_prod = Q @ K.transpose(-2,-1)
        scaled_QK_prod:Tensor = QK_prod * self.norm_factor

        if mask is not None:
            scaled_QK_prod += scaled_QK_prod.masked_fill(mask==0, -1e9)

        attn_sftmax = torch.softmax(scaled_QK_prod, dim=-1)

        context_v = attn_sftmax @ V

        batch_size, _, seq_len, dim_v = context_v.size()

        context_v = context_v.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        context_v = self.output_linear(context_v)

        return context_v


