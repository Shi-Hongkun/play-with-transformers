import torch
from torch import nn as nn
from torch import Tensor
import math

def dot_product_method_1_torch_magic(Q: Tensor, K: Tensor) -> Tensor:
    # Q shape: batch_size * seq_len * d_k
    # K shape: batch_size * seq_len * d_k
    # output shape: batch_size * seq_len * seq_len
    # TODO: Learnthe difference between transpose(-2, -1) and transpose(-1, -2)
    return Q @ K.transpose(-2, -1)

def dot_product_method_2_torch_matmul(Q: Tensor, K: Tensor) -> Tensor:
    # Q shape: batch_size * seq_len * d_k
    # K shape: batch_size * seq_len * d_k
    # output shape: batch_size * seq_len * seq_len
    # TODO: Learnthe difference between transpose(-2, -1) and transpose(-1, -2)
    return torch.matmul(Q, K.transpose(-2, -1))

def dot_product_method_3_torch_bmm(Q: Tensor, K: Tensor) -> Tensor:
    # cons of bmm: only works for 3D tensors
    
    # Q shape: batch_size * seq_len * d_k
    # K shape: batch_size * seq_len * d_k
    # output shape: batch_size * seq_len * seq_len
    # TODO: Learnthe difference between transpose(1,2) and transpose(2,1)
    return torch.bmm(Q, K.transpose(1, 2))
                                    
def dot_product_method_4_torch_einsum(Q: Tensor, K: Tensor) -> Tensor:
    # Q shape: batch_size * seq_len * d_k
    # K shape: batch_size * seq_len * d_k
    # output shape: batch_size * seq_len * seq_len
    return torch.einsum('bsh,bth->bst', Q, K)

def dot_product_method_5_torch_permute(Q: Tensor, K: Tensor) -> Tensor:
    #cons of permute: need to specify all dimensions

    # Q shape: batch_size * seq_len * d_k
    # K shape: batch_size * seq_len * d_k
    # output shape: batch_size * seq_len * seq_len
    return torch.matmul(Q, K.permute(0, 2, 1))

class SelfAttention(nn.Module):
    # input shape: batch_size * seq_len * input_dim
    # Q vector shape: batch_size * seq_len * d_k
    # K vector shape: batch_size * seq_len * d_k
    # V vector shape: batch_size * seq_len * d_v

    # since q, k, must have the same dimension for dot product, 
    # so we need to project q, k, v to the same dimension
    # Wq, Wk, Wv are the projection matrices
    # Wq shape: batch_size * input_dim * d_k
    # Wk shape: batch_size * input_dim * d_k
    # Wv shape: batch_size * input_dim * d_v
    
    
    def __init__(self, input_dim: int, dim_k: int, dim_v: int):
        super(SelfAttention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1/math.sqrt(dim_k)

    def forward(self, x: Tensor) -> Tensor:
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        dot_product_q_k: Tensor = dot_product_method_1_torch_magic(Q, K)
        scaled_prod: Tensor = self._norm_fact * dot_product_q_k
        # shape: batch_size * seq_len * seq_len

        # Attention weights
        normed_prod: Tensor = torch.softmax(scaled_prod, dim=-1)
        # The difference between torch.softmax and nn.Softmax:
        # torch.softmax is more memory efficient and cleaner code
        # nn.Softmax requires module initialization, and it's a class just like a nn.Linear, who need to forward()
        # Note: softmax dim =-1 means applying softmax on the last dimension 
        # For example, for a 3x3 matrix:
        # [[1, 2, 3],
        #  [4, 5, 6],
        #  [7, 8, 9]]
        # The softmax of each row would be:
        # [[0.0900, 0.2447, 0.6652],    
        #  [0.0900, 0.2447, 0.6652],
        #  [0.0900, 0.2447, 0.6652]]
        # Note: the sum of each row is 1

        output: Tensor = normed_prod @ V
        # shape: batch_size * seq_len * d_v 

        return output
    
if __name__ == "__main__":
    # Test the dot product methods
    X: Tensor = torch.randn(4,3,2)  # batch_size * seq_len * input_dim
    print("X shape:", X.shape)
    
    self_attn = SelfAttention(input_dim=2, dim_k=4, dim_v=5)
    output = self_attn(X)
    print("Output shape:", output.shape)
    print("Output:", output)
        
