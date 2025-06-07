import torch
from torch import randn, rand, Tensor

if __name__ == "__main__":
    seq_len = 10
    dim_model = 8

    x = randn(seq_len, dim_model)
    print(f'x: {x}\nx shape: {x.shape}\nx size: {x.size()}')

    