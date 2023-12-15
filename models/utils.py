import torch
EOS = 1e-10


def normalize_adj(adj, mode='sym'):
    assert len(adj.shape) in [2, 3]
    if mode == "sym":#对称归一化
        #计算邻接矩阵每一行的绝对值之和（按行求和），然后取其平方根的倒数。这个结果会被保存在 inv_sqrt_degree 变量中
        inv_sqrt_degree = 1. / (torch.sqrt(adj.abs().sum(dim=-1, keepdim=False)) + EOS)
        if len(adj.shape) == 3:#如果输入邻接矩阵是三维的，则进行适当的广播操作，以使 inv_sqrt_degree 与邻接矩阵相乘，从而实现对称归一化。
            return inv_sqrt_degree[:, :, None] * adj * inv_sqrt_degree[:, None, :]
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]#如果输入邻接矩阵是二维的，则同样进行广播操作。
    elif mode == "row":#行归一化
        #计算邻接矩阵每一行的绝对值之和，然后取其倒数。这个结果会被保存在 inv_degree 变量中。
        inv_degree = 1. / (adj.abs().sum(dim=-1, keepdim=False) + EOS)
        if len(adj.shape) == 3:#如果输入邻接矩阵是三维的，则进行适当的广播操作，以使 inv_degree 与邻接矩阵相乘，从而实现行归一化。
            return inv_degree[:, :, None] * adj
        return inv_degree[:, None] * adj#如果输入邻接矩阵是二维的，则同样进行广播操作。
    else:
        exit("wrong norm mode")
