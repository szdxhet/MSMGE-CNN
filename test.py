import torch
import torch.nn.functional as F
import numpy as np
from scipy import spatial
import math

if __name__ == '__main__':
    u = torch.rand((22, 100))
    v = torch.rand((4, 100))



    arr1 = u.detach().numpy()
    arr2 = v.detach().numpy()
    norm1 = np.linalg.norm(arr1,axis=-1,keepdims=True)
    norm2 = np.linalg.norm(arr2,axis=-1,keepdims=True)
    #print(norm1)
    #print(norm2)
    arr1_norm = arr1 / norm1
    #print(arr1_norm)

    arr2_norm = arr2 / norm2
    #print(arr2_norm)
    cos = np.dot(arr1_norm, arr2_norm.T)
    print(cos)
    print(cos.shape)
    print(cos.mean())






