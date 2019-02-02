'''
Custom datasets
===
Distributed by: Notre Dame CICS (MIT Liscense)
- Associated publication:
url: https://www.sciencedirect.com/science/article/pii/S0021999119300464
doi: https://doi.org/10.1016/j.jcp.2019.01.021
github: https://github.com/cics-nd/rans-uncertainty
===
'''
import torch as th
import torch.utils.data

class FoamNetDataset(th.utils.data.Dataset):
    """
    Dataset wrapping data and target tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Args:
        invar_tensor (Tensor): contains invariant inputs
        tensor_tensor (Tensor): contains tensor basis functions
        k_tensor (Tensor): RANS TKE
        target_tensor (Tensor): Target anisotropic data tensor
    """

    def __init__(self, invar_tensor, tensor_tensor, k_tensor, target_tensor):
        assert invar_tensor.size(0) == target_tensor.size(0)
        self.invar_tensor = invar_tensor
        self.tensor_tensor = tensor_tensor
        self.k_tensor = k_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.invar_tensor[index], self.tensor_tensor[index], self.k_tensor[index], self.target_tensor[index]

    def __len__(self):

        return self.invar_tensor.size(0)

class PredictDataset(th.utils.data.Dataset):
    """
    Dataset wrapping data and target tensors.
    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Args:
        invar_tensor (Tensor): contains invariant inputs
        tensor_tensor (Tensor): contains tensor basis functions
    """

    def __init__(self, invar_tensor, tensor_tensor):
        assert invar_tensor.size(0) == tensor_tensor.size(0)
        self.invar_tensor = invar_tensor
        self.tensor_tensor = tensor_tensor

    def __getitem__(self, index):
        return self.invar_tensor[index], self.tensor_tensor[index]

    def __len__(self):

        return self.invar_tensor.size(0)