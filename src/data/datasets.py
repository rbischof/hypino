import pickle
import numpy as np

from typing import List
from torch.utils.data import Dataset
from src.data.utils import resize, encode_pde_str, to_tensor

class RandomFunctionDataset(Dataset):
    def __init__(self, size, grid_size=224, init_grid_size=224, max_abs_val=1e3, min_std_val=0.1, max_num_inner_boundaries=4):
        super().__init__()
        self.size = size
        self.grid_size = grid_size
        self.init_grid_size = init_grid_size
        self.max_abs_val = max_abs_val
        self.min_std_val = min_std_val
        self.max_num_inner_boundaries = max_num_inner_boundaries

    def __len__(self):
        return self.size
        
class CombinedRandFunDataset(Dataset):
    def __init__(self, datasets: List[RandomFunctionDataset]):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return sum([d.__len__() for d in self.datasets])
    
    def __getitem__(self, idx):
        return self.datasets[idx % len(self.datasets)].__getitem__(idx // len(self.datasets))
        
class EvalPDEDataset(Dataset):
    def __init__(self, pde_paths, grid_size=224, init_grid_size=224):
        super().__init__()
        self.size = len(pde_paths)
        self.pde_paths = pde_paths
        self.grid_size = grid_size
        self.init_grid_size = init_grid_size

        self.pdes = []
        for path in pde_paths:
            with open(path, 'rb') as f:
                self.pdes.append(pickle.load(f))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self.pdes[idx]

        xy_grid = data['domain'].reshape(self.init_grid_size, self.init_grid_size, 2).transpose(2, 0, 1)
        f_grid_big, dir_bound_grid, neu_bound_grid, dir_boundc_grid, neu_boundc_grid, neu_normals_grid = resize([data['f_grid'], data['dir_bound'], data['neu_bound'], data['dir_boundc'], data['neu_boundc'], data['neu_normals']], factor=self.grid_size / self.init_grid_size)

        mat_inputs = np.concatenate([
            np.expand_dims(dir_bound_grid, axis=0), 
            np.expand_dims(neu_bound_grid, axis=0), 
            np.expand_dims(dir_boundc_grid, axis=0), 
            np.expand_dims(neu_boundc_grid, axis=0), 
            np.expand_dims(f_grid_big, axis=0),
        ], axis=0)
        
        pde_coeffs_dict = encode_pde_str(data['diff_op_str'])
        return {
            'pde_str': data['diff_op_str'],
            'pde_derivatives': data['derivatives'],
            'pde_coeffs': to_tensor([c for c in pde_coeffs_dict.values()]),
            'mat_inputs': to_tensor(mat_inputs),
            'dir_boundary': to_tensor(dir_boundc_grid),
            'neu_boundary': to_tensor(neu_boundc_grid),
            'neu_normals': to_tensor(neu_normals_grid),
            'xy_grid': to_tensor(xy_grid),
            'u_grid': to_tensor(data['u_grid']).reshape(self.init_grid_size, self.init_grid_size),
            'f_grid': to_tensor(data['f_grid']).reshape(self.init_grid_size, self.init_grid_size),
            'domain_mask': to_tensor(data['domain_mask']),
        }