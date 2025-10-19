import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.data.utils import custom_collate
from src.data.supervised import SupervisedRandFunDataset
from src.data.unsupervised import UnsupervisedRandFunDataset
from src.data.datasets import CombinedRandFunDataset, EvalPDEDataset


class PDEDataModule(pl.LightningDataModule):
    def __init__(self, data_size=1_280_000, batch_size=128, grid_size=224, 
                 eval_pde_paths=['eval_data/heat.pkl', 'eval_data/helmholtz.pkl',
                                'eval_data/helmholtz_G.pkl', 'eval_data/poisson_C.pkl',
                                'eval_data/poisson_L.pkl', 'eval_data/poisson_G.pkl',
                                'eval_data/wave.pkl'],
                supervised_only=False):
        super().__init__()
        self.data_size = data_size
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.eval_pde_paths = eval_pde_paths
        self.supervised_only = supervised_only

    def setup(self, stage=None):
        if self.supervised_only:
            self.train_dataset = SupervisedRandFunDataset(size=self.data_size, max_num_inner_boundaries=4)
        else:
            supervised_dataset_bounds = SupervisedRandFunDataset(size=self.data_size // 2, max_num_inner_boundaries=4)
            unsupervised_dataset_bounds = UnsupervisedRandFunDataset(size=self.data_size // 2)
            self.train_dataset = CombinedRandFunDataset([supervised_dataset_bounds, unsupervised_dataset_bounds])
        self.val_dataset = EvalPDEDataset(self.eval_pde_paths)
        self.test_dataset = EvalPDEDataset(self.eval_pde_paths)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=20, 
                          persistent_workers=True, pin_memory=True, collate_fn=custom_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, 
                          persistent_workers=True, pin_memory=True, collate_fn=custom_collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, 
                          persistent_workers=True, pin_memory=True, collate_fn=custom_collate)
