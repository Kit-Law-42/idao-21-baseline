import pathlib as path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F

from .dataloader import IDAODataset, img_loader, InferenceDataset


class IDAODataModule(pl.LightningDataModule):
    def __init__(self, data_dir: path.Path, batch_size: int, cfg):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cfg = cfg

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # see dataloader.py
        self.dataset = IDAODataset(
            root=self.data_dir.joinpath("train"),
            loader=img_loader,
            transform=transforms.Compose(
<<<<<<< Updated upstream
                [transforms.ToTensor(), transforms.CenterCrop(120)] # crop image to 120*120
=======
                [transforms.ToTensor(), 
                transforms.CenterCrop(224),
                transforms.Normalize((0.449,), (0.226,)),
                ] # crop image to 120*120
                
>>>>>>> Stashed changes
            ),
            target_transform=transforms.Compose(
                [
                    lambda num: (
<<<<<<< Updated upstream
                        torch.tensor([0, 1]) if num == 0 else torch.tensor([1, 0]) # one hot encoding.
=======
                        torch.tensor([0, 1]) if num == 0 else torch.tensor([1, 0]) # one hot encoding to classfy image class=ER/NR.
>>>>>>> Stashed changes
                    )
                ]
            ),
            extensions=self.cfg["DATA"]["Extension"],
        )

        self.public_dataset = InferenceDataset(
                    main_dir=self.data_dir.joinpath("public_test"),
                    loader=img_loader,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), 
                        transforms.CenterCrop(224),
                        transforms.Normalize((0.449,), (0.226,)),
                        ]
                        
                    ),
                )
        self.private_dataset = InferenceDataset(
                    main_dir=self.data_dir.joinpath("private_test"),
                    loader=img_loader,
                    transform=transforms.Compose(
                        [transforms.ToTensor(), 
                        transforms.CenterCrop(224),
                         transforms.Normalize((0.449,), (0.226,)),
                         ]
                        
                    ),
                )


    def setup(self, stage=None):
        # called on every GPU
        # make assignments here (val/train/test split)
        # ValueError: Sum of input lengths does not equal the length of the input dataset!
        self.train, self.val = random_split(
            self.dataset, [10000, 3404], generator=torch.Generator().manual_seed(666) #in total should have 13526 images?
        )
        

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.val, 1, num_workers=4, shuffle=False, pin_memory=torch.cuda.is_available())
    
    def test_dataloader(self):
        return DataLoader(
            torch.utils.data.ConcatDataset([self.private_dataset, self.public_dataset]),
            1,
            num_workers=4,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
            )

