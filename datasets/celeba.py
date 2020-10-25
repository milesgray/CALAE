import os
import logging
import pathlib
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
from os import listdir
from glob import glob


####################################################################################################################
########### C E L E B A #############-------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# CelebA face image dataset, only returns images and not metadata
# ------------------------------------------------------------------------------------------------------------------
class CelebA(Dataset):
    def __init__(self, path='/root/data/CelebA/img_align_celeba/', part='train'):
        if part=='train':
            self.data = [os.path.join(path, file) for file in os.listdir(path)][:182637]
        else:
            self.data = [os.path.join(path, file) for file in os.listdir(path)][182637:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transform(Image.open(self.data[idx]))
    
def make_celeba_dataloader(dataset, batch_size, image_size=4):
    dataset.transform = transforms.Compose([
                                            transforms.Resize((image_size, image_size)),                
                                            transforms.RandomHorizontalFlip(),      
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
    return DataLoader(dataset, 
                      shuffle=True, 
                      batch_size=batch_size, 
                      num_workers=4, 
                      drop_last=True)
