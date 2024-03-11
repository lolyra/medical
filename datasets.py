import PIL
import numpy as np
from scipy.stats import entropy

import torch
from torch.utils.data import Sampler, DataLoader
from torchvision.transforms import v2

import medmnist
from medmnist import INFO
from variables import *


class FixedSubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class Transform3D:
    def __init__(self):
        self.transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(IMAGE_SIZE, PIL.Image.NEAREST),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5],std=[0.5])
        ])

    def __call__(self, x):
        x = torch.tensor(x)
        return self.transforms(x)


def get_transform(dataname):
    if dataname.endswith('3d'):
        return Transform3D()
    return v2.Compose([
            v2.ToImage(),
            v2.Resize(IMAGE_SIZE, PIL.Image.NEAREST),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5],std=[0.5])
        ])


def load_dataset(dataname):
    info = INFO[dataname]
    n_channels = info['n_channels']
    classes = list(info['label'].values())
    
    DataClass = getattr(medmnist, info['python_class'])

    ds_data = {}
    for split in ['train','val','test']:
        flag = split == 'train'
        dataset = DataClass(split=split, download=flag, transform=get_transform(dataname))
        ds_data[split] = DataLoader(dataset, BATCH_SIZE, flag)

    ds_info = {
        'name': dataname,
        'n_classes': len(classes),
        'n_samples': info['n_samples'],
        'n_channels': info['n_channels'],
        'n_dims': 3 if dataname.endswith('3d') else 2,
        'task': info['task'],
    }
    
    return ds_data,ds_info

def load_entropy(dataname):
    info = INFO[dataname]
    n_channels = info['n_channels']
    classes = list(info['label'].values())

    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split='train', download=True, transform=get_transform(dataname))
    
    x = dataset.imgs
    h = np.apply_along_axis(lambda a: np.histogram(a)[0], 1, x.reshape(x.shape[0],-1))
    h = entropy(h/h.sum(axis=1).reshape(h.shape[0],1), axis=1)
    n = np.argsort(h)[::-1][:MAX_SAMPLES]

    custom_sampler = FixedSubsetSampler(n)
    ds_data = DataLoader(dataset, BATCH_SIZE, sampler=custom_sampler)

    ds_info = {
        'name': dataname,
        'n_classes': len(classes),
        'n_samples': len(n),
        'n_channels': info['n_channels'],
        'n_dims': 3 if dataname.endswith('3d') else 2,
        'task': info['task'],
    }
    return ds_data, ds_info
