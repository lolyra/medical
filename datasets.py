import PIL
import numpy as np
from scipy.stats import entropy

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2

import medmnist
from medmnist import INFO

def load_dataset(dataname, size, batch_size):
    transforms = v2.Compose([
        v2.Resize(size, PIL.Image.NEAREST),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5],std=[0.5])
    ])
    
    info = INFO[dataname]
    n_channels = info['n_channels']
    classes = list(info['label'].values())
    
    DataClass = getattr(medmnist, info['python_class'])

    ds_data = {}
    for split in ['train','val','test']:
        flag = split == 'train'
        dataset = DataClass(split=split, download=flag)
        x = torch.tensor(dataset.imgs)
        if n_channels == 1:
            x = x.reshape(x.shape[0],1,*x.shape[1:])
        else:
            x = x.swapaxes(1,3)
        y = torch.tensor(dataset.labels)
        dataset = TensorDataset(transforms(x),y)
        ds_data[split] = DataLoader(dataset, batch_size, flag)

    ds_info = {
        'n_classes': len(classes),
        'n_samples': info['n_samples'],
        'n_channels': info['n_channels'],
        'n_dims': 3 if dataname.endswith('3d') else 2,
        'task': info['task'],
    }
    
    return ds_data,ds_info

def load_entropy(dataname, size, batch_size, file_limit):
    transforms = v2.Compose([
        v2.Resize(size, PIL.Image.NEAREST),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5],std=[0.5])
    ])
    
    info = INFO[dataname]
    n_channels = info['n_channels']
    classes = list(info['label'].values())

    DataClass = getattr(medmnist, info['python_class'])

    dataset = DataClass(split='train', download=True)
    x = dataset.imgs
    h = np.apply_along_axis(lambda a: np.histogram(a)[0], 1, x.reshape(x.shape[0],-1))
    h = entropy(h/h.sum(axis=1).reshape(h.shape[0],1), axis=1)
    n = np.argsort(h)[::-1][:file_limit]

    x = torch.tensor(dataset.imgs[n])
    if n_channels == 1:
        x = x.reshape(x.shape[0],1,*x.shape[1:])
    else:
        x = x.swapaxes(1,3)
    y = torch.tensor(dataset.labels[n])
    n_samples = len(x)
    
    dataset = TensorDataset(transforms(x),y)
    ds_data = DataLoader(dataset, batch_size, False)

    ds_info = {
        'n_classes': len(classes),
        'n_samples': n_samples,
        'n_channels': info['n_channels'],
        'n_dims': 3 if dataname.endswith('3d') else 2,
        'task': info['task'],
    }
    return ds_data, ds_info
