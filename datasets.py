import PIL
import numpy as np
from scipy.stats import entropy

import torch
from torch.utils.data import Sampler, DataLoader, Subset, random_split
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder

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
    if dataname in ['isic2018','covid']:
        return v2.Compose([
            v2.ToImage(),
            v2.CenterCrop([IMAGE_SIZE,IMAGE_SIZE]),
            v2.Resize([IMAGE_SIZE,IMAGE_SIZE]),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5],std=[0.5])
        ])
    return v2.Compose([
            v2.ToImage(),
            v2.Resize([IMAGE_SIZE, IMAGE_SIZE], PIL.Image.NEAREST),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5],std=[0.5])
        ])


def load_covid(flag: bool):
    datadir = os.path.join(DATA_DIR,'covid')
    dataset = ImageFolder(os.path.join(datadir,'images'), get_transform('covid'))
    ds_data = {}
    n_samples = {}
    for split in ['train','val','test']:
        with open(os.path.join(datadir,f'splits/{split}CT_COVID.txt')) as f:
            data = [x.strip() for x in f.readlines()]
        with open(os.path.join(datadir,f'splits/{split}CT_NonCOVID.txt')) as f:
            data+= [x.strip() for x in f.readlines()]
        idx = [x for x in range(len(dataset)) if dataset.imgs[x][0].split('/')[-1] in data]
        n_samples[split] = len(idx)
        ds_data[split] = DataLoader(
            Subset(dataset, idx), 
            BATCH_SIZE, 
            flag if split == 'train' else False
        )
    ds_info = {
        'name': 'covid',
        'n_classes': 2,
        'n_samples': n_samples,
        'n_channels': 3,
        'n_dims': 2,
        'task': 'binary-class',
    }
    return ds_data, ds_info


def load_isic2018(flag: bool):
    datadir = os.path.join(DATA_DIR,'isic2018','train')
    dataset = ImageFolder(os.path.join(datadir), get_transform('isic2018'))
    n_samples = {'train': int(0.7 * len(dataset))}
    n_samples['val'] = len(dataset) - n_samples['train']
    n_samples['test'] = n_samples['val']
    generator = torch.Generator().manual_seed(42)
    train_data, test_data = random_split(dataset, [n_samples['train'], n_samples['test']], generator)
    ds_data = {
        'train': DataLoader(train_data, BATCH_SIZE, flag),
        'val': DataLoader(test_data, BATCH_SIZE, False),
        'test': DataLoader(test_data, BATCH_SIZE, False)
    }
    ds_info = {
        'name': 'isic2018',
        'n_classes': 7,
        'n_samples': n_samples,
        'n_channels': 3,
        'n_dims': 2,
        'task': 'multi-class',
    }
    return ds_data, ds_info


def load_medmnist(dataname):
    info = INFO[dataname]
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


def load_dataset(dataname, flag = True):
    if dataname == 'covid':
        return load_covid(flag)
    elif dataname == 'isic2018':
        return load_isic2018(flag)
    return load_medmnist(dataname)


def load_entropy(dataname):
    ds_data, ds_info = load_dataset(dataname, False)
    dataset = ds_data['train'].dataset
    z = []
    for x,_ in ds_data['train']:
        h = np.apply_along_axis(lambda a: np.histogram(a)[0], 1, x.reshape(x.shape[0],-1))
        h = entropy(h/h.sum(axis=1).reshape(h.shape[0],1), axis=1)
        z+= h.tolist()
    n = np.argsort(z)[::-1][:MAX_SAMPLES]
    custom_sampler = FixedSubsetSampler(n)
    ds_data = DataLoader(dataset, BATCH_SIZE, sampler = custom_sampler)
    ds_info['n_samples'] = min(len(dataset),MAX_SAMPLES)
    return ds_data, ds_info
