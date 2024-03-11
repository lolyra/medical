import torch
import timm
import PIL
import numpy as np

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2

from scipy.stats import entropy
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import medmnist
from medmnist import INFO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FisherNet(nn.Module):
    def __init__(self, ds_info):
        super(FisherNet, self).__init__()
        self.net = timm.create_model('efficientvit_b2', pretrained=True, in_chans=ds_info['n_channels'], num_classes=ds_info['n_classes'])
        if ds_info['n_dims'] == 3:
            self.net = convert_model_to_3d(self.net)
        self.net.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.net.stem(x)
            x = self.net.stages[0](x)
            x = self.net.stages[1](x)
            x = self.net.stages[2](x)
            v = self.net.stages[3](x)
            N = x.shape[0]
            D = x.shape[1]
            x = torch.cat((
                x.reshape(N,D,-1),
                v.reshape(N,D,-1),
            ),2)
            x = x.swapaxes(1,2)
        return x

    def state_dict(self):
        return self.net.head.classifier.state_dict()

    def train(self):
        self.net.head.classifier.train()
        return self

    def eval(self):
        self.net.eval()
        return self

    def parameters(self):
        return self.net.head.classifier.parameters()


def load_dataset(data_name, batch_size):
    transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5],std=[0.5])
    ])
    
    info = INFO[data_name]
    n_channels = info['n_channels']
    classes = list(info['label'].values())
    
    DataClass = getattr(medmnist, info['python_class'])

    dataset = DataClass(split='train', download=True)

    # Entropy sorted
    x = dataset.imgs
    h = np.apply_along_axis(lambda a: np.histogram(a)[0], 1, x.reshape(x.shape[0],-1))
    h = entropy(h/h.sum(axis=1).reshape(h.shape[0],1), axis=1)
    ne = np.argsort(h)[::-1]
    
    x = torch.tensor(dataset.imgs)
    if n_channels == 1:
        x = x.reshape(x.shape[0],1,*x.shape[1:])
    else:
        x = x.swapaxes(1,3)
    y = torch.tensor(dataset.labels)
    dataset = TensorDataset(transforms(x),y)
    ds_data = DataLoader(dataset, batch_size, shuffle=False)

    ds_info = {
        'n_classes': len(classes),
        'n_samples': info['n_samples']['train'],
        'n_channels': info['n_channels'],
        'n_dims': 3 if data_name.endswith('3d') else 2,
        'task': info['task'],
        'samples_entropy': ne,
    }
    
    return ds_data,ds_info


def estimate_kl_divergence(gmm1, gmm2):
    # Generate random samples
    N = 1000000
    x,_ = gmm1.sample(N)
    # Calculate f(x)
    f = (gmm1.predict_proba(x)*gmm1.weights_).sum(axis=1)
    g = (gmm2.predict_proba(x)*gmm2.weights_).sum(axis=1)
    # Return mean
    return np.log(f/g).mean()


def main(data_name, n_kernels = 16, batch_size = 64):
    print("Loading dataset")
    ds_data, ds_info = load_dataset(data_name,batch_size)
    
    model = FisherNet(ds_info).eval().to(DEVICE)
    
    print("Extracting local features")
    with torch.no_grad():
        x = torch.tensor([])
        pbar = tqdm(total=ds_info['n_samples'], ascii=' >=')
        for inputs,_ in ds_data:
            outputs = model(inputs.to(DEVICE))
            outputs = outputs.to('cpu')
            x = torch.cat((x,outputs),0)
            pbar.update(outputs.shape[0])
        pbar.close()
        x = x.numpy()

    sample_order = ds_info['samples_entropy']

    states = [0,1,2,3,4,5,6,7,8,9]
    percs = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]

    means = np.zeros((len(states),len(percs)))

    for i in range(len(states)):
        print("Fitting GMM - 100.0%")
        n = sample_order
        y = x[n].reshape(-1,x.shape[-1])
        reg_covar = 1e-5*y.std(axis=1).max()
        gmm_t = GaussianMixture(
            n_components=n_kernels,
            covariance_type='diag',
            random_state = states[i],
            reg_covar=reg_covar
        ).fit(y)
    
        for j in range(len(percs)):
            sample_limit = round(percs[j]*len(x))
            print("Fitting GMM - {:.1f}%".format(100*percs[j]))
            n = sample_order[:sample_limit]
            y = x[n].reshape(-1,x.shape[-1])
            gmm_r = GaussianMixture(
                n_components=n_kernels,
                covariance_type='diag',
                random_state = states[i],
                reg_covar=reg_covar
            ).fit(y)
    
            means[i,j] = estimate_kl_divergence(gmm_t, gmm_r)

    endfile = f'_entropy_{n_kernels}.npz'
    np.savez(data_name + endfile, x = means)
    return
