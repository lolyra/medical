import torch
import numpy
import pickle
import os
import argparse
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from datasets import load_entropy
from models import create_model
from variables import *


def main(data_name, n_kernels):
    ''' Fit GMM model '''
    ds_data, ds_info = load_entropy(data_name)
    model = create_model(ds_info, features_only=True).eval().to(DEVICE)
    
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
        
    print("Fitting Gaussian Mixture Model")
    x = x.reshape(-1,x.shape[-1])
    print(x.shape)
    gmm = GaussianMixture(
        n_components=n_kernels,
        covariance_type='diag', 
        verbose=10,
        reg_covar=1e-4*x.std(axis=1).max()
    ).fit(x)
    # Save Gaussian Mixture
    path = os.path.join(GMM_DIR, data_name+'.npz')
    params = gmm._get_parameters()
    numpy.savez_compressed(path,
                        weights=params[0],
                        means=params[1],
                        covars=params[2],
                        precision=params[3])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--dataset',
                        required=True,
                        type=str)
    parser.add_argument('-k','--kernels',
                        default=16,
                        type=int)

    args = parser.parse_args()
    main(
        args.dataset,
        args.kernels,
    )
