import torch
import numpy
import pickle
import os
import argparse
from sklearn.mixture import GaussianMixture
from datasets import load_entropy
from models import FisherNet, DEVICE
from tqdm import tqdm

def main(data_name, file_name, n_kernels, image_size, batch_size, sample_limit):
    ''' Fit GMM model '''
    ds_data, ds_info = load_entropy(data_name,image_size,batch_size,sample_limit)
    path = os.path.join(os.environ['HOME'],'data',data_name,file_name)
    model = FisherNet(ds_info, path, features_only=True).eval().to(DEVICE)
    
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
    params = torch.load(path)
    params['gmm'] = gmm._get_parameters()
    torch.save(params, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--dataset',
                        required=True,
                        type=str)
    parser.add_argument('-f','--file',
                        required=True,
                        type=str)
    parser.add_argument('--kernels',
                        default=16,
                        type=int)
    parser.add_argument('--image_size',
                        default=224,
                        type=int)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)
    parser.add_argument('--sample_limit',
                        default=1000,
                        type=int)

    args = parser.parse_args()
    main(
        args.dataset,
        args.file,
        args.kernels,
        args.image_size,
        args.batch_size,
        args.sample_limit
    )
