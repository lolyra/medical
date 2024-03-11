import argparse
import os

import timm
import medmnist
import torch

from datasets import load_dataset
from variables import *
from converter import convert_model_to_3d
from train_model import test


def main(data_name):
    
    print('Preparing data...')
    ds_data, ds_info = load_dataset(data_name)
    evaluator = medmnist.Evaluator(data_name, 'test')
    
    print('Building model...')
    model = timm.create_model(
            MODEL,
            num_classes=ds_info['n_classes'], 
            in_chans=ds_info['n_channels']
    )
    if ds_info['n_dims'] == 3:
        model = convert_model_to_3d(model)

    path = os.path.join(NET_DIR, data_name + '.pth')
    params = torch.load(path)
    model.load_state_dict(params)
    model.to(DEVICE)
    
    print('Testing model...')
    auc, acc = test(model, ds_data['test'], ds_info['task'], evaluator)
    print('test auc: {:.3f}  acc: {:.3f}'.format(auc,acc))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--dataset',
                        required=True,
                        type=str)

    args = parser.parse_args()
    main(
        args.dataset,
    )

