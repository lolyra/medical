import medmnist
import torch
import argparse
import os

from datasets import load_dataset
from models import FisherNet, DEVICE
from train_model import test

def main(data_name, file_name, image_size, batch_size):
    print('Preparing data...')
    ds_data, ds_info = load_dataset(data_name,image_size,batch_size)
    evaluator = medmnist.Evaluator(data_name, 'test')

    print('Building model...')
    path = os.path.join(os.environ['HOME'],'data',data_name,file_name)
    model = FisherNet(ds_info, path, load_classifier=True).to(DEVICE)

    print('Testing model...')
    auc, acc = test(model, ds_data['test'], ds_info['task'], evaluator)
    print('test auc: {:.3f}  acc: {:.3f}'.format(auc,acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--dataset',
                        required=True,
                        type=str)
    parser.add_argument('-f','--file',
                        required=True,
                        type=str)
    parser.add_argument('--image_size',
                        default=224,
                        type=int)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)

    args = parser.parse_args()
    main(
        args.dataset,
        args.file,
        args.image_size,
        args.batch_size,
    )
