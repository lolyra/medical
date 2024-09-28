import argparse

from datasets import load_dataset
from models import create_model
from train_model import test
from variables import *

def main(data_name):
    print('Preparing data...')
    ds_data, ds_info = load_dataset(data_name)

    print('Building model...')
    model = create_model(ds_info, load_classifier=True).to(DEVICE)

    print('Testing model...')
    auc, acc = test(model, ds_data['test'], ds_info['task'])
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
