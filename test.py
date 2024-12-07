import timm
import torch
import argparse

from datasets import load_dataset
from variables import *
from train_model import train, test


def main(data_name, model_name):
    lr = 0.0001
    gamma = 0.1
    epochs = NET_EPOCHS
    milestones = [0.5 * epochs, 0.75 * epochs]

    print('Preparing data...')
    ds_data, ds_info = load_dataset(data_name)
    
    print('Building model...')
    model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=ds_info['n_classes'], 
            in_chans=ds_info['n_channels']
    )


    print('Training model...')
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    best_metric = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_loss = train(model, ds_data['train'], ds_info['task'], optimizer)
        print("train loss: {:.5f}".format(train_loss))
        val_metrics = test(model, ds_data['test'], ds_info['task'])
        print('test auc: {:.5f}  acc: {:.5f}'.format(*val_metrics))
        scheduler.step()
        
        cur_metric = val_metrics[0] + val_metrics[1]
        if cur_metric > best_metric:
            print(val_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--dataset',
                        required=True,
                        type=str)
    parser.add_argument('-m','--model',
                        required=True,
                        type=str)

    args = parser.parse_args()
    main(
        args.dataset,
        args.model,
    )
