import torch
import argparse
import os

from datasets import load_dataset
from models import create_model
from variables import *
from train_model import train, test


def main(data_name):
    lr = 0.0001
    gamma=0.1
    milestones = [0.5 * CLF_EPOCHS, 0.75 * CLF_EPOCHS]

    print('Preparing data...')
    ds_data, ds_info = load_dataset(data_name)

    print('Building model...')
    model = create_model(ds_info).to(DEVICE)

    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    best_metric = 0

    path = os.path.join(CLF_DIR, data_name + '.pth')
    for epoch in range(CLF_EPOCHS):
        print(f"Epoch {epoch+1}")
        train_loss = train(model, ds_data['train'], ds_info['task'], optimizer)
        print("train loss: {:.5f}".format(train_loss))
        val_metrics = test(model, ds_data['val'], ds_info['task'])
        print('val auc: {:.5f}  acc: {:.5f}'.format(*val_metrics))
        scheduler.step()
        
        cur_metric = val_metrics[0] + val_metrics[1]
        if cur_metric > best_metric:
            best_metric = cur_metric
            torch.save(model.state_dict(), path)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--dataset',
                        required=True,
                        type=str)

    args = parser.parse_args()
    main(
        args.dataset,
    )
