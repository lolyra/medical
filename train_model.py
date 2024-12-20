import argparse
import os

import timm
import torch
import numpy

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from datasets import load_dataset
from converter import convert_model_to_3d
from variables import *


def main(data_name, repeat_axis):
    lr = 0.001
    gamma=0.1
    torch.manual_seed(0)
    milestones = [0.5 * NET_EPOCHS, 0.75 * NET_EPOCHS]

    print('Preparing data...')
    ds_data, ds_info = load_dataset(data_name)
    
    print('Building model...')
    model = timm.create_model(
            MODEL,
            pretrained=True,
            num_classes=ds_info['n_classes'], 
            in_chans=ds_info['n_channels']
    )
    if ds_info['n_dims'] == 3:
        model = convert_model_to_3d(model, repeat_axis)

    path = os.path.join(NET_DIR, data_name + '.pth')
    if NET_EPOCHS > 0:
        print('Training model...')
        model = model.to(DEVICE)
    
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
        best_metric = 0
    
        for epoch in range(NET_EPOCHS):
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
        return
    torch.save(model.state_dict(), path)

       
def train(model, data_loader, task, optimizer):
    total_loss = []
    model.train()
    pbar = tqdm(total=len(data_loader.dataset), ascii=' >=')
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(DEVICE))
        pbar.update(outputs.shape[0])

        if task == 'multi-label, binary-class':
            criterion = torch.nn.BCEWithLogitsLoss()
            targets = targets.to(torch.float32).to(DEVICE)
        else:
            criterion = torch.nn.CrossEntropyLoss()
            if targets.dim() == 2:
                targets = torch.squeeze(targets, 1).long()
            targets = targets.to(DEVICE)
            
        loss = criterion(outputs, targets)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    pbar.close()
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, data_loader, task):
    model.eval()
    y_score = torch.tensor([])
    y_true = torch.tensor([])

    pbar = tqdm(total=len(data_loader.dataset), ascii=' >=')
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(DEVICE))
            pbar.update(outputs.shape[0])

            if task == "multi-label, binary-class":
                m = torch.nn.Sigmoid()
            else:
                m = torch.nn.Softmax(dim=1)

            outputs = m(outputs)
            y_score = torch.cat((y_score, outputs.cpu()), 0)
            y_true = torch.cat((y_true, targets), 0)
    pbar.close()

    y_true = y_true.numpy()
    if task == 'binary-class':
        y_pred = torch.argmax(y_score, dim=1)
        y_score = y_score[:,1].numpy()
    elif task == "multi-label, binary-class":
        y_score = y_score.numpy()
        y_pred = numpy.array(list(map(round, y_score.reshape(-1)))).reshape(y_score.shape[0],-1)
    else:
        y_score = y_score.numpy()
        y_pred = numpy.argmax(y_score, axis=1)
    auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    acc = accuracy_score(y_true, y_pred)
    return [auc, acc]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d','--dataset',
                        required=True,
                        type=str)
    parser.add_argument('-r','--repeat_axis',
                        default=-1,
                        type=int)
    
    args = parser.parse_args()
    main(
        args.dataset,
        args.repeat_axis,
    )

