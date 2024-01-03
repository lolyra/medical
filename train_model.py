import argparse
import os

import timm
import medmnist
import torch

from copy import deepcopy
from tqdm import tqdm

from datasets import load_dataset
from models import DEVICE, MODEL
from converter import convert_model_to_3d

def main(data_name, file_name, image_size, batch_size, num_epochs, pretrained, repeat_axis):
    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    path = os.path.join(os.environ['HOME'],'data',data_name)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, file_name)
    
    print('Preparing data...')
    ds_data, ds_info = load_dataset(data_name, image_size, batch_size)
    
    print('Building model...')
    model = timm.create_model(
            MODEL,
            pretrained=pretrained,
            num_classes=ds_info['n_classes'], 
            in_chans=ds_info['n_channels']
    )
    if ds_info['n_dims'] == 3:
        model = convert_model_to_3d(model, repeat_axis)
    model.load_state_dict(torch.load(path)['net'])
    
    if num_epochs > 0:
        print('Training model...')
        model = model.to(DEVICE)
    
        evaluator = medmnist.Evaluator(data_name, 'val')
    
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
        best_metric = 0
    
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}")
            train_loss = train(model, ds_data['train'], ds_info['task'], optimizer)
            print("train loss: {:.5f}".format(train_loss))
            val_metrics = test(model, ds_data['val'], ds_info['task'], evaluator)
            print('val auc: {:.5f}  acc: {:.5f}'.format(*val_metrics))
            scheduler.step()
            
            cur_metric = val_metrics[0] + val_metrics[1]
            if cur_metric > best_metric:
                best_metric = cur_metric
                torch.save({'net':model.state_dict()}, path)
        return
    torch.save({'net':model.state_dict()}, path)

       
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
            targets = torch.squeeze(targets, 1).long().to(DEVICE)
            
        loss = criterion(outputs, targets)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    pbar.close()
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, data_loader, task, evaluator):
    model.eval()
    y_score = torch.tensor([])
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
        y_score = y_score.numpy()
    pbar.close()
    auc, acc = evaluator.evaluate(y_score)
    return [auc, acc]

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
    parser.add_argument('--num_epochs',
                        default=10,
                        type=int)
    parser.add_argument('-p','--pretrained',
                        action='store_true')
    parser.add_argument('-r','--repeat_axis',
                        default=-1,
                        type=int)

    args = parser.parse_args()
    main(
        args.dataset,
        args.file,
        args.image_size,
        args.batch_size,
        args.num_epochs,
        args.pretrained,
        args.repeat_axis,
    )

