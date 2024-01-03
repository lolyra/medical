import medmnist
import torch
import argparse
import os

from tqdm import tqdm

from datasets import load_dataset
from models import FisherNet, DEVICE
from train_model import test

def main(data_name, file_name, image_size, batch_size, num_epochs):
    lr = 0.0001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    ds_data, ds_info = load_dataset(data_name,image_size,batch_size)
    
    evaluator = medmnist.Evaluator(data_name, 'val')

    path = os.path.join(os.environ['HOME'],'data',data_name,file_name)
    model = FisherNet(ds_info, path).to(DEVICE)

    optimizer = torch.optim.RAdam(model.net.head.classifier.parameters(), lr=lr)
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
            params = torch.load(path)
            params['cfv'] = model.net.head.classifier.state_dict()
            torch.save(params, path)


def train(model, data_loader, task, optimizer):
    total_loss = []
    model.net.head.classifier.train()
    #model.train()
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

    args = parser.parse_args()
    main(
        args.dataset,
        args.file,
        args.image_size,
        args.batch_size,
        args.num_epochs,
    )
