import os
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 28
NUM_LAYERS = 2
BATCH_SIZE = 16
MODEL = 'efficientvit_b2' # Backbone model
MAX_SAMPLES = 1000 # Max samples for GMM estimation

NET_EPOCHS = 10 # Epochs for backbone fine-tuning, 0 means no fine-tuning
CLF_EPOCHS = 10

ROOT_DIR = os.path.join(os.environ['HOME'],'data','models')
DATA_DIR = os.path.join(os.environ['HOME'],'data','datasets')
NET_DIR = os.path.join(ROOT_DIR, 'pretrained' if NET_EPOCHS == 0 else 'finetuned', 'backbone')
GMM_DIR = os.path.join(ROOT_DIR, 'pretrained' if NET_EPOCHS == 0 else 'finetuned', f'L{NUM_LAYERS}')
CLF_DIR = GMM_DIR

for path in [NET_DIR, GMM_DIR, CLF_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)




