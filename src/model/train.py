from argparse import ArgumentParser
from collections import Counter
import json
import os
import sys
sys.path.append('./src')
from lib.modeling import train_model
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch import save as torchsave

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dset


# ---------------
# Parse arguments
# ---------------

parser = ArgumentParser()
parser.add_argument("-t", "--train_path", type=str, default='/hear_and_see/data/rolled_spectrogram/trainset/')
parser.add_argument("-v", "--validation_path", type=str, default='/hear_and_see/data/rolled_spectrogram/valset/')
parser.add_argument("-n", "--n_epochs",type=int, default=15)
parser.add_argument("-l", "--learning_rate",type=float, default=0.001)
parser.add_argument("-m", "--momentum",type=float, default=0.9)
parser.add_argument("-w", "--weight_decay",type=float, default=0.0005)
parser.add_argument("-b", "--batch_size",type=int, default=16)
parser.add_argument("-s", "--save_path",type=str, default='/hear_and_see/data/models')
parser.add_argument("-f", "--filename",type=str, default='model')

args = parser.parse_args()

train_path = args.train_path
val_path = args.validation_path
n_epochs = args.n_epochs
lr = args.learning_rate
momentum = args.momentum
weight_decay = args.weight_decay
batch_size = args.batch_size

# ---------
# Load data
# ---------

# Original shape is 440x600
transform = transforms.Compose([transforms.Scale((224,224)),transforms.ToTensor()])

train_set = dset.ImageFolder(train_path,transform=transform)
val_set = dset.ImageFolder(val_path,transform=transform)

trainLoader = DataLoader(train_set,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 0
)

valLoader = DataLoader(val_set,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 0
)

class_dict = trainLoader.dataset.class_to_idx
class_dict = {v: k for k, v in class_dict.items()} # reverse key-value mappings

train_n = len(trainLoader.dataset)
test_n = len(valLoader.dataset)

print('Training sample size:',train_n)
print('Testing sample size:',test_n)

# ---------------
# Construct model
# ---------------

resnet = models.resnet50(pretrained = True)
print('Original fc layer length:',resnet.fc)
total_classes = len(class_dict.keys())
resnet.fc = nn.Linear(resnet.fc.in_features, total_classes)
print('Revised fc layer length:',resnet.fc)

criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = None
optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# Train model
t0 = time.time()
training_values = train_model(resnet, criterion, optimizer, trainLoader, valLoader, exp_lr_scheduler,lr_step=False, n_epochs=n_epochs, use_gpu=True)
t1 = time.time()
total = t1-t0

# ----------
# Save model
# ----------

# Create write directory if it doesn't exist
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# Save model weights
extension = '.pt'
PATH = ''.join([args.filename,extension])
PATH = os.path.join(args.save_path,PATH)
torchsave(resnet.state_dict(), PATH)

print('Saved',PATH)

# Save hyperparameters
experiment_log = {
    'lr':lr,
    'batch_size':batch_size,
    'momentum':momentum,
    'weight_decay':weight_decay,
    'n_epochs':n_epochs,
    'optimizer':str(type(optimizer)),
    'criterion':str(type(criterion)),
    'training_accuracy':training_values[0],
    'validation_accuracy':training_values[1],
    'training_loss':training_values[2],
    'validation_loss':training_values[3],
    'train_time(min)':total/60,
    'train_path':args.train_path,
    'validation_path':args.validation_path,
    'model_path':PATH
}

extension = '.json'
log_path = ''.join([args.filename,extension])
log_path = os.path.join(args.save_path,log_path)

with open(log_path, 'w') as outfile:
    json.dump(experiment_log, outfile)

print('Saved',log_path)
