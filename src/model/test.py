from argparse import ArgumentParser
import json
import os
import sys
sys.path.append('./src')
from lib.plotting import confusion_matrix as plot_confusion_matrix

from torch.autograd import Variable
import torch.nn as nn
from torch import LongTensor
from torch import load as torchload
from torch import max as torchmax
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchnet.meter as meter

from sklearn import metrics


parser = ArgumentParser()
parser.add_argument("-m", "--model_directory", type=str, default='rolled_spectrograms')
parser.add_argument("-t", "--test_directory", type=str, default='padded_spectrograms') # may not always want testset
args = parser.parse_args()


# Model
read_path = os.path.join('/hear_and_see/data/models',args.model_directory)
model_path = os.path.join(read_path,'model.pt')
log_path = os.path.join(read_path,'model.json')

# Test data
data_path = os.path.join('/hear_and_see/data',args.test_directory,'testset')

# Write log
write_path = os.path.join('/hear_and_see/results/testing',args.model_directory)

# Create write directory if it doesn't exist
test_dir = '/hear_and_see/results/testing'
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

test_dir = write_path
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Load log
with open(log_path, 'r') as f:
    model_log = json.load(f)

# Test data (original shape is 440x600)
transform = transforms.Compose([transforms.Scale((224,224)),transforms.ToTensor()])
test_set = dset.ImageFolder(data_path,transform=transform)
testLoader = DataLoader(
    test_set,
    batch_size = model_log['batch_size'],
    shuffle = True,
    num_workers = 0
)

class_dict = testLoader.dataset.class_to_idx
class_dict = {v: k for k, v in class_dict.items()}
total_classes = len(class_dict.keys())

# Load model
resnet = models.resnet50(pretrained = True)
resnet.fc = nn.Linear(resnet.fc.in_features, total_classes)
resnet.load_state_dict(torchload(model_path,map_location=lambda storage, loc: storage))

use_cuda = True
if use_cuda: 
    resnet.cuda().eval()
else:
    resnet.eval()

y_true, y_pred = [], []
for data, label in testLoader:
    data, label = Variable(data, volatile=True), Variable(label, volatile=True)
    if use_cuda:
        data = data.cuda()
        label = label.cuda()
    output = resnet(data)
    pred = torchmax(output.data, dim=1)[1].cpu().numpy().tolist()
    truth = label.data.cpu().numpy().tolist()
    y_pred.extend(pred)
    y_true.extend(truth)

f1 = metrics.f1_score(y_true, y_pred, average='macro')
recall = metrics.recall_score(y_true,y_pred, average='macro')
precision = metrics.precision_score(y_true, y_pred, average='macro')
acc = metrics.accuracy_score(y_true, y_pred)

print("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(acc, f1))

# Save hyperparameters
metrics_log = {
    'F1-Score':f1,
    'Recall':recall,
    'Precision':precision,
    'Accuracy':acc,
    'y_true':y_true,
    'y_pred':y_pred,
    'class_dict':class_dict
}

filename = 'test_metrics'
log_path = '_'.join([args.test_directory,filename])
extension = '.json'
log_path = ''.join([log_path,extension])
log_path = os.path.join(write_path,log_path)

with open(log_path, 'w') as outfile:
    json.dump(metrics_log, outfile)

print('Wrote',log_path)
