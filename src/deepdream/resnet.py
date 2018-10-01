import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch import load

__all__ = ['resnet50',]

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

# --- Original resnet ----------------------------------- #
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
# ------------------------------------------------------- #

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MyResnet(models.resnet.ResNet):
    def forward(self, x, n_layer=3,neuron_unit=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        for i_layer in range(n_layer):
            x = layers[i_layer](x)

        # --- Removed from original for deep dream ------- #
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        # ------------------------------------------------ #
        return x
    
#         if neuron_unit is None:
#             return x
#         else:
#             #return x[:,neuron_unit,:,:]
#             return x[:,neuron_unit,:,:]


# Original resnet.
# See: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def resnet50(pretrained=False,sound2image=False,local=False,sound2image_local=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MyResnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    
    if sound2image:
        if not local:

            weights_path = '/data/datasets/sound_datasets/pytorch_UrbanSound8K/saved_models/cvr_final_project/resnet50_v3_full-training.pt'
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(load(weights_path,
                                       map_location=lambda storage, 
                                       loc: storage)
                                 )
            print(weights_path)
    
        else:
            weights_path = '../data/models/resnet50_v2_melspect_15_968.pt'
            weights_path = '../data/models/resnet50_v3_full-training.pt'
            model.fc = nn.Linear(model.fc.in_features, 10)
            model.load_state_dict(load(weights_path,
                                       map_location=lambda storage,
                                       loc: storage)
                                 )
            print(weights_path)
    return model

