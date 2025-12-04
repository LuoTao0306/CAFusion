import torch.nn as nn
class Basic3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic3x3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes, kernel_size=3, stride=stride,
               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class Basic1x1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, stride)
        self.act=nn.Hardsigmoid()
    def forward(self, x):
        out = self.conv1(x)
        out=self.act(out)
        return out