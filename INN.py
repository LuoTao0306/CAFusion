import torch
import torch.nn as nn




import module_util as mutil
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 6, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 6, 6, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 6, 6, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 6, 6, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 6, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


# 可逆块
class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=2.0):
        super().__init__()
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(6, 6)
        # η
        self.y = subnet_constructor(6, 6)
        # φ
        self.f = subnet_constructor(6, 6)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        if not rev:
            t2 = self.f(x)
            y1 = x + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x + t1
        else:
            s1, t1 = self.r(x), self.y(x)
            y2 = (x - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x - t2)

        return y1


# 可逆神经网络
class ReversibleNetwork(nn.Module):
    def __init__(self, num_blocks=1):
        super(ReversibleNetwork, self).__init__()
        self.blocks = nn.ModuleList([INV_block() for _ in range(num_blocks)])

    def forward(self, x, rev=False):
        if not rev:
            for block in self.blocks:
                x = block(x)
        else:
            for block in reversed(self.blocks):
                x = block(x, rev=True)
        return x
