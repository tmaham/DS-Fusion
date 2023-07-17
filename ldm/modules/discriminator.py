from torch import nn
import pdb 

class Discriminator(nn.Module):
    def __init__(self, bnorm=True, leakyparam=0.0, 
                 bias=False, generic=False, use_leaky=True,
                 use_bn=True):
        
        super(Discriminator, self).__init__()

        self.bnorm = bnorm
        self.generic = generic
        dim = 64
        channels = 4

        if use_leaky:
            self.relu = nn.LeakyReLU(leakyparam, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.sig = nn.Sigmoid()
        self.bn2 = nn.BatchNorm2d(dim*2)
        self.bn3 = nn.BatchNorm2d(dim*4)
        self.bn4 = nn.BatchNorm2d(dim*8)

        self.layer1 = nn.Conv2d(channels, dim, 4, 2, 1, bias=bias)
        self.layer2 = nn.Conv2d(dim, dim*2, 4, 2, 1, bias=bias)
        self.layer3 = nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=bias)
        self.layer4 = nn.Conv2d(dim*4, dim*8, 4, 2, 1, bias=bias)
        if generic:
            self.layer5 = nn.Conv2d(dim*8, 26, 2, 1, 0, bias=bias)
        else:
            self.layer5 = nn.Conv2d(dim*8, 1, 2, 1, 0, bias=bias)

    def forward(self, input, letter):
    
        out1 = self.relu(self.layer1(input))
        
        if self.bnorm:
            out2 = self.relu(self.bn2(self.layer2(out1)))
            out3 = self.relu(self.bn3(self.layer3(out2)))
            out4= self.relu(self.bn4(self.layer4(out3)))
        else:
            out2 = self.relu(self.layer2(out1))
            out3 = self.relu(self.layer3(out2))
            out4= self.relu(self.layer4(out3))

        out5 = self.sig(self.layer5(out4))
        out5 = out5.flatten()

        if self.generic: 
            out5 = out5[letter].mean()
        else:
            out5 = out5.mean()

        return out5
