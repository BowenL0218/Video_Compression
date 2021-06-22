import torch
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        ##m.weight.data.normal_(0.0, 0.02)
        init.kaiming_uniform_(m.weight)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
class Encoder(nn.Module):
    def __init__(self,input_nc,ngf=64,C_channel=8, n_downsampling=4,padd_type="reflect",one_D_conv=False, one_D_conv_size=63,max_ngf=512):
        assert(n_downsampling>=0)
        super(Encoder, self).__init__()
        activation = nn.ReLU(True)
        norm_layer = get_norm_layer("instance")
        if one_D_conv:
            model = [nn.ReflectionPad2d((0,0,(one_D_conv_size-1)//2,(one_D_conv_size-1)//2)),nn.Conv2d(input_nc,int(ngf/2),kernel_size=(one_D_conv_size,1)),norm_layer(ngf),activation,
                     nn.ReflectionPad2d(3), nn.Conv2d(int(ngf/2),ngf, kernel_size=7, padding=0), norm_layer(ngf),activation ]
        else:
            model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ##downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=3, stride=2, padding=1),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]
        self.model = nn.Sequential(*model)
        self.projection = nn.Sequential(*[nn.Conv2d(min(ngf*(2**n_downsampling),max_ngf),C_channel,kernel_size=3,stride=1,padding=1),norm_layer(C_channel),nn.Sigmoid()])
    def forward(self, input):
        z =  self.model(input)
        return  self.projection(z)
class Encoder2(nn.Module):
    def __init__(self,input_nc, ngf = 64, C_channel=8,n_downsampling=4, padd_type="reflect",max_ngf = 512):
        assert(n_downsampling>=0)
        super(Encoder2, self).__init__()
        activation = nn.LeakyReLU(0.2)
        norm_layer = get_norm_layer("batch")
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=3, stride=2, padding=1),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]
        self.model = nn.Sequential(*model)
        self.projection = nn.Sequential(*[nn.Linear(8*8*512,2048),norm_layer(2048),activation,nn.Linear(2048,512),norm_layer(512),norm_layer(512),nn.Sigmoid()])
    def forward(self,input):
        z = self.model(input)
        z = z.view(input.shape[0],-1)
        return self.projection(z)