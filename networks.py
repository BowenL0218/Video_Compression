### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from .custom_layer import custom_BatchNorm2d, custom_BN
from .Bpgan_VGG_Extractor import Bpgan_VGGLoss
from .quantization_modules import FixedConvTranspose,FixedLinear,FixedConv2d
###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_uniform_(m.weight)
    elif classname.find('ConvTranspose2d') != -1:
        init.kaiming_uniform_(m.weight)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'cus':
        norm_layer = functools.partial(custom_BatchNorm2d,affine=True,track_running_stats=True)
    elif norm_type == 'cus_2':
        norm_layer = functools.partial(custom_BN)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_E(input_nc,ngf,n_downsample=3,C_channel=8,norm='instance', gpu_ids=[],one_D_conv=False, one_D_conv_size=63, max_ngf=512,Conv_type = "C"):
    norm_layer = get_norm_layer(norm_type=norm)
    netE = Encoder(input_nc=input_nc,ngf=ngf,C_channel=C_channel,n_downsampling=n_downsample,norm_layer=norm_layer,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size, max_ngf=max_ngf, Conv_type= Conv_type)
    if len(gpu_ids) >0:
        assert (torch.cuda.is_available())
        netE.cuda(gpu_ids[0])
    netE.apply(weights_init)
    return netE

def define_Decoder(output_nc,ngf,n_downsample=3,C_channel=8,n_blocks_global=9,norm="instance",gpu_ids=[],one_D_conv=False, one_D_conv_size=63, max_ngf = 512, Conv_type="C",Dw_Index=None):
    norm_layer = get_norm_layer(norm_type=norm)
    netDecoder = Decoder(ngf=ngf,C_channel=C_channel,n_downsampling=n_downsample,output_nc=output_nc,n_blocks=n_blocks_global,norm_layer=norm_layer,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size, max_ngf=max_ngf, Conv_type = Conv_type, Dw_Index=Dw_Index)
    if len(gpu_ids) >0:
        assert (torch.cuda.is_available())
        netDecoder.cuda(gpu_ids[0])
    netDecoder.apply(weights_init)
    return netDecoder


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[],one_D_conv=False, one_D_conv_size=63):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)



##############################################################################
# Generator
##############################################################################


class Encoder(nn.Module):
    def __init__(self,input_nc,ngf=64,C_channel=8, n_downsampling=3,norm_layer=nn.BatchNorm2d,padd_type="reflect",one_D_conv=False, one_D_conv_size=63,max_ngf=512, Conv_type="C"):
        assert(n_downsampling>=0)
        super(Encoder, self).__init__()
        activation = nn.ReLU(True)
        if one_D_conv:
            model = [nn.ReflectionPad2d((0,0,(one_D_conv_size-1)//2,(one_D_conv_size-1)//2)),my_Conv(input_nc,int(ngf/2),kernel_size=(one_D_conv_size,1),type=Conv_type, norm_layer=norm_layer, activation=activation),norm_layer(ngf),activation,
                     nn.ReflectionPad2d(3), my_Conv(int(ngf/2),ngf, kernel_size=7, padding=0,type=Conv_type, norm_layer=norm_layer, activation=activation), norm_layer(ngf),activation ]
        else:
            model = [nn.ReflectionPad2d(3), my_Conv(input_nc, ngf, kernel_size=7, padding=0,type=Conv_type, norm_layer=norm_layer, activation=activation), norm_layer(ngf), activation]
        ##downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [my_Conv(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=3, stride=2, padding=1, type=Conv_type, norm_layer=norm_layer, activation=activation),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]
        self.model = nn.Sequential(*model)
        self.projection = nn.Sequential(*[my_Conv(min(ngf*(2**n_downsampling),max_ngf),C_channel,kernel_size=3,stride=1,padding=1, type=Conv_type, norm_layer=norm_layer, activation=activation),norm_layer(C_channel),nn.Sigmoid()])
    def forward(self, input):
        z =  self.model(input)
        return  self.projection(z)
class Decoder(nn.Module):
    def __init__(self,ngf=64,C_channel=8, n_downsampling=3,output_nc=1,n_blocks=9, norm_layer=nn.BatchNorm2d,padding_type="reflect",one_D_conv=False, one_D_conv_size=63,max_ngf=512, Conv_type="C", Dw_Index=None):
        assert (n_blocks>=0)
        super(Decoder, self).__init__()
        activation = nn.ReLU(True)
        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        model = [my_Conv(C_channel,ngf_dim,kernel_size=3,stride=1,padding=1, type=Conv_type, activation=activation, norm_layer=norm_layer),norm_layer(ngf_dim),activation]

        for i in range(n_blocks):
            model += [ResnetBlock(ngf_dim, padding_type=padding_type, activation=activation, norm_layer=norm_layer, Conv_type=Conv_type)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            temp_type = Conv_type
            if Dw_Index == None:
                temp_type = Conv_type
            elif i in Dw_Index and Conv_type == "E":
                temp_type = "E"
            else:
                temp_type = "C"
            model += [my_Deconv(min(ngf * mult,max_ngf), min(ngf * mult //2, max_ngf), type=temp_type, norm_layer=norm_layer, activation=activation),
                      norm_layer(min(ngf * mult //2,max_ngf)), activation]
        if one_D_conv:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, int(ngf/2), kernel_size=7, padding=0),activation,norm_layer(int(ngf/2)),
                      nn.ReflectionPad2d((0,0,(one_D_conv_size-1)//2,(one_D_conv_size-1)//2)),nn.Conv2d(int(ngf/2),output_nc,kernel_size=(one_D_conv_size,1),padding=0),nn.Tanh()]
        else:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)
class quantizer(nn.Module):
    def __init__(self,center,Temp):
        super(quantizer, self).__init__()
        self.center = nn.Parameter(center)
        self.register_parameter('center',self.center)
        self.Temp = Temp
    def forward(self, x, Q_type="None"):
        if Q_type=="Soft":
            W_stack = torch.stack([x for _ in range(len(self.center))],dim=-1)
            W_index = torch.argmin(torch.abs(W_stack-self.center),dim=-1)
            W_hard = self.center[W_index]
            smx = torch.softmax(-1.0*self.Temp*(W_stack-self.center)**2,dim=-1)
            W_soft = torch.einsum('ijklm,m->ijkl',[smx,self.center])
            with torch.no_grad():
                w_bias = (W_hard - W_soft)
            return w_bias + W_soft
        elif Q_type=='None':
            return x
        elif Q_type == 'Hard':
            W_stack = torch.stack([x for _ in range(len(self.center))], dim=-1)
            W_index = torch.argmin(torch.abs(W_stack - self.center), dim=-1)
            W_hard = self.center[W_index]
            return W_hard
    def update_Temp(self,new_temp):
        self.Temp = new_temp
    def update_center(self,new_center):
        self.center = nn.Parameter(new_center)
class vector_quantizer(nn.Module):
    def __init__(self,center,Temp):
        super(vector_quantizer, self).__init__()
        self.center = nn.Parameter(center)
        self.register_parameter('center',self.center)
        self.Temp = Temp
    def forward(self, x,Q_type='None'):
        x_ = x.view(x.shape[0],-1,4)
        if Q_type=="Soft":
            W_stack = torch.stack([x_ for _ in range(len(self.center))],dim=-1)
            E = torch.norm(W_stack - self.center.transpose(1,0),2,dim=-2)
            W_index = torch.argmin(E,dim=-1)
            W_hard = self.center[W_index]
            smx = torch.softmax(-1.0*self.Temp*E**2,dim=-1)
            W_soft = torch.einsum('ijk,km->ijm',[smx,self.center])
            with torch.no_grad():
                w_bias = (W_hard - W_soft)
            output = w_bias + W_soft
        elif Q_type=='None':
            output = x_
        elif Q_type == 'Hard':
            W_stack = torch.stack([x_ for _ in range(len(self.center))], dim=-1)
            E = torch.norm(W_stack - self.center.transpose(1, 0), 2, dim=-2)
            W_index = torch.argmin(E, dim=-1)
            W_hard = self.center[W_index]
            output =  W_hard
        return output.view(x.shape)
    def update_Temp(self,new_temp):
        self.Temp = new_temp
    def update_center(self,new_center):
        self.center = nn.Parameter(new_center)
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, Conv_type="C"):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout,Conv_type=Conv_type)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, Conv_type="C"):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [my_Conv(dim, dim, kernel_size=3, type=Conv_type
            ,padding=p,norm_layer=norm_layer, activation=activation),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [my_Conv(dim, dim, kernel_size=3, padding=p,type=Conv_type, norm_layer=norm_layer, activation=activation),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False,one_D_conv=False, one_D_conv_size=63):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D-1):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)
        netD = NLayerDiscriminator(input_nc,ndf,n_layers,norm_layer,use_sigmoid,getIntermFeat,one_D_conv=one_D_conv,one_D_conv_size=one_D_conv_size)
        if getIntermFeat:
            for j in range(n_layers+2):
                setattr(self, 'scale' + str(num_D-1) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
        else:
            setattr(self,'layer'+str(num_D-1),netD.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.one_D_conv = one_D_conv
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input

        for i in range(num_D-1):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        if self.getIntermFeat:
            model = [getattr(self, 'scale' + str(num_D - 1) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
        else:
            model = getattr(self, 'layer' + str(num_D - 1))
        if self.one_D_conv:
            result.append(self.singleD_forward(model, input))
        else:
            result.append(self.singleD_forward(model,input_downsampled))
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False,one_D_conv=False, one_D_conv_size=63):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        if one_D_conv:
            sequence = [[nn.Conv2d(input_nc,int(ndf/2),kernel_size=(one_D_conv_size,1),padding=(0,(one_D_conv_size-1)//2)),nn.LeakyReLU(0.2,True),
                         nn.Conv2d(int(ndf/2),ndf,kernel_size=kw,stride=2,padding=padw),nn.LeakyReLU(0.2,True)]]
        else:
            sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
def my_Conv(nin, nout, kernel_size = 3, stride = 1, padding=1, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True), type="C"):
    if type == "C":
        return nn.Sequential(nn.Conv2d(nin,nout,kernel_size=kernel_size,stride=stride,padding=padding))
    elif type == "E" or "E_test":
        return nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, stride=stride),
            norm_layer(nin),
            activation,
            nn.Conv2d(nin,nout,kernel_size=1)
        )
    elif type == "FC":
        return nn.Sequential(FixedConv2d(nin,nout,kernel_size=kernel_size,stride=stride, padding=padding))
    elif type == "FE":
        return nn.Sequential(
            FixedConv2d(nin,nin,kernel_size=kernel_size,padding=padding,groups=nin,stride=stride),
            norm_layer(nin),
            activation,
            FixedConv2d(nin,nout,kernel_size=1)
        )
    else:
        raise NotImplementedError
def my_Deconv(nin, nout, norm_layer=nn.BatchNorm2d,activation=nn.ReLU(True), type = "C"):
    if type == "C":
        return nn.Sequential(nn.ConvTranspose2d(nin, nout, kernel_size=3, stride=2, padding=1, output_padding=1))
    elif type == "E":
        return nn.Sequential(
            upsample_pad(),
            nn.Conv2d(nin,nin,kernel_size=3,stride=1, padding=1,groups=nin),
            norm_layer(nin),
            activation,
            nn.Conv2d(nin, nout, kernel_size=1)
        )
    elif type == "E_test":
        return nn.Sequential(upsample_pad(),
                nn.Conv2d(nin, nout, kernel_size=3, stride=1, padding=1))
    elif type == "FC":
        return nn.Sequential(FixedConvTranspose(nin,out,kernel_size=3,stride=2,padding=1,output_padding=1))
    elif type == "FE":
        return nn.Sequential(
            upsample_pad(),
            FixedConv2d(nin, nin, kernel_size=3, stride=1, padding=1, groups=nin),
            norm_layer(nin),
            activation,
            FixedConv2d(nin, nout, kernel_size=1)
        )
    else:
        raise NotImplementedError

class upsample(nn.Module):
    def __init__(self):
        super(upsample, self).__init__()
        self.upsample =  lambda x: torch.nn.functional.interpolate(x, scale_factor=2)
    def forward(self, x):
        return self.upsample(x)
class upsample_pad(nn.Module):
    def __init__(self):
        super(upsample_pad, self).__init__()
    def forward(self,x):
        out = torch.zeros(x.shape[0],x.shape[1],2*x.shape[2],2*x.shape[3],device = x.device,dtype = x.dtype)
        out[:,:,0::2,:][:,:,:,0::2]=x
        return out
