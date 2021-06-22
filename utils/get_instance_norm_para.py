from options.test_options import TestOptions
from models.models import create_model
import torch.nn
import os
from torch.autograd import Variable
import imageio
import numpy as np
import ntpath
from data.data_loader import CreateDataLoader
import scipy
from collections import OrderedDict
from util.visualizer import Visualizer
opt = TestOptions().parse(save=False)
opt.quantize_type = 'scalar'
if_quantization = False
input_path ="./test.png"
output_path = "./output/"
model = create_model(opt)
opt.batchSize = 512
device = torch.device("cuda")
image = imageio.imread(input_path)
image = np.array(image)/65535
image = torch.Tensor(image).to(device)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
if os.path.exists(output_path) is False:
    os.makedirs(output_path)
x = image.view(1,1,image.shape[0],image.shape[1])
netE_model_instance = OrderedDict()
netDecoer_model_instance = OrderedDict()
i = 0
for layer in model.netE.model:
    x = layer(x)
    if layer.__class__.__name__ == "InstanceNorm2d":
        base = dict()
        base['mean'] = []
        base['std'] = []
        netE_model_instance.update({i: base})
        i = i + 1

for layer in model.netE.projection:
    x = layer(x)
    if layer.__class__.__name__ == "InstanceNorm2d":
        base = dict()
        base['mean'] = []
        base['std'] = []
        netE_model_instance.update({i:base})
        i = i +1
i = 0
for layer in model.netDecoder.model:
    x = layer(x)
    if layer.__class__.__name__ == "InstanceNorm2d":
        base = dict()
        base['mean'] = []
        base['std'] = []
        netDecoer_model_instance.update({i:base})
        i = i +1
base = dict()
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
with torch.no_grad():
    for j, data in enumerate(dataset):
        i = 0
        if j >1:
            break
        input = Variable(data['label']).cuda()
        x = input
        for layer in model.netE.model:
            if layer.__class__.__name__ == "InstanceNorm2d":
                temp_tensor = x.view(x.shape[0], x.shape[1], -1)
                mean = torch.mean(temp_tensor, keepdim = True, dim = 2)
                std = torch.mean((temp_tensor - mean) ** 2, dim = 2)
                netE_model_instance[i]['mean'] = mean
                netE_model_instance[i]['std'] = std
                i = i + 1
            x = layer(x)
        for layer in model.netE.projection:
            if layer.__class__.__name__ == "InstanceNorm2d":
                temp_tensor = x.view(x.shape[0],x.shape[1],-1)
                mean = torch.mean(temp_tensor,keepdim= True, dim=2)
                std = torch.mean((temp_tensor - mean)**2,dim=2)
                netE_model_instance[i]['mean'] = mean
                netE_model_instance[i]['std'] = std
                i = i +1
            x = layer(x)
        i = 0
        for layer in model.netDecoder.model:
            if layer.__class__.__name__ == "InstanceNorm2d":
                temp_tensor = x.view(x.shape[0],x.shape[1],-1)
                mean = torch.mean(temp_tensor,keepdim= True, dim=2)
                std = torch.mean((temp_tensor - mean)**2,dim=2)
                netDecoer_model_instance[i]['mean'] = mean
                netDecoer_model_instance[i]['std'] = std
                i = i +1
            x = layer(x)
with torch.no_grad():
    i = 0
    for layer in model.netE.model:
        if layer.__class__.__name__ == "InstanceNorm2d":
            netE_model_instance[i]['mean'] = torch.mean(netE_model_instance[i]['mean'].squeeze(),dim=0)
            netE_model_instance[i]['std'] = torch.mean(netE_model_instance[i]['std'].squeeze(),dim=0)
            i = i + 1

    for layer in model.netE.projection:
        if layer.__class__.__name__ == "InstanceNorm2d":
            netE_model_instance[i]['mean'] = torch.mean(netE_model_instance[i]['mean'].squeeze(),dim=0)
            netE_model_instance[i]['std'] = torch.mean(netE_model_instance[i]['std'].squeeze(),dim=0)
            i = i + 1
    i = 0
    for layer in model.netDecoder.model:
        if layer.__class__.__name__ == "InstanceNorm2d":
            netDecoer_model_instance[i]['mean'] = torch.mean(netDecoer_model_instance[i]['mean'].squeeze(),dim=0)
            netDecoer_model_instance[i]['std'] = torch.mean(netDecoer_model_instance[i]['mean'].std(),dim=0)
            i = i + 1

for j, data in enumerate(dataset):
    i = 0
    input = Variable(data['label']).cuda()
    x = input
    with torch.no_grad():
        for layer in model.netE.model:
            if layer.__class__.__name__ == "InstanceNorm2d":
                mean = netE_model_instance[i]['mean']
                var = netE_model_instance[i]['std']
                x = (x - mean.view(1,-1,1,1)) / torch.sqrt(var.view(1,-1,1,1) + layer.eps)
                i = i + 1
            else:
                x = layer(x)
        for layer in model.netE.projection:
            if layer.__class__.__name__ == "InstanceNorm2d":
                mean = netE_model_instance[i]['mean']
                var = netE_model_instance[i]['std']
                x = (x - mean.view(1,-1,1,1)) / torch.sqrt(var.view(1,-1,1,1) + layer.eps)
                i = i + 1
            else:
                x = layer(x)
        i = 0
        for layer in model.netDecoder.model:
            if layer.__class__.__name__ == "InstanceNorm2d":
                mean = netDecoer_model_instance[i]['mean']
                var = netDecoer_model_instance[i]['std']
                x = (x - mean.view(1,-1,1,1)) / torch.sqrt(var.view(1,-1,1,1) + layer.eps)
                i = i + 1
            else:
                x = layer(x)
        generated = x
    for index in range(data['label'].shape[0]):
        gen_img = generated[index][0].detach().cpu().numpy()
        org_img = input[index][0].detach().cpu().numpy()
        gen_img = gen_img * 0.5 - 0.5
        org_img = org_img * 0.5 - 0.5
        gen_img = np.exp(10 * (gen_img))
        org_img = np.exp(10 * (org_img))
        org_gen_img = (gen_img / np.max(gen_img.flatten()) * 65535).astype(np.uint16)
        org_org_img = (org_img / np.max(org_img.flatten()) * 65535).astype(np.uint16)
        short_path = ntpath.basename(data['path'][index])
        name_ = os.path.splitext(short_path)[0]
        imageio.imwrite(os.path.join(output_path, name_ + '_syn.png'), org_gen_img)
        imageio.imwrite(os.path.join(output_path, name_ + '_real.png'), org_org_img)

for j, data in enumerate(dataset):
    i = 0
    input = Variable(data['label']).cuda()
    x = input
    with torch.no_grad():
        for layer in model.netE.model:
            if layer.__class__.__name__ == "InstanceNorm2d":
                N, C, H, W = x.shape
                temp_tensor = x.view(x.shape[0], x.shape[1], -1)
                mean = torch.mean(temp_tensor, keepdim = True, dim = 2)
                var = torch.mean((temp_tensor - mean) ** 2, dim = 2)
                x = (x - mean.view(N,C,1,1))/torch.sqrt( var.view(N,C,1,1)+ layer.eps)
            else:
                x = layer(x)
        for layer in model.netE.projection:
            if layer.__class__.__name__ == "InstanceNorm2d":
                N, C, H, W = x.shape
                temp_tensor = x.view(x.shape[0], x.shape[1], -1)
                mean = torch.mean(temp_tensor, keepdim = True, dim = 2)
                var = torch.mean((temp_tensor - mean) ** 2, dim = 2)
                x = (x - mean.view(N, C, 1, 1)) / torch.sqrt(var.view(N, C, 1, 1) + layer.eps)
            else:
                x = layer(x)
        i = 0
        for layer in model.netDecoder.model:
            if layer.__class__.__name__ == "InstanceNorm2d":
                N, C, H, W = x.shape
                temp_tensor = x.view(x.shape[0], x.shape[1], -1)
                mean = torch.mean(temp_tensor, keepdim = True, dim = 2)
                var = torch.mean((temp_tensor - mean) ** 2, dim = 2)
                x = (x - mean.view(N, C, 1, 1)) / torch.sqrt(var.view(N, C, 1, 1) + layer.eps)
            else:
                x = layer(x)
        generated = x
    for index in range(data['label'].shape[0]):
        gen_img = generated[index][0].detach().cpu().numpy()
        org_img = input[index][0].detach().cpu().numpy()
        gen_img = gen_img * 0.5 - 0.5
        org_img = org_img * 0.5 - 0.5
        gen_img = np.exp(10 * (gen_img))
        org_img = np.exp(10 * (org_img))
        org_gen_img = (gen_img / np.max(gen_img.flatten()) * 65535).astype(np.uint16)
        org_org_img = (org_img / np.max(org_img.flatten()) * 65535).astype(np.uint16)
        short_path = ntpath.basename(data['path'][index])
        name_ = os.path.splitext(short_path)[0]
        imageio.imwrite(os.path.join(output_path, name_ + '_syn.png'), org_gen_img)
        imageio.imwrite(os.path.join(output_path, name_ + '_real.png'), org_org_img)
