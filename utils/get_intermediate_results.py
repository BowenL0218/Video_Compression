from options.test_options import TestOptions
from models.models import create_model
import torch.nn
import os
import imageio
import numpy as np
import scipy
from util.visualizer import Visualizer
opt = TestOptions().parse(save=False)
opt.quantize_type = 'scalar'
if_quantization = False
input_path ="./test.png"
output_path = "./output/"
model = create_model(opt)
device = torch.device("cuda")
image = imageio.imread(input_path)
image = np.array(image)/65535
image = torch.Tensor(image).to(device)
if os.path.exists(output_path) is False:
    os.makedirs(output_path)
x = image.view(1,1,image.shape[0],image.shape[1])
parameter_dict = {}
operation_list = []
i = 0
for layer in model.netE.model:
    x = layer(x)
    with torch.no_grad():
        parameter_dict.update({str(i)+str(layer):x.cpu().numpy()})
        operation_list.append(str(layer))
    i = i +1
for layer in model.netE.projection:
    x = layer(x)
    with torch.no_grad():
        parameter_dict.update({str(i)+str(layer):x.cpu().numpy()})
        operation_list.append(str(layer))
    i = i +1
scipy.io.savemat(os.path.join(output_path,'netE_operation.mat'),{'opt':operation_list})
scipy.io.savemat(os.path.join(output_path,'netE_parameter.mat'),parameter_dict)
parameter_dict = {}
operation_list = []
i = 0
for layer in model.netDecoder.model:
    x = layer(x)
    with torch.no_grad():
        parameter_dict.update({str(i)+str(layer):x.cpu().numpy()})
        operation_list.append(str(layer))
    i = i +1
scipy.io.savemat(os.path.join(output_path,'netDecoder_operation.mat'),{'opt':operation_list})
scipy.io.savemat(os.path.join(output_path,'netDecoer_parameter.mat'),parameter_dict)