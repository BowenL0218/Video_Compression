### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

def create_model(opt):
    if opt.model == 'Bpgan_GAN':
    	from .Audio_GAN_model import Audio_GAN_Model
    	model = Audio_GAN_Model()
    elif opt.model == 'Bpgan_GAN_Q':
        from .Audio_GAN_Q_model import Audio_GAN_Q_Model
        model = Audio_GAN_Q_Model()
    elif opt.model == 'Dis_Bpgan_GAN':
        from .Bpgan_GAN_Distillation_model import Bpgan_GAN_DIS_Model
        model = Bpgan_GAN_DIS_Model()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
