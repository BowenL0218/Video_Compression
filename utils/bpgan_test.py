### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.nnls import nnls
import ntpath
import os
import imageio
import librosa
from torch.autograd import Variable
import numpy as np
import torch
from sklearn.cluster import KMeans
from models.networks import  quantizer
if_quantization = True

output_path = "./forward_test/"
opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
how_many_infer = 200
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
inverse_matrix = librosa.filters.mel(sr=opt.sampling_ratio,n_fft=opt.n_fft,n_mels=opt.n_mels)
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
for i, data in enumerate(dataset):
    if i>how_many_infer:
        break
    with torch.no_grad():
        input = Variable(data['label']).cuda()
        vector = model.netE(input)
        if i == 0:
            vector_dis = vector.detach().cpu().numpy()
        else:
            vector_tem = vector.detach().cpu().numpy()
            vector_dis = np.concatenate((vector_dis, vector_tem), axis=0)
vector_dis = vector_dis.reshape(-1)
kmeans = KMeans(n_clusters=opt.n_cluster,n_jobs=-1).fit(vector_dis.reshape(-1,1))
center = kmeans.cluster_centers_.flatten()
center = torch.Tensor(kmeans.cluster_centers_).cuda()
Quantizer = quantizer(center=center.flatten(),Temp=10)
Quantizer = Quantizer.cuda()
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    input_label, image = model.encode_input(Variable(data['label']), infer=True)
    if if_quantization == True:
        generated = model.netDecoder(Quantizer(model.netE(input_label),"Hard"))
    else:
        generated = model.netDecoder(model.netE(input_label))
    for index in range(data['label'].shape[0]):
        gen_img = generated[index].detach().cpu().numpy()
        org_img = input_label[index].detach().cpu().numpy()
        gen_img = gen_img * 0.5 - 0.5
        org_img = org_img * 0.5 - 0.5
        gen_img = np.exp(10 * (gen_img))
        org_img = np.exp(10 * (org_img))
        inverse_gen = np.abs(nnls(inverse_matrix, gen_img[0, :, :]))
        inverse_org = np.abs(nnls(inverse_matrix, org_img[0, :, :]))
        inverse_gen_img = (inverse_gen / np.max(inverse_gen.flatten()) * 65535).astype(np.uint16)
        inverse_org_img = (inverse_org / np.max(inverse_org.flatten()) * 65535).astype(np.uint16)
        short_path = ntpath.basename(data['path'][index])
        name_ = os.path.splitext(short_path)[0]
        imageio.imwrite(os.path.join(output_path, name_ + '_syn.png'), inverse_gen_img)

        imageio.imwrite(os.path.join(output_path, name_ + '_real.png'), inverse_org_img)
