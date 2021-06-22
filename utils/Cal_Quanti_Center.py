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

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 32  # test code only supports batchSize = 1
opt.serial_batches = False  # no shuffle
opt.no_flip = True  # no flip
how_many_infer = 200
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)

inverse_matrix = librosa.filters.mel(sr=opt.sampling_ratio,n_fft=opt.n_fft,n_mels=opt.n_mels)
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
np.save('./Quantization_Center.npy',center)
