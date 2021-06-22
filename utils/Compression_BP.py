from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
from util.visualizer import Visualizer
from torch.autograd import Variable
import torch.nn
from sklearn.cluster import KMeans
from models.networks import  quantizer
from models.Bpgan_VGG_Extractor import Bpgan_VGGLoss
from util.nnls import nnls
import ntpath
import os
import imageio
import librosa
opt = TestOptions().parse(save=False)
opt.nThreads = 4  # test code only supports nThreads = 1
opt.batchSize = 128  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.quantize_type = 'scalar'
opt.model ="Bpgan_GAN"
how_many_infer = 200
if_quantization = False
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

device = torch.device("cuda")

Critiretion = torch.nn.MSELoss().to(device)
## ADMM setting
lr = 0.05
BP_iter = 250
alpha = 15
mu = 0.001

output_path = "./BP_Results/"
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
inverse_matrix = librosa.filters.mel(sr=opt.sampling_ratio,n_fft=opt.n_fft,n_mels=opt.n_mels)
if opt.feature_loss == True:
    VGG_loss = Bpgan_VGGLoss(d=40, sampling_ratio=opt.sampling_ratio, n_fft=opt.n_fft,n_mels=opt.n_mels,path=None)
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


for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    input_label, image = model.encode_input(Variable(data['label']), infer=True)
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    for itera in range(BP_iter):
        generated_img = model.netDecoder.forward(latent_vector)
        optmize_Com.zero_grad()
        if opt.feature_loss == True:
            vgg_loss = VGG_loss(generated_img,input_label)
        else:
            vgg_loss = 0
        mse_loss = Critiretion(generated_img, input_label)
        Com_loss = vgg_loss + alpha * mse_loss
        Com_loss.backward()
        optmize_Com.step()
    if if_quantization:
        generated_img = model.netDecoder(Quantizer(latent_vector, "Hard"))
    else:
        generated_img = model.netDecoder(latent_vector)
    for index in range(input_label.shape[0]):
        gen_img = generated_img[index].detach().cpu().numpy()
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

