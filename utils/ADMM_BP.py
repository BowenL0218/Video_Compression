from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
from util.visualizer import Visualizer
from torch.autograd import Variable
import torch.nn
import librosa
from sklearn.cluster import KMeans
from models.networks import  quantizer
from VGG_Extractor import VGGLoss
from util.nnls import  nnls
opt = TestOptions().parse(save=False)
opt.nThreads = 4  # test code only supports nThreads = 1
opt.batchSize = 128  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.quantize_type = 'scalar'
opt.model ="pix2pixHDQ"
##opt.dataroot ='./dataset/timit_fast_test'
##opt.how_many =30

device = torch.device("cuda")

Critiretion = torch.nn.MSELoss().to(device)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
lr = 0.05
Com_iter = 250
alpha = 15
mu = 0.001
A = librosa.filters.mel(sr=16000,n_fft=512,n_mels=40)
B = librosa.filters.mel(sr=16000,n_fft=512,n_mels=128)
C = A.dot(np.linalg.pinv(B))
Transform_tensor = torch.Tensor(C).cuda()
VGG_Loss = VGGLoss(d=40)
VGG_Loss.load_param('VGG_Extractor.pt')

for i, data in enumerate(dataset):
    if i>20:
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

#### 8bit
# kmeans = KMeans(n_clusters=256,n_jobs=-1).fit(vector_dis.reshape(-1,1))
# center = kmeans.cluster_centers_.flatten()
# center = torch.Tensor(kmeans.cluster_centers_).cuda()
# Quantizer8 = quantizer(center=center.flatten(),Temp=10)
# Quantizer8 = Quantizer8.cuda()
#### 6 bit
kmeans = KMeans(n_clusters=64,n_jobs=-1).fit(vector_dis.reshape(-1,1))
center = kmeans.cluster_centers_.flatten()
center = torch.Tensor(kmeans.cluster_centers_).cuda()
Quantizer6 = quantizer(center=center.flatten(),Temp=10)
Quantizer6 = Quantizer6.cuda()
#### 5 bit
kmeans = KMeans(n_clusters=32,n_jobs=-1).fit(vector_dis.reshape(-1,1))
center = kmeans.cluster_centers_.flatten()
center = torch.Tensor(kmeans.cluster_centers_).cuda()
Quantizer5 = quantizer(center=center.flatten(),Temp=10)
Quantizer5 = Quantizer5.cuda()
#### 4bit
kmeans = KMeans(n_clusters=16,n_jobs=-1).fit(vector_dis.reshape(-1,1))
center = kmeans.cluster_centers_.flatten()
center = torch.Tensor(kmeans.cluster_centers_).cuda()
Quantizer4 = quantizer(center=center.flatten(),Temp=10)
Quantizer4 = Quantizer4.cuda()
#### 3bit
kmeans = KMeans(n_clusters=8,n_jobs=-1).fit(vector_dis.reshape(-1,1))
center = kmeans.cluster_centers_.flatten()
center = torch.Tensor(kmeans.cluster_centers_).cuda()
Quantizer3 = quantizer(center=center.flatten(),Temp=10)
Quantizer3 = Quantizer3.cuda()
#### 2bit
kmeans = KMeans(n_clusters=4,n_jobs=-1).fit(vector_dis.reshape(-1,1))
center = kmeans.cluster_centers_.flatten()
center = torch.Tensor(kmeans.cluster_centers_).cuda()
Quantizer2 = quantizer(center=center.flatten(),Temp=10)
Quantizer2 = Quantizer2.cuda()

PSNR_6_track=[]
PSNR_ADMM_6_track=[]
PSNR_IHT_6_track = []
PSNR_5_track=[]
PSNR_ADMM_5_track=[]
PSNR_IHT_5_track=[]
PSNR_4_track=[]
PSNR_ADMM_4_track=[]
PSNR_IHT_4_track =[]
PSNR_3_track=[]
PSNR_ADMM_3_track=[]
PSNR_IHT_3_track=[]
PSNR_2_track=[]
PSNR_ADMM_2_track=[]
PSNR_IHT_2_track =[]
PSNR_BP_track = []
Sub_step = 10
total_size = vector[0].numel()
total_size = vector[0].numel()
K = int(total_size/Sub_step)
step_iter = int(Com_iter/Sub_step)
for i, data in enumerate(dataset):
    if i>=2:
        break
    input_label, image = model.encode_input(Variable(data['label']), infer=True)
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(),requires_grad=True)
        latent_vector.data = Compressed_p.clone()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    z_img = torch.einsum("mj,idjk->idmk", [Transform_tensor, input_label])
    z_img = z_img[:, 0, :, :]
    z_img = z_img.transpose(1, 2)
    for iter in range(Com_iter):
        generated_img = model.netDecoder(latent_vector)
        optmize_Com.zero_grad()
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        Com_loss = vgg_loss + alpha * mse_loss
        Com_loss.backward()
        optmize_Com.step()
    target_loss = vgg_loss + alpha * mse_loss
    PSNR_BP_track.append(target_loss.detach().cpu().numpy())

    # ### 8bit
    # g_img = model.netDecoder(Quantizer8(latent_vector, "Hard"))
    # with torch.no_grad():
    #     z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
    #     z_g = z_g[:, 0, :, :]
    #     z_g = z_g.transpose(1, 2)
    #     vgg_loss = VGG_Loss(z_g, z_img)
    #     mse_loss = Critiretion(generated_img, input_label)
    #     target_loss = vgg_loss + alpha * mse_loss
    #     PSNR_8_track.append(target_loss.detach().numpy())
    ### 6bit

    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer6(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_6_track.append(target_loss.detach().cpu().numpy())
    ###5bit

    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer5(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_5_track.append(target_loss.detach().cpu().numpy())
    ###4bit

    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer4(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_4_track.append(target_loss.detach().cpu().numpy())
    ##3bit

    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer3(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_3_track.append(target_loss.detach().cpu().numpy())
    ###2bit
    generated_img = model.netDecoder(Quantizer2(latent_vector, "Hard"))
    with torch.no_grad():
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_2_track.append(target_loss.detach().cpu().numpy())
    # ### 8bit
    # with  torch.no_grad():
    #     Compressed_p = model.module.netE.forward(input_label)
    #     vector_shape = Compressed_p.shape
    #     latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(),requires_grad=True)
    #     latent_vector.data = Compressed_p.clone()
    #     Z = Quantizer8(latent_vector, "Hard")
    #     eta = torch.zeros(latent_vector.shape).cuda()
    # for itera in range(Com_iter):
    #     generated_img = model.netDecoder.forward(latent_vector)
    #     optmize_Com.zero_grad()
    #     z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
    #     z_g = z_g[:, 0, :, :]
    #     z_g = z_g.transpose(1, 2)
    #     vgg_loss = VGG_Loss(z_g, z_img)
    #     mse_loss = Critiretion(generated_img, input_label)
    #     Com_loss = vgg_loss + alpha*mse_loss+mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2/latent_vector.shape[0]
    #     Com_loss.backward()
    #     optmize_Com.step()
    #     with torch.no_grad():
    #         Z = Quantizer8(latent_vector + eta, "Hard")
    #         eta = eta + latent_vector - Z
    # g_img = model.netDecoder(Quantizer8(latent_vector, "Hard"))
    # with torch.no_grad():
    #     z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
    #     z_g = z_g[:, 0, :, :]
    #     z_g = z_g.transpose(1, 2)
    #     vgg_loss = VGG_Loss(z_g, z_img)
    #     mse_loss = Critiretion(generated_img, input_label)
    #     target_loss = vgg_loss + alpha * mse_loss
    #     PSNR_ADMM_8_track.append(target_loss.detach().numpy())
    ## 6bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        Z = Quantizer6(latent_vector, "Hard")
        eta = torch.zeros(latent_vector.shape).cuda()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    for itera in range(Com_iter):
        generated_img = model.netDecoder.forward(latent_vector)
        optmize_Com.zero_grad()
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                   latent_vector.shape[0]
        Com_loss.backward()
        optmize_Com.step()
        with torch.no_grad():
            Z = Quantizer6(latent_vector + eta, "Hard")
            eta = eta + latent_vector - Z

    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer6(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_ADMM_6_track.append(target_loss.detach().cpu().numpy())
    ### 5bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        Z = Quantizer5(latent_vector, "Hard")
        eta = torch.zeros(latent_vector.shape).cuda()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    for itera in range(Com_iter):
        generated_img = model.netDecoder.forward(latent_vector)
        optmize_Com.zero_grad()
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                   latent_vector.shape[0]
        Com_loss.backward()
        optmize_Com.step()
        with torch.no_grad():
            Z = Quantizer5(latent_vector + eta, "Hard")
            eta = eta + latent_vector - Z

    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer5(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_ADMM_5_track.append(target_loss.detach().cpu().numpy())
    ### 4bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        Z = Quantizer4(latent_vector, "Hard")
        eta = torch.zeros(latent_vector.shape).cuda()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    for itera in range(Com_iter):
        generated_img = model.netDecoder.forward(latent_vector)
        optmize_Com.zero_grad()
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                   latent_vector.shape[0]
        Com_loss.backward()
        optmize_Com.step()
        with torch.no_grad():
            Z = Quantizer4(latent_vector + eta, "Hard")
            eta = eta + latent_vector - Z
    torch.save(Quantizer4(latent_vector,"Hard"),"./Intermediate3/vector_"+str(i)+".pt")
    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer4(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_ADMM_4_track.append(target_loss.detach().cpu().numpy())
    torch.save(input_label,"./Intermediate2/Image.pt")
    torch.save(generated_img,"./Intermediate2/OImage.pt")
    ### 3bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        Z = Quantizer3(latent_vector, "Hard")
        eta = torch.zeros(latent_vector.shape).cuda()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    for itera in range(Com_iter):
        generated_img = model.netDecoder.forward(latent_vector)
        optmize_Com.zero_grad()
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                   latent_vector.shape[0]
        Com_loss.backward()
        optmize_Com.step()
        with torch.no_grad():
            Z = Quantizer3(latent_vector + eta, "Hard")
            eta = eta + latent_vector - Z
    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer3(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_ADMM_3_track.append(target_loss.detach().cpu().numpy())
    ### 2bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        Z = Quantizer2(latent_vector, "Hard")
        eta = torch.zeros(latent_vector.shape).cuda()
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    for itera in range(Com_iter):
        generated_img = model.netDecoder.forward(latent_vector)
        optmize_Com.zero_grad()
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                   latent_vector.shape[0]
        Com_loss.backward()
        optmize_Com.step()
        with torch.no_grad():
            Z = Quantizer2(latent_vector + eta, "Hard")
            eta = eta + latent_vector - Z
    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer2(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_ADMM_2_track.append(target_loss.detach().cpu().numpy())
    ### 6bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        mask = torch.ones_like(latent_vector).to(latent_vector.device)
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    N = Compressed_p.shape[0]
    for step_time in range(Sub_step):
        for itera in range(step_iter):
            generated_img = model.netDecoder.forward(latent_vector)
            optmize_Com.zero_grad()
            z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
            z_g = z_g[:, 0, :, :]
            z_g = z_g.transpose(1, 2)
            vgg_loss = VGG_Loss(z_g, z_img)
            mse_loss = Critiretion(generated_img, input_label)
            Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                       latent_vector.shape[0]
            Com_loss.backward()
            with torch.no_grad():
                latent_vector.grad = latent_vector.grad * mask
            optmize_Com.step()
        with torch.no_grad():
            quantized_vecotr = Quantizer6(latent_vector, "Hard")
            arg_index = torch.argsort(torch.abs(quantized_vecotr - latent_vector).view(N, -1), dim = 1)
            arg_index = arg_index <= K * step_time
            arg_index = arg_index.view(latent_vector.shape)
            latent_vector[arg_index] = quantized_vecotr[arg_index]
            mask[arg_index] = 0.0
    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer6(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_IHT_6_track.append(target_loss.detach().cpu().numpy())
    ### 5bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        mask = torch.ones_like(latent_vector).to(latent_vector.device)
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    N = Compressed_p.shape[0]
    for step_time in range(Sub_step):
        for itera in range(step_iter):
            generated_img = model.netDecoder.forward(latent_vector)
            optmize_Com.zero_grad()
            z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
            z_g = z_g[:, 0, :, :]
            z_g = z_g.transpose(1, 2)
            vgg_loss = VGG_Loss(z_g, z_img)
            mse_loss = Critiretion(generated_img, input_label)
            Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                       latent_vector.shape[0]
            Com_loss.backward()
            with torch.no_grad():
                latent_vector.grad = latent_vector.grad * mask
            optmize_Com.step()
        with torch.no_grad():
            quantized_vecotr = Quantizer5(latent_vector, "Hard")
            arg_index = torch.argsort(torch.abs(quantized_vecotr - latent_vector).view(N, -1), dim = 1)
            arg_index = arg_index <= K * step_time
            arg_index = arg_index.view(latent_vector.shape)
            latent_vector[arg_index] = quantized_vecotr[arg_index]
            mask[arg_index] = 0.0
    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer5(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_IHT_5_track.append(target_loss.detach().cpu().numpy())
    ### 4bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        mask = torch.ones_like(latent_vector).to(latent_vector.device)
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    N = Compressed_p.shape[0]
    for step_time in range(Sub_step):
        for itera in range(step_iter):
            generated_img = model.netDecoder.forward(latent_vector)
            optmize_Com.zero_grad()
            z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
            z_g = z_g[:, 0, :, :]
            z_g = z_g.transpose(1, 2)
            vgg_loss = VGG_Loss(z_g, z_img)
            mse_loss = Critiretion(generated_img, input_label)
            Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                       latent_vector.shape[0]
            Com_loss.backward()
            with torch.no_grad():
                latent_vector.grad = latent_vector.grad * mask
            optmize_Com.step()
        with torch.no_grad():
            quantized_vecotr = Quantizer4(latent_vector, "Hard")
            arg_index = torch.argsort(torch.abs(quantized_vecotr - latent_vector).view(N, -1), dim = 1)
            arg_index = arg_index <= K * step_time
            arg_index = arg_index.view(latent_vector.shape)
            latent_vector[arg_index] = quantized_vecotr[arg_index]
            mask[arg_index] = 0.0
    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer4(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_IHT_4_track.append(target_loss.detach().cpu().numpy())
    ### 3bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        mask = torch.ones_like(latent_vector).to(latent_vector.device)
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    N = Compressed_p.shape[0]
    for step_time in range(Sub_step):
        for itera in range(step_iter):
            generated_img = model.netDecoder.forward(latent_vector)
            optmize_Com.zero_grad()
            z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
            z_g = z_g[:, 0, :, :]
            z_g = z_g.transpose(1, 2)
            vgg_loss = VGG_Loss(z_g, z_img)
            mse_loss = Critiretion(generated_img, input_label)
            Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                       latent_vector.shape[0]
            Com_loss.backward()
            with torch.no_grad():
                latent_vector.grad = latent_vector.grad * mask
            optmize_Com.step()
        with torch.no_grad():
            quantized_vecotr = Quantizer3(latent_vector, "Hard")
            arg_index = torch.argsort(torch.abs(quantized_vecotr - latent_vector).view(N, -1), dim = 1)
            arg_index = arg_index <= K * step_time
            arg_index = arg_index.view(latent_vector.shape)
            latent_vector[arg_index] = quantized_vecotr[arg_index]
            mask[arg_index] = 0.0
    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer3(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_IHT_3_track.append(target_loss.detach().cpu().numpy())
    ### 2bit
    with  torch.no_grad():
        Compressed_p = model.netE.forward(input_label)
        vector_shape = Compressed_p.shape
        latent_vector = Variable(torch.FloatTensor(vector_shape).fill_(0.5).cuda(), requires_grad=True)
        latent_vector.data = Compressed_p.clone()
        mask = torch.ones_like(latent_vector).to(latent_vector.device)
    optmize_Com = torch.optim.Adam([latent_vector], lr=lr)
    N = Compressed_p.shape[0]
    for step_time in range(Sub_step):
        for itera in range(step_iter):
            generated_img = model.netDecoder.forward(latent_vector)
            optmize_Com.zero_grad()
            z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
            z_g = z_g[:, 0, :, :]
            z_g = z_g.transpose(1, 2)
            vgg_loss = VGG_Loss(z_g, z_img)
            mse_loss = Critiretion(generated_img, input_label)
            Com_loss = vgg_loss + alpha * mse_loss + mu / 2.0 * torch.norm(latent_vector - Z + eta, 2) ** 2 / \
                       latent_vector.shape[0]
            Com_loss.backward()
            with torch.no_grad():
                latent_vector.grad = latent_vector.grad * mask
            optmize_Com.step()
        with torch.no_grad():
            quantized_vecotr = Quantizer2(latent_vector, "Hard")
            arg_index = torch.argsort(torch.abs(quantized_vecotr - latent_vector).view(N, -1), dim = 1)
            arg_index = arg_index <= K * step_time
            arg_index = arg_index.view(latent_vector.shape)
            latent_vector[arg_index] = quantized_vecotr[arg_index]
            mask[arg_index] = 0.0
    with torch.no_grad():
        generated_img = model.netDecoder(Quantizer2(latent_vector, "Hard"))
        z_g = torch.einsum("mj,idjk->idmk", [Transform_tensor, generated_img])
        z_g = z_g[:, 0, :, :]
        z_g = z_g.transpose(1, 2)
        vgg_loss = VGG_Loss(z_g, z_img)
        mse_loss = Critiretion(generated_img, input_label)
        target_loss = vgg_loss + alpha * mse_loss
        PSNR_IHT_2_track.append(target_loss.detach().cpu().numpy())
PSNR_BP_track = np.array(PSNR_BP_track)
PSNR_ADMM_5_track = np.array(PSNR_ADMM_5_track)
PSNR_ADMM_6_track = np.array(PSNR_ADMM_6_track)
PSNR_ADMM_4_track = np.array(PSNR_ADMM_4_track)
PSNR_ADMM_3_track = np.array(PSNR_ADMM_3_track)
PSNR_ADMM_2_track = np.array(PSNR_ADMM_2_track)
PSNR_IHT_6_track = np.array(PSNR_IHT_6_track)
PSNR_IHT_5_track = np.array(PSNR_IHT_5_track)
PSNR_IHT_4_track = np.array(PSNR_IHT_4_track)
PSNR_IHT_3_track = np.array(PSNR_IHT_3_track)
PSNR_IHT_2_track = np.array(PSNR_IHT_2_track)

PSNR_5_track = np.array(PSNR_5_track)
PSNR_6_track = np.array(PSNR_6_track)
PSNR_4_track = np.array(PSNR_4_track)
PSNR_3_track = np.array(PSNR_3_track)
PSNR_2_track = np.array(PSNR_2_track)
np.save('./Test/PSNR_BP.npy',PSNR_BP_track)
#np.save('./PSNR_ADMM_16.npy',PSNR_ADMM_16_track)
np.save('./Test/PSNR_ADMM_5.npy',PSNR_ADMM_5_track)
np.save('./Test/PSNR_ADMM_6.npy',PSNR_ADMM_6_track)
np.save('./Test/PSNR_ADMM_4.npy',PSNR_ADMM_4_track)
np.save('./Test/PSNR_ADMM_3.npy',PSNR_ADMM_3_track)
np.save('./Test/PSNR_ADMM_2.npy',PSNR_ADMM_2_track)
#np.save('./PSNR_16.npy',PSNR_16_track)
np.save('./Test/PSNR_5.npy',PSNR_5_track)
np.save('./Test/PSNR_6.npy',PSNR_6_track)
np.save('./Test/PSNR_4.npy',PSNR_4_track)
np.save('./Test/PSNR_3.npy',PSNR_3_track)
np.save('./Test/PSNR_2.npy',PSNR_2_track)
np.save('./Test/PSNR_IHT_5.npy',PSNR_IHT_5_track)
np.save('./Test/PSNR_IHT_6.npy',PSNR_IHT_6_track)
np.save('./Test/PSNR_IHT_4.npy',PSNR_IHT_4_track)
np.save('./Test/PSNR_IHT_3.npy',PSNR_IHT_3_track)
np.save('./Test/PSNR_IHT_2.npy',PSNR_IHT_2_track)
print("PSNRBP",PSNR_BP_track.mean())
print(PSNR_2_track.mean())
print(PSNR_3_track.mean())
print(PSNR_4_track.mean())
print(PSNR_5_track.mean())
print(PSNR_6_track.mean())
print(PSNR_ADMM_2_track.mean())
print(PSNR_ADMM_3_track.mean())
print(PSNR_ADMM_4_track.mean())
print(PSNR_ADMM_5_track.mean())
print(PSNR_ADMM_6_track.mean())


