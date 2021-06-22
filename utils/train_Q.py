### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_Q_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
import copy
from torch.utils.tensorboard import  SummaryWriter
from sklearn.cluster import KMeans
import numpy as np
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
intial_flag = True
opt.model = 'pix2pixHDQ'
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    Temp = (opt.Q_final - opt.Q_init_Temp)/(opt.Q_hard_epoch-opt.Q_train_epoch) *(start_epoch-opt.Q_train_epoch)+opt.Q_init_Temp
else:
    start_epoch, epoch_iter = 1, 0
if opt.debug:
    opt.batchSize=2
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 100
    opt.niter_decay = 50
    opt.max_dataset_size = 10
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
summary_path = os.path.join(opt.checkpoints_dir,opt.name,'logs/')
writter = SummaryWriter(log_dir=summary_path)
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
random_index = 20

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    if epoch < opt.Q_train_epoch:
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0

            ############## Forward Pass ######################
            losses, generated,vector = model(Variable(data['label']),
                                      Variable(data['image']), infer=save_fake,if_vector=True,Q_type = "None")

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))
            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['Feature'] + loss_dict['MSE_Loss']
            writter.add_scalar('train/loss_D',loss_D.item(),global_step=total_steps)
            writter.add_scalar('train/loss_G',loss_G.item(), global_step=total_steps)
            writter.add_scalar('train/MSE',loss_dict['MSE_Loss'].item(),global_step=total_steps)
            if i == random_index:
                writter.add_histogram('train/latent_vector',vector,global_step=epoch)
            ############### Backward Pass ####################
            # update generator weights
            model.module.optimizer_G.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == 0:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], 0)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    elif epoch >= opt.Q_train_epoch and epoch < opt.Q_hard_epoch:
        if epoch == opt.Q_train_epoch:
            model.module.save('floating_final')
        if intial_flag:
            intial_flag = False
            for i, data in enumerate(dataset, start=0):
                # whether to collect output images
                save_fake = False
                ############## Forward Pass ######################
                input_label, real_image = model.module.encode_input(Variable(data['label']),
                                                                    Variable(data['image']), infer=save_fake)
                model.module.netE.eval()
                with torch.no_grad():
                    gen_vector = model.module.netE.forward(input_label)
                    if opt.quantize_type == 'vector':
                        gen_vector = gen_vector.view(-1,4)
                    if i ==0:
                            vector_dis = gen_vector.detach().cpu().numpy()
                    else:
                        vector_tem = gen_vector.detach().cpu().numpy()
                        vector_dis = np.concatenate((vector_dis,vector_tem),axis=0)
                if i >= opt.batch_num_ini:
                    break
            if opt.quantize_type == 'scalar':
                vector_dis = vector_dis.reshape(-1)
                kmeans = KMeans(n_clusters=opt.n_cluster).fit(vector_dis.reshape(-1,1))
                center = kmeans.cluster_centers_.flatten()
            elif opt.quantize_type == 'vector':
                vector_dis = vector_dis.reshape(-1,4)
                kmeans = KMeans(n_clusters=opt.n_cluster).fit(vector_dis)
                center = kmeans.cluster_centers_
            center = torch.Tensor(center).cuda()
            model.module.update_center(center)
        model.module.netE.train()
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0

            ############## Forward Pass ######################
            losses, generated, Encode_vector = model(Variable(data['label']),
                                                     Variable(data['image']), infer=save_fake,if_vector= True,Q_type='Soft')
            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G= loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['Feature'] +loss_dict['MSE_Loss']
            writter.add_scalar('train/loss_D',loss_D.item(),global_step=total_steps)
            writter.add_scalar('train/loss_G',loss_G.item(),global_step=total_steps)
            writter.add_scalar('train/MSE', loss_dict['MSE_Loss'].item(),global_step=total_steps)
            if i == random_index:
                writter.add_histogram('train/latent_vector',Encode_vector,global_step=epoch)
            ##loss_G = loss_ADMM
            # update generator weights
            model.module.optimizer_G.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            if total_steps % opt.print_freq == 0:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], 0)),
                                       ('synthesized_image_Q', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ## save latest model
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    else:
        if epoch == opt.Q_hard_epoch:
            model.module.save('Q_soft')
            model.module.netE.train()
        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0

            losses, generated, Encode_vector = model(Variable(data['label']),
                                                     Variable(data['image']), infer=save_fake, if_vector=True,
                                                     Q_type='Hard')
            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['Feature'] + loss_dict['MSE_Loss']

            writter.add_scalar('train/loss_D',loss_D.item(),global_step=total_steps)
            writter.add_scalar('train/loss_G',loss_G.item(), global_step=total_steps)
            writter.add_scalar('train/MSE',loss_dict['MSE_Loss'].item(),global_step=total_steps)
            ##loss_G = loss_ADMM
            # update generator weights
            model.module.optimizer_G.zero_grad()
            loss_G.backward()
            model.module.optimizer_G.step()

            # update discriminator weights
            model.module.optimizer_D.zero_grad()
            loss_D.backward()
            model.module.optimizer_D.step()

            if total_steps % opt.print_freq == 0:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], 0)),
                                       ('synthesized_image_Q', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.module.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
    if epoch >= opt.Q_train_epoch and epoch < opt.Q_hard_epoch:
        Temp = (opt.Q_final - opt.Q_init_Temp) / (opt.Q_hard_epoch - opt.Q_train_epoch) * (
                    epoch - opt.Q_train_epoch) + opt.Q_init_Temp
        model.module.update_Temp(Temp)
        print('Temp is %f'%(Temp))
writter.close()
