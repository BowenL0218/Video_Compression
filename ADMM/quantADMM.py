import sys
import os
import os.path as pth

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs

import json
import yaml
import argparse

import layers as L
import net
import dataset
import tools

# =====================================
# Training configuration default params
# =====================================
config = {}
        
'''
Weights quantization with ADMM
'''

# ADMM solver for 6-layer CNN
class ADMMSolver():
  def __init__(self, model, loss, trainable_variables, masks):    
    self.A11 = tf.placeholder(tf.float32, shape = [3, 3, 3, 16])
    self.B11 = tf.placeholder(tf.float32, shape = [3, 3, 3, 16])
    self.A12 = tf.placeholder(tf.float32, shape = [3, 3, 16, 16])
    self.B12 = tf.placeholder(tf.float32, shape = [3, 3, 16, 16])
    self.A21 = tf.placeholder(tf.float32, shape = [3, 3, 16, 32])
    self.B21 = tf.placeholder(tf.float32, shape = [3, 3, 16, 32])
    self.A22 = tf.placeholder(tf.float32, shape = [3, 3, 32, 32])
    self.B22 = tf.placeholder(tf.float32, shape = [3, 3, 32, 32])
    self.A31 = tf.placeholder(tf.float32, shape = [3, 3, 32, 64])
    self.B31 = tf.placeholder(tf.float32, shape = [3, 3, 32, 64])
    self.A32 = tf.placeholder(tf.float32, shape = [3, 3, 64, 64])
    self.B32 = tf.placeholder(tf.float32, shape = [3, 3, 64, 64])
    self.A4 = tf.placeholder(tf.float32, shape = [4 * 4 * 64, 10])
    self.B4 = tf.placeholder(tf.float32, shape = [4 * 4 * 64, 10])

    conv1_1 = model.conv1_1
    conv1_2 = model.conv1_2
    conv2_1 = model.conv2_1
    conv2_2 = model.conv2_2
    conv3_1 = model.conv3_1
    conv3_2 = model.conv3_2
    fc4 = model.fc4
    
    # loss1 = loss+0.00005*(tf.nn.l2_loss(conv1_1)+tf.nn.l2_loss(conv1_2)+tf.nn.l2_loss(conv2_1)+tf.nn.l2_loss(conv2_2)\
    #                         +tf.nn.l2_loss(conv3_1)+tf.nn.l2_loss(conv3_2)+tf.nn.l2_loss(fc4))
    loss_combined = loss+0.00005*(tf.nn.l2_loss(conv1_1)+tf.nn.l2_loss(conv1_2)+tf.nn.l2_loss(conv2_1)+tf.nn.l2_loss(conv2_2)\
                            +tf.nn.l2_loss(conv3_1)+tf.nn.l2_loss(conv3_2)+tf.nn.l2_loss(fc4))\
                        +0.0001*(tf.nn.l2_loss(conv1_1-self.A11+self.B11)+ tf.nn.l2_loss(conv1_2-self.A12+self.B12)\
                            +tf.nn.l2_loss(conv2_1-self.A21+self.B21)+tf.nn.l2_loss(conv2_2-self.A22+self.B22)\
                            +tf.nn.l2_loss(conv3_1-self.A31+self.B31)+ tf.nn.l2_loss(conv3_2-self.A32+self.B32)\
                            +tf.nn.l2_loss(fc4-self.A4+self.B4))

    self.grads_step = maskedGradient(trainable_variables, masks, loss_combined, lr = 0.001)

# Uniform quantizaiton
def projection(weights, quant_bits = 8):
    weights_tmp = weights.flatten()
    abs_max = max(abs(weights_tmp))
    bound = 2**(quant_bits-1)

    # if abs_max >= bound:
    #     interval = np.ceil(abs_max/bound)
    # else:
    #     interval = 1/np.ceil(bound/abs_max)
    interval = 2**(np.ceil(np.log2(abs_max)) - (quant_bits-1))
    print('maximum weights: %.4f, interval: %.4f' % (abs_max, interval))

    weights_quant = np.round(weights_tmp/interval)
    weights_quant = weights_quant * interval
    weights_quant = np.reshape(weights_quant, weights.shape)
    return weights_quant

def retriveMasks(sess, trainable_variables):
    # set variables reusable
    vs.get_variable_scope().reuse_variables()

    # initialize mask list
    masks_list = []

    for i, var in enumerate(trainable_variables):
        if 'weights' in var.name:  
            weights_tmp = sess.run(var)

            # retrive mask for weights
            mask = (np.abs(weights_tmp) > 0).astype(np.float32)      
            masks_list.append(mask)
    return masks_list

def maskedGradient(trainable_variables, masks, loss, lr = 0.001):
    # set variables reusable
    vs.get_variable_scope().reuse_variables()

    # initialize gradient list
    gradients_list = tf.gradients(xs=trainable_variables, ys=loss)       
    gradients_op_list = []
    # lr * gradients
    gradients = [lr*g for g in gradients_list]

    idx = 0
    for i, var in enumerate(trainable_variables):
        if 'weights' in var.name and (var.name.startswith('conv') or var.name.startswith('fftmult') or var.name.startswith('fc')):  
            # Get masked gradient list
            gradients_mask = gradients[i] * masks[idx]
            new_var = tf.subtract(var, gradients_mask)
            gradients_op = var.assign(new_var)
            gradients_op_list.append(gradients_op)
            idx = idx + 1

        else:
            new_var = tf.subtract(var, gradients[i])
            gradients_op = var.assign(new_var)
            gradients_op_list.append(gradients_op)

    return gradients_op_list

# ================================
# Build the same model as training
# ================================
def buildModel(input_data_tensor, input_label_tensor):
    num_classes = config['num_classes']
    weight_decay = config['weight_decay']
    network = net.CNN(input_data_tensor, n_classes=num_classes)
    logits = network.out
    probs = tf.nn.softmax(logits)
    # loss = L.loss(logits, tf.one_hot(input_label_tensor, num_classes))
    loss_classify = L.loss(logits, tf.one_hot(input_label_tensor, num_classes))
    loss_weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('variables')]))
    loss = loss_classify + weight_decay*loss_weight_decay
    error_top5 = L.topKError(probs, input_label_tensor, K=5)
    error_top1 = L.topKError(probs, input_label_tensor, K=1)

    # you must return a dictionary with loss as a key, other variables
    return network, dict(probs=probs,
                         loss=loss,
                         logits=logits,
                         error_top5=error_top5,
                         error_top1=error_top1)

#retrain
def retrain(train_data_generator, vld_data=None):
    data_dims = config['data_dims']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_samples_per_epoch = config['num_samples_per_epoch']
    retrain_lr = config['retraining_learning_rate']

    checkpoint_iter = config['checkpoint_iter']
    vld_iter = config['vld_iter']

    checkpoint_dir = config['load_checkpoint_dir']
    retrain_ckpt_dir = config['retrain_ckpt_dir']
    experiment_dir = config['experiment_dir']
    retrain_log_fpath = pth.join(experiment_dir, 'quant.log')
    log = tools.StatLogger(retrain_log_fpath)

    steps_per_epoch = num_samples_per_epoch // batch_size
    num_steps = steps_per_epoch * num_epochs

    # Change default GPU and CUDA_VISIBLE_DEVICES
    # os.environ["CUDA_VISIBLE_DEVICES"] ="1" 

    input_data_tensor = tf.placeholder(tf.float32, [None] + data_dims)
    input_label_tensor = tf.placeholder(tf.int32, [None])
    network, model = buildModel(input_data_tensor, input_label_tensor)
    optimizer = tf.train.AdamOptimizer(retrain_lr)
    trainable_variables = tf.trainable_variables()
    saver = tf.train.Saver()

    # quantization bits, layer1_conv, layer1_fft_real, layer1_fft_imag, layer2, ..., fc1, ...
    quant_bits = [5, 5, 5, 5, 5, 5, 5]

    print('Start retraining...')

    with tf.Session() as sess:
        print('-- loading ckeckpoint from %s' % checkpoint_dir)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        
        # retrive pruning masks & apply masked gradient
        masks = retriveMasks(sess, trainable_variables)
        grad_step = maskedGradient(trainable_variables, masks, model['loss'], retrain_lr)

        ele_count = 0
        one_count = 0
        for i in range(len(masks)):
            ele_count = ele_count + masks[i].size
            one_count = one_count + np.sum(masks[i])
        print('compress rate: %.4f total weights: %d left weights: % d ' % (one_count*1.0 / ele_count, ele_count, one_count))
        log.report(step=0,
                   compress_rate=one_count*1.0 / ele_count,
                   total_weights=ele_count,
                   left_weights=one_count)

        solver = ADMMSolver(network, model['loss'], trainable_variables, masks)
        grads_step = solver.grads_step

        conv1_1 = network.conv1_1
        conv1_2 = network.conv1_2
        conv2_1 = network.conv2_1
        conv2_2 = network.conv2_2
        conv3_1 = network.conv3_1
        conv3_2 = network.conv3_2
        fc4 = network.fc4
  
        A11 = solver.A11
        B11 = solver.B11
        A12 = solver.A12
        B12 = solver.B12
        A21 = solver.A21
        B21 = solver.B21
        A22 = solver.A22
        B22 = solver.B22
        A31 = solver.A31
        B31 = solver.B31
        A32 = solver.A32
        B32 = solver.B32
        A4 = solver.A4
        B4 = solver.B4
   
        Z11 = sess.run(conv1_1)
        Z11 = projection(Z11, quant_bits[0])

        U11 = np.zeros_like(Z11)

        Z12 = sess.run(conv1_2)
        Z12 = projection(Z12, quant_bits[1])

        U12 = np.zeros_like(Z12)

        Z21 = sess.run(conv2_1)
        Z21 = projection(Z21, quant_bits[2])

        U21 = np.zeros_like(Z21)

        Z22 = sess.run(conv2_2)
        Z22 = projection(Z22, quant_bits[3])

        U22 = np.zeros_like(Z22)

        Z31 = sess.run(conv3_1)
        Z31 = projection(Z31, quant_bits[4])

        U31 = np.zeros_like(Z31)

        Z32 = sess.run(conv3_2)
        Z32 = projection(Z32, quant_bits[5])

        U32 = np.zeros_like(Z32)

        Z4 = sess.run(fc4)
        Z4 = projection(Z4, quant_bits[6])

        U4 = np.zeros_like(Z4)

        # ADMM iterations & retrain steps per iteration
        for itr in range(30):
            for step in range(5000):
                data_batch, label_batch = next(train_data_generator)
                inputs = {input_data_tensor: data_batch, input_label_tensor: label_batch, A11:Z11, B11:U11, A12:Z12, B12:U12, A21:Z21, B21:U21, 
                        A22:Z22, B22:U22, A31:Z31, B31:U31, A32:Z32, B32:U32, A4:Z4, B4:U4}

                results = sess.run([grads_step] + [model[k] for k in sorted(model.keys())], feed_dict=inputs)
                results = dict(zip(sorted(model.keys()), results[1:]))
            
                print('TRAIN step:%-5d error_top1: %.4f error_top5: %.4f loss:%s' % (step,
                                                                                     results['error_top1'],
                                                                                     results['error_top5'],
                                                                                     results['loss']))
                log.report(step=step,
                           split='TRAIN',
                           error_top5=float(results['error_top5']),
                           error_top1=float(results['error_top1']),
                           loss=float(results['loss']))
            
            Z11 = sess.run(conv1_1) + U11
            Z11 = projection(Z11, quant_bits[0])

            U11 = U11 + sess.run(conv1_1) - Z11

            Z12 = sess.run(conv1_2) + U12
            Z12 = projection(Z12, quant_bits[1])

            U12 = U12 + sess.run(conv1_2) - Z12

            Z21 = sess.run(conv2_1) + U21
            Z21 = projection(Z21, quant_bits[2])

            U21 = U21 + sess.run(conv2_1) - Z21

            Z22 = sess.run(conv2_2) + U22
            Z22 = projection(Z22, quant_bits[3])

            U22 = U22 + sess.run(conv2_2) - Z22

            Z31 = sess.run(conv3_1) + U31
            Z31 = projection(Z31, quant_bits[4])

            U31 = U31 + sess.run(conv3_1) - Z31

            Z32 = sess.run(conv3_2) + U32
            Z32 = projection(Z32, quant_bits[5])

            U32 = U32 + sess.run(conv3_2) - Z32

            Z4 = sess.run(fc4) + U4
            Z4 = projection(Z4, quant_bits[6])

            U4 = U4 + sess.run(fc4) - Z4

            print('-- running evaluation on validation split')
            X_vld = vld_data[0]
            Y_vld = vld_data[1]
            inputs = [input_data_tensor, input_label_tensor]
            args = [X_vld, Y_vld]

            results = tools.iterativeReduce([model[k] for k in sorted(model.keys())], 
                                            inputs, args, batch_size=100, fn=lambda x: np.mean(x, axis=0))
            results = dict(zip(sorted(model.keys()), results))

            print('VALID step:%-5d error_top1: %.4f error_top5: %.4f loss:%s' % (itr,
                                                                                 results['error_top1'],
                                                                                 results['error_top5'],
                                                                                 results['loss']))
            log.report(step=itr,
                       split='VALID',
                       error_top5=float(results['error_top5']),
                       error_top1=float(results['error_top1']),
                       loss=float(results['loss']))          
                
            # Norm difference
            from numpy import linalg as LA
            print(LA.norm(sess.run(conv1_1) - Z11))
            print(LA.norm(sess.run(conv1_2) - Z12))
            print(LA.norm(sess.run(conv2_1) - Z21))
            print(LA.norm(sess.run(conv2_2) - Z22))
            print(LA.norm(sess.run(conv3_1) - Z31))
            print(LA.norm(sess.run(conv3_2) - Z32))
            print(LA.norm(sess.run(fc4) - Z4))

        sess.run(tf.assign(conv1_1, Z11))
        sess.run(tf.assign(conv1_2, Z12))
        sess.run(tf.assign(conv2_1, Z21))
        sess.run(tf.assign(conv2_2, Z22))
        sess.run(tf.assign(conv3_1, Z31))
        sess.run(tf.assign(conv3_2, Z32))
        sess.run(tf.assign(fc4, Z4))

        print('-- running evaluation on validation split')
        X_vld = vld_data[0]
        Y_vld = vld_data[1]
        inputs = [input_data_tensor, input_label_tensor]
        args = [X_vld, Y_vld]

        results = tools.iterativeReduce([model[k] for k in sorted(model.keys())], 
                                         inputs, args, batch_size=100, fn=lambda x: np.mean(x, axis=0))
        results = dict(zip(sorted(model.keys()), results))

        print('VALID step:%-5d error_top1: %.4f error_top5: %.4f loss:%s' % (itr,
                                                                             results['error_top1'],
                                                                             results['error_top5'],
                                                                             results['loss']))
        log.report(step=itr,
                   split='VALID',
                   error_top5=float(results['error_top5']),
                   error_top1=float(results['error_top1']),
                   loss=float(results['loss']))    

        print('-- saving check point')
        # tools.save_weights(G, pth.join(retrain_ckpt_dir, 'weights.%s' % step))
        saver.save(sess, pth.join(retrain_ckpt_dir,'model'), global_step=step)   
                   
    print('Done retraining!')


def main():
    batch_size = config['batch_size']
    retrain_ckpt_dir = config['retrain_ckpt_dir']

    if not pth.exists(retrain_ckpt_dir):
        os.makedirs(retrain_ckpt_dir)

    train_data_generator, vld_data = dataset.getCifar10(batch_size)
    retrain(train_data_generator, vld_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML formatted config file')
    args = parser.parse_args()
    with open(args.config_file) as fp:
        config.update(yaml.load(fp))

    print("Experiment config")
    print("------------------")
    print(json.dumps(config, indent=4))
    print("------------------")
    main()
