import sys
import os
import os.path as pth

import tensorflow as tf
import numpy as np
import json
import yaml
import argparse

import net
import layers as L
import dataset
import tools


# =====================================
# Training configuration default params
# =====================================
config = {}

# =========================
# customize your model here
# =========================
def buildModel(input_data_tensor, input_label_tensor):
    num_classes = config['num_classes']
    weight_decay = config['weight_decay']
    # network = net.CNN(input_data_tensor, n_classes=num_classes)
    network = net.CNN(input_data_tensor, n_classes=num_classes)
    # network = net.FFTNN(input_data_tensor, n_classes=num_classes)
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


def train(train_data_generator, vld_data=None):
    data_dims = config['data_dims']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_samples_per_epoch = config['num_samples_per_epoch']
    learning_rate = config['learning_rate']

    pretrained_weights = config.get('pretrained_weights', None)

    checkpoint_iter = config['checkpoint_iter']
    vld_iter = config['vld_iter']

    checkpoint_dir = config['checkpoint_dir']
    experiment_dir = config['experiment_dir']
    train_log_fpath = pth.join(experiment_dir, 'retrain2.log')
    log = tools.StatLogger(train_log_fpath)

    steps_per_epoch = num_samples_per_epoch // batch_size
    num_steps = steps_per_epoch * num_epochs

    # Change default GPU and CUDA_VISIBLE_DEVICES
    # os.environ["CUDA_VISIBLE_DEVICES"] ="1" 

    # ========================
    # construct training graph
    # ========================
    G = tf.Graph()
    with G.as_default():
        input_data_tensor = tf.placeholder(tf.float32, [None] + data_dims)
        # input_data_tensor = tf.placeholder(tf.float32, [batch_size] + data_dims)
        input_label_tensor = tf.placeholder(tf.int32, [None])
        # input_label_tensor = tf.placeholder(tf.int32, [batch_size])
        network, model = buildModel(input_data_tensor, input_label_tensor)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        grads = optimizer.compute_gradients(model['loss'])
        grad_step = optimizer.apply_gradients(grads)
        init = tf.global_variables_initializer()
        trainable_variables = tf.trainable_variables()
        saver = tf.train.Saver()

    # ===================================
    # initialize and run training session
    # ===================================
    sess = tf.Session(graph=G, config=tf.ConfigProto(allow_soft_placement=True))
    # sess = tf.Session(graph=G, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    sess.run(init)

    # # Print an analysis of the memory usage and the timing information broken down by operations.
    # run_metadata = tf.RunMetadata()
    # with sess.as_default():
    #     data_batch, label_batch = next(train_data_generator)
    #     inputs = {input_data_tensor: data_batch, input_label_tensor: label_batch}
    #     _ = sess.run([grad_step] + [model[k] for k in sorted(model.keys())], feed_dict=inputs,
    #         options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
    #         run_metadata=run_metadata)
    
    # tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(),
    #     run_meta=run_metadata,
    #     tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

    with sess.as_default():
        # print(trainable_variables)
        if pretrained_weights:
            print('-- loading weights from %s' % pretrained_weights)
            saver.restore(sess, tf.train.latest_checkpoint(pretrained_weights))

        # Start training loop
        for step in range(num_steps):
            data_batch, label_batch = next(train_data_generator)
            # print(label_batch)
            # with tf.Session() as sess:
            #     data_tensor = tf.convert_to_tensor(data_batch[0][:][:][:])
            #     data_dct = tf.signal.dct(tf.reshape(data_tensor, [1, 32*32, 3]))
            #     print(data_dct.eval())
            # with tf.Session() as sess:
            #     data_tensor = tf.convert_to_tensor(data_batch)
            #     data_fft = tf.signal.fft2d(tf.transpose(tf.cast(data_tensor, tf.complex64), [0,3,1,2]))
            #     data_fft = tf.transpose(data_fft, [0,2,3,1])
            #     print(list(data_batch.shape))
            #     print(data_tensor[0,:,:,:].get_shape().as_list())
            #     print(data_fft.get_shape().as_list())
            #     print(data_fft[0,:,:,0].eval())
            #     print('***************')
            #     print(tf.signal.fft2d(tf.cast(data_tensor[0,:,:,0], tf.complex64)).eval())
            # with tf.Session() as sess:
            #     data_tensor = tf.constant([[1,2,3,4,4],[5,6,5,7,8],[9,10,3,2,1],[2,2,7,4,5],[4,3,6,8,1]], dtype=tf.complex64)
            #     print(tf.signal.fft2d(data_tensor).eval())
            # with tf.Session() as sess:
            #     data_tensor = tf.constant([[[1,2,3],[2,3,4]],[[3,4,5],[4,5,6]]], dtype=tf.float32)
            #     weights_tensor = tf.constant([[1,2,3],[2,3,4]], dtype=tf.float32)
            #     print((data_tensor*weights_tensor).eval())
            #     print(tf.reverse(data_tensor[0,:,:],axis=[0,1]).eval())

            inputs = {input_data_tensor: data_batch, input_label_tensor: label_batch}
            # print("#######################")
            # print(grads)
            results = sess.run([grad_step] + [model[k] for k in sorted(model.keys())], feed_dict=inputs)
            results = dict(zip(sorted(model.keys()), results[1:]))

            # print(results['probs'])
 
            print('TRAIN step:%-5d error_top1: %.4f error_top5: %.4f loss:%s' % (step,
                                                                                 results['error_top1'],
                                                                                 results['error_top5'],
                                                                                 results['loss']))
            log.report(step=step,
                       split='TRAIN',
                       error_top5=float(results['error_top5']),
                       error_top1=float(results['error_top1']),
                       loss=float(results['loss']))

            # report evaluation metrics every vld_iter training steps
            if (step % vld_iter == 0):
                print('-- running evaluation on validation split')
                X_vld = vld_data[0]
                Y_vld = vld_data[1]
                inputs = [input_data_tensor, input_label_tensor]
                args = [X_vld, Y_vld] 
                results = tools.iterativeReduce([model[k] for k in sorted(model.keys())], 
                                            inputs, args, batch_size=200, fn=lambda x: np.mean(x, axis=0))
                results = dict(zip(sorted(model.keys()), results))

                print('VALID step:%-5d error_top1: %.4f error_top5: %.4f loss:%s' % (step,
                                                                                     results['error_top1'],
                                                                                     results['error_top5'],
                                                                                     results['loss']))
                log.report(step=step,
                           split='VALID',
                           error_top5=float(results['error_top5']),
                           error_top1=float(results['error_top1']),
                           loss=float(results['loss']))

            if (step % checkpoint_iter == 0) or (step + 1 == num_steps):
                print('-- saving check point')
                # tools.saveWeights(G, pth.join(checkpoint_dir, 'weights.%s' % step))
                saver.save(sess, pth.join(checkpoint_dir,'model'), global_step=step)

def main():
    batch_size = config['batch_size']
    checkpoint_dir = config['checkpoint_dir']
    experiment_dir = config['experiment_dir']

    # setup experiment and checkpoint directories
    if not pth.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not pth.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_data_generator, vld_data = dataset.getCifar10(batch_size)
    train(train_data_generator, vld_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML formatted config file')
    args = parser.parse_args()
    with open(args.config_file) as fp:
        config.update(yaml.load(fp))

    print('Experiment config')
    print('------------------')
    print(json.dumps(config, indent=4))
    print('------------------')
    main()
