'''
Build CNN model.
'''

import numpy as np
import tensorflow as tf

import layers as L


'''
custermized mix FFT-cnn for CIFAR-10
'''
class MIXNN():
    def __init__(self, input_tensor, n_classes=10, rgb_mean=None):
        # assuming 32x32x3 input_tensor
        # define image mean
        if rgb_mean is None:
            rgb_mean = np.array([128, 128, 128], dtype=np.float32)
        mu = tf.constant(rgb_mean, name='rgb_mean')

        # subtract image mean
        net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

        # define block size
        block_size = 6
        fsize = 8

        # block 1 -- inputs 36x36x3 outputs 18x18x16
        paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
        net = tf.pad(net, paddings, 'CONSTANT')
        net1, self.conv1_1 = L.conv(net, name='conv1_1', kh=3, kw=3, n_out=fsize)
        net2, self.fftmult1_1_real, self.fftmult1_1_imag = L.fftmultblock2d(net, block_size, name='fftmult1_1', n_out=fsize)
        # net = tf.math.add(net1, net2)
        net = tf.concat([net1, net2], axis=-1)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net1, self.conv1_2 = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=fsize)
        net2, self.fftmult1_2_real, self.fftmult1_2_imag = L.fftmultblock2d(net, block_size, name='fftmult1_2', n_out=fsize)
        # net = tf.math.add(net1, net2)
        net = tf.concat([net1, net2], axis=-1)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
        # net = tf.layers.dropout(net, rate=0.2)

        # block 2 -- inputs 18x18x16 outputs 9x9x32
        net1, self.conv2_1 = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=fsize*2)
        net2, self.fftmult2_1_real, self.fftmult2_1_imag = L.fftmultblock2d(net, block_size, name='fftmult2_1', n_out=fsize*2)
        # net = tf.math.add(net1, net2)
        net = tf.concat([net1, net2], axis=-1)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net1, self.conv2_2 = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=fsize*2)
        net2, self.fftmult2_2_real, self.fftmult2_2_imag = L.fftmultblock2d(net, block_size, name='fftmult2_2', n_out=fsize*2)
        # net = tf.math.add(net1, net2)
        net = tf.concat([net1, net2], axis=-1)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
        # net = tf.layers.dropout(net, rate=0.3)

        # block 3 -- inputs 12x12x32 outputs 6x6x64
        paddings = tf.constant([[0,0],[1,2],[1,2],[0,0]])
        net = tf.pad(net, paddings, 'CONSTANT')
        net1, self.conv3_1 = L.conv(net, name='conv3_1', kh=3, kw=3, n_out=fsize*4)
        net2, self.fftmult3_1_real, self.fftmult3_1_imag = L.fftmultblock2d(net, block_size, name='fftmult3_1', n_out=fsize*4)
        # net = tf.math.add(net1, net2)
        net = tf.concat([net1, net2], axis=-1)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net1, self.conv3_2 = L.conv(net, name='conv3_2', kh=3, kw=3, n_out=fsize*4)
        net2, self.fftmult3_2_real, self.fftmult3_2_imag = L.fftmultblock2d(net, block_size, name='fftmult3_2', n_out=fsize*4)
        # net = tf.math.add(net1, net2)
        net = tf.concat([net1, net2], axis=-1)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
        # net = tf.layers.dropout(net, rate=0.4)

        # flatten
        flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
        net = tf.reshape(net, [-1, flattened_shape], name='flatten')

        # fully connected
        self.out, self.fc4 = L.fc(net, name='fc4', n_out=n_classes)


'''
Equivalent FFT-NN
'''
class FFTNN():
    def __init__(self, input_tensor, n_classes=10, rgb_mean=None):
        # assuming 32x32x3 input_tensor
        # define image mean
        if rgb_mean is None:
            rgb_mean = np.array([128, 128, 128], dtype=np.float32)
        mu = tf.constant(rgb_mean, name='rgb_mean')

        # subtract image mean
        net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

        # define block size
        block_size = 6
        fsize = 16

        # block 1 -- inputs 36x36x3 outputs 18x18x16
        paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
        net = tf.pad(net, paddings, 'CONSTANT')
        net, self.fftmult1_1_real, self.fftmult1_1_imag = L.fftmultblock2d(net, block_size, name='fftmult1_1', n_out=fsize)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net, self.fftmult1_2_real, self.fftmult1_2_imag = L.fftmultblock2d(net, block_size, name='fftmult1_2', n_out=fsize)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
        # net = tf.layers.dropout(net, rate=0.2)

        # block 2 -- inputs 18x18x16 outputs 9x9x32
        net, self.fftmult2_1_real, self.fftmult2_1_imag = L.fftmultblock2d(net, block_size, name='fftmult2_1', n_out=fsize*2)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net, self.fftmult2_2_real, self.fftmult2_2_imag = L.fftmultblock2d(net, block_size, name='fftmult2_2', n_out=fsize*2)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
        # net = tf.layers.dropout(net, rate=0.3)

        # block 3 -- inputs 12x12x32 outputs 6x6x64
        paddings = tf.constant([[0,0],[1,2],[1,2],[0,0]])
        net = tf.pad(net, paddings, 'CONSTANT')
        net, self.fftmult3_1_real, self.fftmult3_1_imag = L.fftmultblock2d(net, block_size, name='fftmult3_1', n_out=fsize*4)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net, self.fftmult3_2_real, self.fftmult3_2_imag = L.fftmultblock2d(net, block_size, name='fftmult3_2', n_out=fsize*4)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)

        net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
        # net = tf.layers.dropout(net, rate=0.4)

        # flatten
        flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
        net = tf.reshape(net, [-1, flattened_shape], name='flatten')

        # fully connected
        self.out, self.fc4 = L.fc(net, name='fc4', n_out=n_classes)


'''
Simple cnn for CIFAR-10
'''
class CNN():
    def __init__(self, input_tensor, n_classes=10, rgb_mean=None):
        # assuming 32x32x3 input_tensor
        # define image mean
        if rgb_mean is None:
            rgb_mean = np.array([128, 128, 128], dtype=np.float32)
        mu = tf.constant(rgb_mean, name='rgb_mean')

        # subtract image mean
        net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

        # block 1 -- outputs 16x16x32
        net, self.conv1_1 = L.conv(net, name='conv1_1', kh=3, kw=3, n_out=16)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net, self.conv1_2 = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=16)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
        # net = tf.nn.dropout(net, rate=0.2)

        # block 2 -- outputs 8x8x64
        net, self.conv2_1 = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=32)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net, self.conv2_2 = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=32)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
        # net = tf.nn.dropout(net, rate=0.3)

        # # block 3 -- outputs 4x4x128
        net, self.conv3_1 = L.conv(net, name='conv3_1', kh=3, kw=3, n_out=64)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net, self.conv3_2 = L.conv(net, name='conv3_2', kh=3, kw=3, n_out=64)
        net = tf.nn.relu(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
        # net = tf.nn.dropout(net, rate=0.4)

        # flatten
        flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
        net = tf.reshape(net, [-1, flattened_shape], name='flatten')

        # fully connected
        self.out, self.fc4 = L.fc(net, name='fc4', n_out=n_classes)

# '''
# FFT cnn for CIFAR-10, block version.
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean', dtype=tf.float32)

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # define block size
#     block_size = 4

#     # block 1 -- outputs 16x16x32
#     net = L.fftmultblock2d(net, block_size, name='fftmult1_1', n_out=32)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.fftmultblock2d(net, block_size, name='fftmult1_2', n_out=32)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
#     # net = tf.nn.dropout(net, rate=0.2)

#     # block 2 -- outputs 8x8x64
#     net = L.fftmultblock2d(net, block_size, name='fftmult2_1', n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.fftmultblock2d(net, block_size, name='fftmult2_2', n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.3)

#     # # block 3 -- outputs 4x4x128
#     net = L.fftmultblock2d(net, block_size, name='fftmult3_1', n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.fftmultblock2d(net, block_size, name='fftmult3_2', n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.4)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net


# '''
# FFT cnn for CIFAR-10
# Complex number training is not supported in tensorflow, variable initializer
# cannot take complex data type.
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean', dtype=tf.float32)

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # block 1 -- outputs 16x16x32
#     net = L.fftmult2d(net, name='fftmult1_1', n_out=32)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.fftmult2d(net, name='fftmult1_2', n_out=32)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
#     net = tf.nn.dropout(net, rate=0.2)

#     # block 2 -- outputs 8x8x64
#     net = L.fftmult2d(net, name='fftmult2_1', n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.fftmult2d(net, name='fftmult2_2', n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
#     net = tf.nn.dropout(net, rate=0.3)

#     # # block 3 -- outputs 4x4x128
#     net = L.fftmult2d(net, name='fftmult3_1', n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.fftmult2d(net, name='fftmult3_2', n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
#     net = tf.nn.dropout(net, rate=0.4)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net


# '''
# Winograd cnn for CIFAR-10
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean')

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # block 1 -- outputs 16x16x32
#     net = L.wino(net, name='wino1_1', n_out=32)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.wino(net, name='wino1_2', n_out=32)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
#     # net = tf.nn.dropout(net, rate=0.2)

#     # block 2 -- outputs 8x8x64
#     net = L.wino(net, name='wino2_1', n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.wino(net, name='wino2_2', n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.3)

#     # # block 3 -- outputs 4x4x128
#     net = L.wino(net, name='wino3_1', n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.wino(net, name='wino3_2', n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.4)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net




# '''
# FFT cnn for CIFAR-10, block version. test with 1 layer, 8 filters
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean', dtype=tf.float32)

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # define block size
#     block_size = 4

#     # block 1 -- outputs 16x16x32
#     net = L.fftmultblock2d(net, block_size, name='fftmult1_1', n_out=8)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net


# '''
# Equivalent cnn for fft cnn test with 1 layer, 8 filters
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean')

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # block 1 -- outputs 16x16x32
#     net = L.conv(net, name='conv1_1', kh=3, kw=3, n_out=8)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     # net = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=32)
#     # net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)
#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
#     # # net = tf.nn.dropout(net, rate=0.2)

#     # # block 2 -- outputs 8x8x64
#     # net = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=64)
#     # net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)
#     # net = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=64)
#     # net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)
#     # net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
#     # # net = tf.nn.dropout(net, rate=0.3)

#     # # # block 3 -- outputs 4x4x128
#     # net = L.conv(net, name='conv3_1', kh=3, kw=3, n_out=128)
#     # net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)
#     # net = L.conv(net, name='conv3_2', kh=3, kw=3, n_out=128)
#     # net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)
#     # net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
#     # # net = tf.nn.dropout(net, rate=0.4)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net


# '''
# Winograd cnn for CIFAR-10 test with 1 layer, 4 filters
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean')

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # block 1 -- outputs 16x16x32
#     net = L.wino(net, name='wino1_1', n_out=4)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=32)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
#     # net = tf.nn.dropout(net, rate=0.2)

#     # block 2 -- outputs 8x8x64
#     net = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.3)

#     # # block 3 -- outputs 4x4x128
#     net = L.conv(net, name='conv3_1', kh=3, kw=3, n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.conv(net, name='conv3_2', kh=3, kw=3, n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.4)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net


# '''
# Equivalent cnn for Winograd cnn test with 1 layer, 4 filters
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean')

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # block 1 -- outputs 16x16x32
#     net = L.conv(net, name='conv1_1', kh=3, kw=3, n_out=4)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=32)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
#     # net = tf.nn.dropout(net, rate=0.2)

#     # block 2 -- outputs 8x8x64
#     net = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=64)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.3)

#     # # block 3 -- outputs 4x4x128
#     net = L.conv(net, name='conv3_1', kh=3, kw=3, n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.conv(net, name='conv3_2', kh=3, kw=3, n_out=128)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)
#     net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.4)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net




# '''
# DCT cnn for CIFAR-10
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean')

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # block 1 -- outputs 16x16x32
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net_dct = L.emult(net_dct, name='emult1_1', n_out=32)
#     net_dct = tf.signal.idct(net_dct)
#     net = tf.reshape(net_dct, [-1, net.get_shape()[1].value, net.get_shape()[2].value, 32])
#     net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)

#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net_dct = L.emult(net_dct, name='emult1_2', n_out=32)
#     net_dct = tf.signal.idct(net_dct)
#     net = tf.reshape(net_dct, [-1, net.get_shape()[1].value, net.get_shape()[2].value, 32])
#     net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
#     # net = tf.nn.dropout(net, rate=0.2)

#     # block 2 -- outputs 8x8x64
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net_dct = L.emult(net_dct, name='emult2_1', n_out=64)
#     net_dct = tf.signal.idct(net_dct)
#     net = tf.reshape(net_dct, [-1, net.get_shape()[1].value, net.get_shape()[2].value, 64])
#     net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)

#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net_dct = L.emult(net_dct, name='emult2_2', n_out=64)
#     net_dct = tf.signal.idct(net_dct)
#     net = tf.reshape(net_dct, [-1, net.get_shape()[1].value, net.get_shape()[2].value, 64])
#     net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.3)

#     # # block 3 -- outputs 4x4x128
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net_dct = L.emult(net_dct, name='emult3_1', n_out=128)
#     net_dct = tf.signal.idct(net_dct)
#     net = tf.reshape(net_dct, [-1, net.get_shape()[1].value, net.get_shape()[2].value, 128])
#     net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)

#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net_dct = L.emult(net_dct, name='emult3_2', n_out=128)
#     net_dct = tf.signal.idct(net_dct)
#     net = tf.reshape(net_dct, [-1, net.get_shape()[1].value, net.get_shape()[2].value, 128])
#     net = tf.nn.relu(net)
#     # net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
#     # net = tf.nn.dropout(net, rate=0.4)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net


# '''
# custermized mix DCT-cnn for CIFAR-10
# '''
# def build(input_tensor, n_classes=10, rgb_mean=None):
#     # assuming 32x32x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([128, 128, 128], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean')

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # block 1 -- outputs 16x16x32
#     net1 = L.conv(net, name='conv1_1', kh=3, kw=3, n_out=16)
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net2 = L.emult(net_dct, name='emult1_1', n_out=16)
#     net2 = tf.signal.idct(net2)
#     net2 = tf.reshape(net2, tf.shape(net1))
#     net = tf.math.add(net1, net2)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net1 = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=16)
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net2 = L.emult(net_dct, name='emult1_2', n_out=16)
#     net2 = tf.signal.idct(net2)
#     net2 = tf.reshape(net2, tf.shape(net1))
#     net = tf.math.add(net1, net2)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)
#     net = tf.nn.dropout(net, rate=0.2)

#     # block 2 -- outputs 8x8x64
#     net1 = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=32)
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net2 = L.emult(net_dct, name='emult2_1', n_out=32)
#     net2 = tf.signal.idct(net2)
#     net2 = tf.reshape(net2, tf.shape(net1))
#     net = tf.math.add(net1, net2)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net1 = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=32)
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net2 = L.emult(net_dct, name='emult2_2', n_out=32)
#     net2 = tf.signal.idct(net2)
#     net2 = tf.reshape(net2, tf.shape(net1))
#     net = tf.math.add(net1, net2)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)
#     net = tf.nn.dropout(net, rate=0.3)

#     # # block 3 -- outputs 4x4x128
#     net1 = L.conv(net, name='conv3_1', kh=3, kw=3, n_out=64)
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net2 = L.emult(net_dct, name='emult3_1', n_out=64)
#     net2 = tf.signal.idct(net2)
#     net2 = tf.reshape(net2, tf.shape(net1))
#     net = tf.math.add(net1, net2)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net1 = L.conv(net, name='conv3_2', kh=3, kw=3, n_out=64)
#     net_dct = tf.signal.dct(tf.reshape(net, [-1, net.get_shape()[1].value*net.get_shape()[2].value, net.get_shape()[3].value]))
#     net2 = L.emult(net_dct, name='emult3_2', n_out=64)
#     net2 = tf.signal.idct(net2)
#     net2 = tf.reshape(net2, tf.shape(net1))
#     net = tf.math.add(net1, net2)
#     net = tf.nn.relu(net)
#     net = tf.keras.layers.BatchNormalization()(net)

#     net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)
#     net = tf.nn.dropout(net, rate=0.4)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc4', n_out=n_classes)
#     return net


'''
VGG-16 net for imagenet
'''
# def build(input_tensor, n_classes=1000, rgb_mean=None):
#     # assuming 224x224x3 input_tensor

#     # define image mean
#     if rgb_mean is None:
#         rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)
#     mu = tf.constant(rgb_mean, name='rgb_mean')
#     keep_prob = 0.5

#     # subtract image mean
#     net = tf.math.subtract(input_tensor, mu, name='input_mean_centered')

#     # block 1 -- outputs 112x112x64
#     net = L.conv(net, name='conv1_1', kh=3, kw=3, n_out=64)
#     net = L.conv(net, name='conv1_2', kh=3, kw=3, n_out=64)
#     net = L.pool(net, name='pool1', kh=2, kw=2, dw=2, dh=2)

#     # block 2 -- outputs 56x56x128
#     net = L.conv(net, name='conv2_1', kh=3, kw=3, n_out=128)
#     net = L.conv(net, name='conv2_2', kh=3, kw=3, n_out=128)
#     net = L.pool(net, name='pool2', kh=2, kw=2, dh=2, dw=2)

#     # # block 3 -- outputs 28x28x256
#     net = L.conv(net, name='conv3_1', kh=3, kw=3, n_out=256)
#     net = L.conv(net, name='conv3_2', kh=3, kw=3, n_out=256)
#     net = L.pool(net, name='pool3', kh=2, kw=2, dh=2, dw=2)

#     # block 4 -- outputs 14x14x512
#     net = L.conv(net, name='conv4_1', kh=3, kw=3, n_out=512)
#     net = L.conv(net, name='conv4_2', kh=3, kw=3, n_out=512)
#     net = L.conv(net, name='conv4_3', kh=3, kw=3, n_out=512)
#     net = L.pool(net, name='pool4', kh=2, kw=2, dh=2, dw=2)

#     # block 5 -- outputs 7x7x512
#     net = L.conv(net, name='conv5_1', kh=3, kw=3, n_out=512)
#     net = L.conv(net, name='conv5_2', kh=3, kw=3, n_out=512)
#     net = L.conv(net, name='conv5_3', kh=3, kw=3, n_out=512)
#     net = L.pool(net, name='pool5', kh=2, kw=2, dw=2, dh=2)

#     # flatten
#     flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
#     net = tf.reshape(net, [-1, flattened_shape], name='flatten')

#     # fully connected
#     net = L.fc(net, name='fc6', n_out=4096)
#     net = tf.nn.dropout(net, keep_prob)
#     net = L.fc(net, name='fc7', n_out=4096)
#     net = tf.nn.dropout(net, keep_prob)
#     net = L.fc(net, name='fc8', n_out=n_classes)
#     return net

