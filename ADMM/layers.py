'''
Define all layers convolutional (conv), fully connected (fc), pooling (pool) 
and element-wise multiplication (emult). 
Define functions to compute top K error & averaged gradient
'''

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from scipy.linalg import hadamard


def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, padding='SAME'):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding=padding)
        activation = tf.nn.bias_add(conv, biases)
        return activation, weights


def fc(input_tensor, name, n_out):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        return logits, weights


def pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)

'''
Hadamard transform layer
'''
def hmtblock2d(input_tensor, block_size, name, n_out):
    n_in = input_tensor.get_shape()[-1].value
    ih = input_tensor.get_shape()[1].value
    iw = input_tensor.get_shape()[2].value

    n_block = int(ih/block_size)

    hmtmat = hadamard(block_size, dtype=np.float32)
    hmtmat = hmtmat/np.sqrt(block_size)
    ihmtmat = np.linalg.inv(hmtmat)


    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [block_size, block_size, n_in, n_out], tf.float32, xavier_initializer())
        H = tf.constant(hmtmat, name='H')
        IH = tf.constant(ihmtmat, name='IH')

        input_block = []
        for hidx in range(n_block):
            input_block_w = []
            for widx in range(n_block):
                patch = input_tensor[:, hidx*block_size:(hidx+1)*block_size, widx*block_size:(widx+1)*block_size, :] # batch by blk_size by blk_size by n_in 
                patch_t = tf.tensordot(H, patch, axes=[[1],[1]]) # blk_size by batch_size by blk_size by n_in tensor
                patch_t = tf.tensordot(patch_t, H, axes=[[2],[0]]) # blk_size by batch_size by n_in by blk_size tensor
                patch_t = tf.transpose(patch_t, [1,0,3,2]) # batch by blk_size by blk_size by n_in
                input_block_w.append(patch_t)
            input_block.append(tf.stack([input_block_w[i] for i in range(n_block)], axis=-1)) # batch by blk_size by blk_size by n_in by n_block tensor
        input_hmt = tf.stack([input_block[i] for i in range(n_block)], axis=-1) # batch by blk_size by blk_size by n_in by n_block by n_block tensor
        input_hmt = tf.transpose(input_hmt, [0,4,5,1,2,3]) # batch by n_block by n_block by blk_size by blk_size by n_in tensor

        hmult2d = []
        for idx in range(n_out):
            hmult2d.append(tf.reduce_sum(input_hmt*weights[:,:,:,idx], axis=-1)) # batch by n_block by n_block by blk_size by blk_size tensor
        hmult = tf.stack([hmult2d[i] for i in range(n_out)], axis=-1) # batch by n_block by n_block by blk_size by blk_size by n_out tensor

        # apply inverse Hadamard transform
        hmult = tf.tensordot(IH, hmult, axes=[[1],[3]]) # blk_size by batch_size by n_block by n_block by blk_size by n_out tensor
        hmult = tf.tensordot(hmult, IH, axes=[[4],[0]]) # blk_size by batch_size by n_block by n_block by n_out by blk_size tensor
        hmult = tf.transpose(hmult, [1,0,5,4,2,3]) # batch by blk_size by blk_size by n_out by n_block by n_block tensor

        hmult = tf.concat([hmult[:,:,:,:,i,:] for i in range(n_block)], axis=2) # batch_size by blk_size by blk_size by n_out by n_block tensor
        hmult = tf.concat([hmult[:,:,:,:,i] for i in range(n_block)], axis=1) # batch_size by blk_size by blk_size by n_out tensor

        print(hmult.get_shape().as_list())

        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        activation = tf.nn.bias_add(hmult, biases)
        return activation, weights

'''
2D element-wise multiplication
'''
def emult2d(input_tensor, name, n_out):
    n_in = input_tensor.get_shape()[-1].value
    ih = input_tensor.get_shape()[1].value
    iw = input_tensor.get_shape()[2].value
    with tf.variable_scope(name):
        emult = []
        for idx in range(n_out):
            weights = tf.get_variable('weights_%s' % idx, [ih, iw, n_in], tf.float32, xavier_initializer())
            emult.append(tf.reduce_sum(tf.math.multiply(input_tensor, weights), axis=-1))

        emults = tf.stack([emult[i] for i in range(n_out)], axis=-1)
        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        activation = tf.nn.bias_add(emults, biases)
        return activation

'''
1D element-wise multiplication
'''
def emult(input_tensor, name, n_out):
    n_in = input_tensor.get_shape()[-1].value
    klen = input_tensor.get_shape()[1].value
    with tf.variable_scope(name):
        emult = []
        for idx in range(n_out):
            weights = tf.get_variable('weights_%s' % idx, [klen, n_in], tf.float32, xavier_initializer())
            emult.append(tf.reduce_sum(tf.math.multiply(input_tensor, weights), axis=-1))

        emults = tf.stack([emult[i] for i in range(n_out)], axis=-1)
        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        activation = tf.nn.bias_add(emults, biases)
        return activation

'''
layer with Winograd transform 2x2, 3x3
'''
def wino(input_tensor, name, n_out):
    n_in = input_tensor.get_shape()[-1].value
    ih = input_tensor.get_shape()[1].value
    iw = input_tensor.get_shape()[2].value

    Bmat = np.array([[1,0,0,0], [0,1,-1,1], [-1,1,1,0], [0,0,0,-1]], dtype=np.float32)
    BmatT = Bmat.transpose()
    Gmat = np.array([[1,0,0], [0.5,0.5,0.5], [0.5,-0.5,0.5], [0,0,1]], dtype=np.float32)
    GmatT = Gmat.transpose()
    Amat = np.array([[1,0], [1,1], [1,-1], [0,-1]], dtype=np.float32)
    AmatT = Amat.transpose()

    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [3, 3, n_in, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        B = tf.constant(Bmat, name='B')
        BT = tf.constant(BmatT, name='BT')
        G = tf.constant(Gmat, name='G')
        GT = tf.constant(GmatT, name='GT')
        A = tf.constant(Amat, name='A')
        AT = tf.constant(AmatT, name='AT')
        paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
        input_tensor_padding = tf.pad(input_tensor, paddings, 'CONSTANT')
        
        h_out = []
        for hidx in range(0, ih, 2):
            w_out = []
            for widx in range(0, iw, 2):
                patch = input_tensor_padding[:, hidx:hidx+4, widx:widx+4, :] # batch_size by 4 by 4 by n_in tensor
                block_out = []
                for out_idx in range(n_out):
                    patch_tmp = tf.tensordot(BT, patch, axes=[[1],[1]]) # 4 by batch_size by 4 by n_in tensor
                    patch_tmp = tf.tensordot(patch_tmp, B, axes=[[2],[0]]) # 4 by batch_size by n_in by 4 tensor
                    
                    weight_tmp = tf.tensordot(G, weights[:,:,:,out_idx], axes=[[1],[0]]) # 4 by 3 by n_in
                    weight_tmp = tf.tensordot(weight_tmp, GT, axes=[[1],[0]]) # 4 by n_in by 4 tensor 

                    patch_tmp = tf.transpose(patch_tmp, [1,0,3,2]) # batch_size by 4 by 4 by n_in tensor
                    weight_tmp = tf.transpose(weight_tmp, [0,2,1]) # 4 by 4 by n_in tensor 
                    out_tmp = tf.math.multiply(weight_tmp, patch_tmp) # batch_size by 4 by 4 by n_in tensor    

                    out_tmp = tf.tensordot(AT, out_tmp, axes=[[1],[1]]) # 2 by batch_size by 4 by n_in                    
                    out_tmp = tf.tensordot(out_tmp, A, axes=[[2],[0]]) # 2 by batch_size by n_in by 2 tensor
                    
                    patchout = tf.transpose(out_tmp, [1,0,3,2]) # batch_size by 2 by 2 by n_in tensor
                    patchout = tf.reduce_sum(patchout, axis=-1) # batch_size by 2 by 2 tensor 
                    
                    block_out.append(patch)
                blockout = tf.stack([block_out[i] for i in range(n_out)], axis=-1) # batch_size by 2 by 2 by n_out tensor

                w_out.append(blockout)
            wout = tf.concat([w_out[i] for i in range(int(iw/2))], axis=2) # batch_size by 2 by kw by n_out tensor
            
            h_out.append(wout)
        wino = tf.concat([h_out[i] for i in range(int(ih/2))], axis=1) # batch_size by hw by kw by n_out tensor
        print(wino.get_shape().as_list())

        activation = tf.nn.bias_add(wino, biases)
        return activation

'''
2D fft layer, input tensor & output tensor are in type tf.float32, n by n element-wize multiplication
'''
def fftmult2d(input_tensor, name, n_out):
    n_in = input_tensor.get_shape()[-1].value

    paddings = tf.constant([[0,0],[0,2],[0,2],[0,0]])
    input_tensor = tf.pad(input_tensor, paddings, 'CONSTANT')

    ih = input_tensor.get_shape()[1].value
    iw = input_tensor.get_shape()[2].value

    input_tensor = tf.transpose(input_tensor, [0,3,1,2]) # batch by i_in by ih by iw tensor
    input_fft = tf.signal.fft2d(tf.cast(input_tensor, tf.complex64))
    input_fft = tf.transpose(input_tensor, [0,2,3,1]) # batch by ih by iw by i_in tensor
    input_fft_real = tf.math.real(input_fft)
    input_fft_imag = tf.math.imag(input_fft)

    with tf.variable_scope(name):
        weights_real, weights_imag = fftweights(ih, iw, n_in, n_out)
        fftmult2d_real = []
        fftmult2d_imag = []

        for idx in range(n_out):             
            fftmult2d_real.append(tf.reduce_sum(input_fft_real*weights_real[:,:,:,idx]-input_fft_imag*weights_imag[:,:,:,idx], axis=-1))
            fftmult2d_imag.append(tf.reduce_sum(input_fft_real*weights_imag[:,:,:,idx]+input_fft_imag*weights_real[:,:,:,idx], axis=-1))

        fftmult2ds_real = tf.stack([fftmult2d_real[i] for i in range(n_out)], axis=-1)
        fftmult2ds_imag = tf.stack([fftmult2d_imag[i] for i in range(n_out)], axis=-1)
        fftmult2ds = tf.complex(fftmult2ds_real, fftmult2ds_imag)
        fftmult2ds = tf.transpose(fftmult2ds, [0,3,1,2]) # #batch by i_out by ih by iw tensor
        fftmult2ds_ifft = tf.signal.ifft2d(fftmult2ds)
        fftmult = fftmult2ds_ifft[:,:,1:ih-1,1:iw-1]
        fftmult = tf.transpose(fftmult, [0,2,3,1]) # batch by ih by iw by i_out tensor

        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        activation = tf.nn.bias_add(tf.math.real(fftmult), biases)
        return activation


'''
2D fft layer, input tensor & output tensor are in type tf.float32, b by b block element-wize multiplication
Assume conv kernel has size 3 by 3.
'''
def fftmultblock2d(input_tensor, block_size, name, n_out):
    n_in = input_tensor.get_shape()[-1].value
    ih = input_tensor.get_shape()[1].value
    iw = input_tensor.get_shape()[2].value

    paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
    input_tensor = tf.pad(input_tensor, paddings, 'CONSTANT')

    kh = block_size + 2
    kw = block_size + 2
    n_block = int(ih/block_size)

    with tf.variable_scope(name):
        weights_real, weights_imag, weights_real_var, weights_imag_var = fftweights(kh, kw, n_in, n_out)
        # weights_real, weights_imag = fftweightsConv(kh, kw, n_in, n_out)

        input_block = []
        for hidx in range(n_block):
            input_block_w = []
            for widx in range(n_block):
                input_patch = input_tensor[:, hidx*block_size:hidx*block_size+kh, widx*block_size:widx*block_size+kw, :]
                input_block_w.append(tf.roll(input_patch, shift=[-1,-1], axis=[1,2]))
            input_block.append(tf.stack([input_block_w[i] for i in range(n_block)], axis=-1)) # batch by kh by kw by n_in by n_block tensor
        input_blocks = tf.stack([input_block[i] for i in range(n_block)], axis=-1) # batch by kh by kw by n_in by n_block by n_block tensor

        input_blocks = tf.transpose(input_blocks, [0,4,5,3,1,2]) # batch by n_block by n_block by n_in by kh by kw tensor
        input_fft = tf.signal.fft2d(tf.cast(input_blocks, tf.complex64))
        input_fft = tf.transpose(input_fft, [0,1,2,4,5,3]) # batch by n_block by n_block by kh by kw by n_in tensor
        input_fft_real = tf.math.real(input_fft)
        input_fft_imag = tf.math.imag(input_fft)
        fftmult2d = []
        for idx in range(n_out):                           
            block_out_real = input_fft_real*weights_real[:,:,:,idx]-input_fft_imag*weights_imag[:,:,:,idx]
            block_out_imag = input_fft_real*weights_imag[:,:,:,idx]+input_fft_imag*weights_real[:,:,:,idx]
            fft_block = tf.complex(block_out_real, block_out_imag) # batch by n_block by n_block by kh by kw by n_in tensor
            fftmult2d.append(tf.reduce_sum(fft_block, axis=-1)) # batch by n_block by n_block by kh by kw tensor

        fftmult = tf.stack([fftmult2d[i] for i in range(n_out)], axis=-1) # batch by n_block by n_block by kh by kw by n_out tensor
        fftmult = tf.transpose(fftmult, [0,1,2,5,3,4]) # #batch by n_block by n_block by n_out by kh by kw tensor
        fftmult = tf.signal.ifft2d(fftmult)
        fftmult = fftmult[:,:,:,:,1:block_size+1,1:block_size+1]
        fftmult = tf.transpose(fftmult, [0,4,5,3,1,2]) # batch by kh by kw by n_out by n_block by n_block tensor
        fftmult = tf.math.real(fftmult)

        fftmult = tf.concat([fftmult[:,:,:,:,i,:] for i in range(n_block)], axis=2) # batch_size by kh by iw by n_out by n_block tensor
        fftmult = tf.concat([fftmult[:,:,:,:,i] for i in range(n_block)], axis=1) # batch_size by ih by iw by n_out tensor

        print(fftmult.get_shape().as_list())

        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        activation = tf.nn.bias_add(fftmult, biases)
        return activation, weights_real_var, weights_imag_var


'''
2D fft layer, input tensor & output tensor are in type tf.float32, b by b block element-wize multiplication
Assume conv kernel has size 3 by 3.
'''
def fftmultblock2d_old(input_tensor, block_size, name, n_out):
    n_in = input_tensor.get_shape()[-1].value
    ih = input_tensor.get_shape()[1].value
    iw = input_tensor.get_shape()[2].value

    paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
    input_tensor = tf.pad(input_tensor, paddings, 'CONSTANT')

    kh = block_size + 2
    kw = block_size + 2
    n_block = int(ih/block_size)

    with tf.variable_scope(name):
        weights_real, weights_imag, weights_real_var, weights_imag_var = fftweights(kh, kw, n_in, n_out)

        fftmult2d = []
        for idx in range(n_out):
            fftmult2d_h = []
            for hidx in range(n_block):
                fftmult2d_w = []
                for widx in range(n_block):
                    input_block = input_tensor[:, hidx*block_size:hidx*block_size+kh, widx*block_size:widx*block_size+kw, :]
                    input_block = tf.roll(input_block, shift=[-1,-1], axis=[1,2])
                    input_block = tf.transpose(input_block, [0,3,1,2]) # batch by i_in by kh by kw tensor
                    
                    input_fft = tf.signal.fft2d(tf.cast(input_block, tf.complex64))
                    input_fft = tf.transpose(input_fft, [0,2,3,1]) # batch by kh by kw by i_in tensor
                    input_fft_real = tf.math.real(input_fft)
                    input_fft_imag = tf.math.imag(input_fft)

                    block_out_real = input_fft_real*weights_real[:,:,:,idx] - input_fft_imag*weights_imag[:,:,:,idx]
                    block_out_imag = input_fft_real*weights_imag[:,:,:,idx] + input_fft_imag*weights_real[:,:,:,idx]
                    fftmult_block = tf.complex(block_out_real, block_out_imag)
                    fftmult_block = tf.transpose(fftmult_block, [0,3,1,2]) # #batch by i_out by kh by kw tensor
                    fftmult_block = tf.signal.ifft2d(fftmult_block)
                    fftmult_block = tf.transpose(fftmult_block, [0,2,3,1]) # batch by kh by kw by i_out tensor
                    fftmult_block = tf.math.real(fftmult_block)

                    fftmult2d_w.append(tf.reduce_sum(fftmult_block[:,1:block_size+1,1:block_size+1,:], axis=-1))

                fftmult2d_wout = tf.concat([fftmult2d_w[i] for i in range(n_block)], axis=2) # batch_size by block_size by iw tensor
                fftmult2d_h.append(fftmult2d_wout)

            fftmult2d_hout = tf.concat([fftmult2d_h[i] for i in range(n_block)], axis=1) # batch_size by ih by iw tensor
            fftmult2d.append(fftmult2d_hout)

        fftmult = tf.stack([fftmult2d[i] for i in range(n_out)], axis=-1)
        print(fftmult.get_shape().as_list())

        biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
        activation = tf.nn.bias_add(fftmult, biases)
        return activation, weights_real_var, weights_imag_var


# def fftmultblock2d(input_tensor, block_size, name, n_out):
#     n_in = input_tensor.get_shape()[-1].value
#     ih = input_tensor.get_shape()[1].value
#     iw = input_tensor.get_shape()[2].value

#     paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
#     input_tensor = tf.pad(input_tensor, paddings, 'CONSTANT')

#     kh = block_size + 2
#     kw = block_size + 2
#     n_block = int(ih/block_size)

#     with tf.variable_scope(name):
#         weights_real, weights_imag, weights_real_var, weights_imag_var = fftweights(kh, kw, n_in, n_out)
#         # weights_real, weights_imag = fftweightsConv(kh, kw, n_in, n_out)
#         fftmult2d_h = []
#         for hidx in range(n_block):
#             fftmult2d_w = []
#             for widx in range(n_block):
#                 input_block = input_tensor[:, hidx*block_size:hidx*block_size+kh, widx*block_size:widx*block_size+kw, :]
#                 input_block = tf.roll(input_block, shift=[-1,-1], axis=[1,2])
#                 input_block = tf.transpose(input_block, [0,3,1,2]) # batch by i_in by kh by kw tensor
#                 input_fft = tf.signal.fft2d(tf.cast(input_block, tf.complex64))
#                 input_fft = tf.transpose(input_fft, [0,2,3,1]) # batch by kh by kw by i_in tensor
#                 input_fft_real = tf.math.real(input_fft)
#                 input_fft_imag = tf.math.imag(input_fft)

#                 fftmult2d = []
#                 for idx in range(n_out):                           
#                     block_out_real = input_fft_real*weights_real[:,:,:,idx]-input_fft_imag*weights_imag[:,:,:,idx]
#                     block_out_imag = input_fft_real*weights_imag[:,:,:,idx]+input_fft_imag*weights_real[:,:,:,idx]

#                     fft_block = tf.complex(block_out_real, block_out_imag) # batch by kh by kw by i_in tensor
#                     fftmult2d.append(tf.reduce_sum(fft_block, axis=-1)) # batch by kh by kw tensor

#                 fftmult_block = tf.stack([fftmult2d[i] for i in range(n_out)], axis=-1) # batch by kh by kw by n_out tensor
#                 fftmult_block = tf.transpose(fftmult_block, [0,3,1,2]) # #batch by n_out by kh by kw tensor
#                 fftmult_block = tf.signal.ifft2d(fftmult_block)
#                 fftmult_block = tf.transpose(fftmult_block, [0,2,3,1]) # batch by kh by kw by n_out tensor
#                 fftmult_block = tf.math.real(fftmult_block)

#                 fftmult2d_w.append(fftmult_block[:,1:block_size+1,1:block_size+1,:])

#             fftmult2d_wout = tf.concat([fftmult2d_w[i] for i in range(n_block)], axis=2) # batch_size by block_size by iw by n_out tensor
#             fftmult2d_h.append(fftmult2d_wout)

#         fftmult = tf.concat([fftmult2d_h[i] for i in range(n_block)], axis=1) # batch_size by ih by iw by n_out tensor

#         print(fftmult.get_shape().as_list())

#         biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
#         activation = tf.nn.bias_add(fftmult, biases)
#         return activation, weights_real, weights_imag

'''
Define fft weights, apply constraints to make sure its ifft is real, new version to put all variables together
'''
def fftweights(ih, iw, n_in, n_out):
    if(ih%2 == 1): # ih odd
        weights_real_var = tf.get_variable('weights_real', [int((ih+1)/2)**2+int((ih-1)/2)**2, n_in, n_out], tf.float32, xavier_initializer()) 
        weights_imag_var = tf.get_variable('weights_imag', [int((ih+1)/2)**2+int((iw+1)/2)**2, n_in, n_out], tf.float32, xavier_initializer()) 

        weights_real1 = weights_real_var[:int((ih+1)/2)**2, :, :]
        weights_real1 = tf.reshape(weights_real1, [int((ih+1)/2),int((ih+1)/2),n_in,n_out])
        weights_imag1 = weights_imag_var[:int((ih+1)/2)**2, :, :]
        weights_imag1 = tf.reshape(weights_imag1, [int((ih+1)/2),int((ih+1)/2),n_in,n_out])
        weights_real2 = weights_real_var[int((ih+1)/2)**2:, :, :]
        weights_real2 = tf.reshape(weights_real2, [int((ih-1)/2),int((ih-1)/2),n_in,n_out])
        weights_imag2 = weights_imag_var[int((ih+1)/2)**2:, :, :]
        weights_imag2 = tf.reshape(weights_imag2, [int((ih-1)/2),int((ih-1)/2),n_in,n_out])

        weights_real_upper = tf.reverse(weights_real1[0,1:int((iw+1)/2),:,:], axis=[0]) # (iw-1)/2 by n_in tensor
        weights_real_upper = tf.reshape(weights_real_upper, [1,int((iw-1)/2),n_in,n_out]) # 1 by (iw-1)/2 by n_in tensor
        weights_imag_upper = -tf.reverse(weights_imag1[0,1:int((iw+1)/2),:,:], axis=[0])
        weights_imag_upper = tf.reshape(weights_imag_upper, [1,int((iw-1)/2),n_in,n_out])

        weights_real = tf.concat([weights_real_upper, weights_real2], axis=0) # (ih+1)/2 by (iw-1)/2 by n_in tensor
        weights_imag = tf.concat([weights_imag_upper, weights_imag2], axis=0)
        weights_real = tf.concat([weights_real1, weights_real], axis=1) # (ih+1)/2 by iw by n_in tensor
        weights_imag = tf.concat([weights_imag1, weights_imag], axis=1)

        weights_real_lower_left = tf.reverse(weights_real[1:,0,:,:], axis=[0]) #(ih-1)/2 by n_in tensor
        weights_real_lower_left = tf.reshape(weights_real_lower_left, [int((ih-1)/2),1,n_in,n_out]) #(ih-1)/2 by 1 by n_in tensor
        weights_imag_lower_left = -tf.reverse(weights_imag[1:,0,:,:], axis=[0])
        weights_imag_lower_left = tf.reshape(weights_imag_lower_left, [int((ih-1)/2),1,n_in,n_out])

        weights_real_lower_right = tf.reverse(weights_real[1:,1:,:,:],axis=[0,1]) # (ih-1)/2 by iw-1 by n_in tensor
        weights_real_lower_right = tf.reshape(weights_real_lower_right, [-1,int(iw-1),n_in,n_out])
        weights_imag_lower_right = -tf.reverse(weights_imag[1:,1:,:,:],axis=[0,1])
        weights_imag_lower_right = tf.reshape(weights_imag_lower_right, [-1,int(iw-1),n_in,n_out])

        weights_real_lower = tf.concat([weights_real_lower_left, weights_real_lower_right], axis=1) # (ih-1)/2 by iw by n_in tensor
        weights_imag_lower = tf.concat([weights_imag_lower_left, weights_imag_lower_right], axis=1)
        weights_real = tf.concat([weights_real, weights_real_lower], axis=0) # ih by iw by n_in tensor
        weights_imag = tf.concat([weights_imag, weights_imag_lower], axis=0)

        mask = np.ones([ih,iw,n_in,n_out])
        mask[0,0,:,:] = 0
        masks = tf.constant(mask, name='mask', dtype=tf.float32)
        weights_imag = weights_imag*masks

    else: # ih even
        weights_real_var = tf.get_variable('weights_real', [int(ih/2+1)**2+int(ih/2-1)**2, n_in, n_out], tf.float32, xavier_initializer()) 
        weights_imag_var = tf.get_variable('weights_imag', [int(ih/2+1)**2+int(ih/2-1)**2, n_in, n_out], tf.float32, xavier_initializer()) 

        weights_real1 = weights_real_var[:int(ih/2+1)**2, :, :]
        weights_real1 = tf.reshape(weights_real1, [int(ih/2+1),int(ih/2+1),n_in,n_out])
        weights_imag1 = weights_imag_var[:int(ih/2+1)**2, :, :]
        weights_imag1 = tf.reshape(weights_imag1, [int(ih/2+1),int(ih/2+1),n_in,n_out])
        weights_real2 = weights_real_var[int(ih/2+1)**2:, :, :]
        weights_real2 = tf.reshape(weights_real2, [int(ih/2-1),int(ih/2-1),n_in,n_out])
        weights_imag2 = weights_imag_var[int(ih/2+1)**2:, :, :]
        weights_imag2 = tf.reshape(weights_imag2, [int(ih/2-1),int(ih/2-1),n_in,n_out])

        weights_real_upper = tf.reverse(weights_real1[0,1:int(iw/2),:,:], axis=[0]) # iw/2-1 by n_in tensor
        weights_real_upper = tf.reshape(weights_real_upper, [1,int(iw/2-1),n_in,n_out]) # 1 by iw/2-1 by n_in tensor
        weights_imag_upper = -tf.reverse(weights_imag1[0,1:int(iw/2),:,:], axis=[0])
        weights_imag_upper = tf.reshape(weights_imag_upper, [1,int(iw/2-1),n_in,n_out])

        weights_real_mid = tf.reverse(weights_real1[-1,1:int(iw/2),:,:], axis=[0]) #iw/2-1 by n_in tensor
        weights_real_mid = tf.reshape(weights_real_mid, [1,int(iw/2-1),n_in,n_out]) # 1 by iw/2-1 by n_in tensor
        weights_imag_mid = tf.reverse(weights_imag1[-1,1:int(iw/2),:,:], axis=[0])
        weights_imag_mid = tf.reshape(weights_imag_mid, [1,int(iw/2-1),n_in,n_out])

        weights_real = tf.concat([weights_real_upper, weights_real2, weights_real_mid], axis=0) # ih/2+1 by iw/2-1 by n_in tensor
        weights_imag = tf.concat([weights_imag_upper, weights_imag2, weights_imag_mid], axis=0)
        weights_real = tf.concat([weights_real1, weights_real], axis=1) # ih/2+1 by iw by n_in tensor
        weights_imag = tf.concat([weights_imag1, weights_imag], axis=1)

        weights_real_lower_left = tf.reverse(weights_real[1:int(ih/2),0,:,:], axis=[0]) #ih/2-1 by n_in tensor
        weights_real_lower_left = tf.reshape(weights_real_lower_left, [int(ih/2-1),1,n_in,n_out]) #ih/2-1 by 1 by n_in tensor
        weights_imag_lower_left = -tf.reverse(weights_imag[1:int(ih/2):,0,:,:], axis=[0])
        weights_imag_lower_left = tf.reshape(weights_imag_lower_left, [int(ih/2-1),1,n_in,n_out])

        weights_real_lower_right = tf.reverse(weights_real[1:int(ih/2),1:,:,:],axis=[0,1]) # ih/2-1 by iw-1 by n_in tensor
        weights_real_lower_right = tf.reshape(weights_real_lower_right, [-1,int(iw-1),n_in,n_out])
        weights_imag_lower_right = -tf.reverse(weights_imag[1:int(ih/2),1:,:,:],axis=[0,1])
        weights_imag_lower_right = tf.reshape(weights_imag_lower_right, [-1,int(iw-1),n_in,n_out])

        weights_real_lower = tf.concat([weights_real_lower_left, weights_real_lower_right], axis=1) # ih/2-1 by iw by n_in tensor
        weights_imag_lower = tf.concat([weights_imag_lower_left, weights_imag_lower_right], axis=1)
        weights_real = tf.concat([weights_real, weights_real_lower], axis=0) # ih by iw by n_in tensor
        weights_imag = tf.concat([weights_imag, weights_imag_lower], axis=0)

        mask = np.ones([ih,iw,n_in,n_out])
        mask[0,0,:,:] = 0
        mask[0,int(iw/2),:,:] = 0
        mask[int(ih/2),0,:,:] = 0
        mask[int(ih/2),int(iw/2),:,:] = 0
        masks = tf.constant(mask, name='mask', dtype=tf.float32)
        weights_imag = weights_imag*masks

    return weights_real, weights_imag, weights_real_var, weights_imag_var

def fftweightsConv(ih, iw, n_in, n_out):
    weights = tf.get_variable('weights', [n_in, n_out, 3, 3], tf.float32, xavier_initializer())
    paddings = tf.constant([[0,0],[0,0],[int((ih-3)/2),ih-3-int((ih-3)/2)],[int((ih-3)/2),ih-3-int((ih-3)/2)]])
    weights_fft = tf.signal.fft2d(tf.cast(tf.pad(input_tensor, weights, 'CONSTANT'), tf.complex64))
    weights_real = tf.transpose(tf.math.real(weights_fft), [2,3,0,1])
    weights_imag = tf.transpose(tf.math.imag(weights_fft), [2,3,0,1])
    return weights_real, weights_imag

# '''
# 2D fft layer, input tensor & output tensor are in type tf.float32, b by b block element-wize multiplication
# Assume conv kernel has size 3 by 3.
# '''
# def fftmultblock2d(input_tensor, block_size, name, n_out):
#     n_in = input_tensor.get_shape()[-1].value
#     ih = input_tensor.get_shape()[1].value
#     iw = input_tensor.get_shape()[2].value

#     paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
#     input_tensor = tf.pad(input_tensor, paddings, 'CONSTANT')

#     kh = block_size + 2
#     kw = block_size + 2
#     n_block = int(ih/block_size)

#     with tf.variable_scope(name):
#         fftmult2d = []
#         for idx in range(n_out):
#             weights_real, weights_imag = fftweights(kh, kw, idx, n_in)
#             fftmult2d_h = []
#             for hidx in range(n_block):
#                 fftmult2d_w = []
#                 for widx in range(n_block):
#                     input_block = input_tensor[:, hidx*block_size:hidx*block_size+kh, widx*block_size:widx*block_size+kw, :]
#                     input_block = tf.roll(input_block, shift=[-1,-1], axis=[1,2])
#                     input_block = tf.transpose(input_block, [0,3,1,2]) # batch by i_in by kh by kw tensor
                    
#                     input_fft = tf.signal.fft2d(tf.cast(input_block, tf.complex64))
#                     input_fft = tf.transpose(input_fft, [0,2,3,1]) # batch by kh by kw by i_in tensor
#                     input_fft_real = tf.math.real(input_fft)
#                     input_fft_imag = tf.math.imag(input_fft)

#                     block_out_real = input_fft_real*weights_real-input_fft_imag*weights_imag
#                     block_out_imag = input_fft_real*weights_imag+input_fft_imag*weights_real
#                     fftmult_block = tf.complex(block_out_real, block_out_imag)
#                     fftmult_block = tf.transpose(fftmult_block, [0,3,1,2]) # #batch by i_out by kh by kw tensor
#                     fftmult_block = tf.signal.ifft2d(fftmult_block)
#                     fftmult_block = tf.transpose(fftmult_block, [0,2,3,1]) # batch by kh by kw by i_out tensor
#                     fftmult_block = tf.math.real(fftmult_block)

#                     fftmult2d_w.append(tf.reduce_sum(fftmult_block[:,1:block_size+1,1:block_size+1,:], axis=-1))

#                 fftmult2d_wout = tf.concat([fftmult2d_w[i] for i in range(n_block)], axis=2) # batch_size by block_size by iw tensor
#                 fftmult2d_h.append(fftmult2d_wout)

#             fftmult2d_hout = tf.concat([fftmult2d_h[i] for i in range(n_block)], axis=1) # batch_size by ih by iw tensor
#             fftmult2d.append(fftmult2d_hout)

#         fftmult = tf.stack([fftmult2d[i] for i in range(n_out)], axis=-1)
#         print(fftmult.get_shape().as_list())

#         biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
#         activation = tf.nn.bias_add(fftmult, biases)
#         return activation

# '''
# Define fft weights, apply constraints to make sure its ifft is real, old version
# '''
# def fftweights(ih, iw, idx, n_in):
#     if(ih%2 == 1): # ih odd
#         weights_real1 = tf.get_variable('weights_real1_%s' % idx, [int((ih+1)/2), int((iw+1)/2), n_in], tf.float32, xavier_initializer()) # (ih+1)/2 by (iw+1)/2 by n_in tensor
#         weights_imag1 = tf.get_variable('weights_imag1_%s' % idx, [int((ih+1)/2), int((iw+1)/2), n_in], tf.float32, xavier_initializer())
#         weights_real2 = tf.get_variable('weights_real2_%s' % idx, [int((ih-1)/2), int((iw-1)/2), n_in], tf.float32, xavier_initializer()) # (ih-1)/2 by (iw-1)/2 by n_in tensor
#         weights_imag2 = tf.get_variable('weights_imag2_%s' % idx, [int((ih-1)/2), int((iw-1)/2), n_in], tf.float32, xavier_initializer())

#         weights_real_upper = tf.reverse(weights_real1[0,1:int((iw+1)/2),:], axis=[0]) # (iw-1)/2 by n_in tensor
#         weights_real_upper = tf.reshape(weights_real_upper, [1,int((iw-1)/2),n_in]) # 1 by (iw-1)/2 by n_in tensor
#         weights_imag_upper = -tf.reverse(weights_imag1[0,1:int((iw+1)/2),:], axis=[0])
#         weights_imag_upper = tf.reshape(weights_imag_upper, [1,int((iw-1)/2),n_in])

#         weights_real = tf.concat([weights_real_upper, weights_real2], axis=0) # (ih+1)/2 by (iw-1)/2 by n_in tensor
#         weights_imag = tf.concat([weights_imag_upper, weights_imag2], axis=0)
#         weights_real = tf.concat([weights_real1, weights_real], axis=1) # (ih+1)/2 by iw by n_in tensor
#         weights_imag = tf.concat([weights_imag1, weights_imag], axis=1)

#         weights_real_lower_left = tf.reverse(weights_real[1:,0,:], axis=[0]) #(ih-1)/2 by n_in tensor
#         weights_real_lower_left = tf.reshape(weights_real_lower_left, [int((ih-1)/2),1,n_in]) #(ih-1)/2 by 1 by n_in tensor
#         weights_imag_lower_left = -tf.reverse(weights_imag[1:,0,:], axis=[0])
#         weights_imag_lower_left = tf.reshape(weights_imag_lower_left, [int((ih-1)/2),1,n_in])

#         weights_real_lower_right = tf.reverse(weights_real[1:,1:,:],axis=[0,1]) # (ih-1)/2 by iw-1 by n_in tensor
#         weights_real_lower_right = tf.reshape(weights_real_lower_right, [-1,int(iw-1),n_in])
#         weights_imag_lower_right = -tf.reverse(weights_imag[1:,1:,:],axis=[0,1])
#         weights_imag_lower_right = tf.reshape(weights_imag_lower_right, [-1,int(iw-1),n_in])

#         weights_real_lower = tf.concat([weights_real_lower_left, weights_real_lower_right], axis=1) # (ih-1)/2 by iw by n_in tensor
#         weights_imag_lower = tf.concat([weights_imag_lower_left, weights_imag_lower_right], axis=1)
#         weights_real = tf.concat([weights_real, weights_real_lower], axis=0) # ih by iw by n_in tensor
#         weights_imag = tf.concat([weights_imag, weights_imag_lower], axis=0)

#         mask = np.ones([ih,iw,n_in])
#         mask[0,0,:] = 0
#         masks = tf.constant(mask, name='mask', dtype=tf.float32)
#         weights_imag = weights_imag*masks

#     else: # ih even
#         weights_real1 = tf.get_variable('weights_real1_%s' % idx, [int(ih/2+1), int(iw/2+1), n_in], tf.float32, xavier_initializer()) # ih/2+1 by iw/2+1 by n_in tensor
#         weights_imag1 = tf.get_variable('weights_imag1_%s' % idx, [int(ih/2+1), int(iw/2+1), n_in], tf.float32, xavier_initializer())
#         weights_real2 = tf.get_variable('weights_real2_%s' % idx, [int(ih/2-1), int(iw/2-1), n_in], tf.float32, xavier_initializer()) # ih/2-1 by iw/2-1 by n_in tensor
#         weights_imag2 = tf.get_variable('weights_imag2_%s' % idx, [int(ih/2-1), int(iw/2-1), n_in], tf.float32, xavier_initializer())

#         weights_real_upper = tf.reverse(weights_real1[0,1:int(iw/2),:], axis=[0]) # iw/2-1 by n_in tensor
#         weights_real_upper = tf.reshape(weights_real_upper, [1,int(iw/2-1),n_in]) # 1 by iw/2-1 by n_in tensor
#         weights_imag_upper = -tf.reverse(weights_imag1[0,1:int(iw/2),:], axis=[0])
#         weights_imag_upper = tf.reshape(weights_imag_upper, [1,int(iw/2-1),n_in])

#         weights_real_mid = tf.reverse(weights_real1[-1,1:int(iw/2),:], axis=[0]) #iw/2-1 by n_in tensor
#         weights_real_mid = tf.reshape(weights_real_mid, [1,int(iw/2-1),n_in]) # 1 by iw/2-1 by n_in tensor
#         weights_imag_mid = tf.reverse(weights_imag1[-1,1:int(iw/2),:], axis=[0])
#         weights_imag_mid = tf.reshape(weights_imag_mid, [1,int(iw/2-1),n_in])

#         weights_real = tf.concat([weights_real_upper, weights_real2, weights_real_mid], axis=0) # ih/2+1 by iw/2-1 by n_in tensor
#         weights_imag = tf.concat([weights_imag_upper, weights_imag2, weights_imag_mid], axis=0)
#         weights_real = tf.concat([weights_real1, weights_real], axis=1) # ih/2+1 by iw by n_in tensor
#         weights_imag = tf.concat([weights_imag1, weights_imag], axis=1)

#         weights_real_lower_left = tf.reverse(weights_real[1:int(ih/2),0,:], axis=[0]) #ih/2-1 by n_in tensor
#         weights_real_lower_left = tf.reshape(weights_real_lower_left, [int(ih/2-1),1,n_in]) #ih/2-1 by 1 by n_in tensor
#         weights_imag_lower_left = -tf.reverse(weights_imag[1:int(ih/2):,0,:], axis=[0])
#         weights_imag_lower_left = tf.reshape(weights_imag_lower_left, [int(ih/2-1),1,n_in])

#         weights_real_lower_right = tf.reverse(weights_real[1:int(ih/2),1:,:],axis=[0,1]) # ih/2-1 by iw-1 by n_in tensor
#         weights_real_lower_right = tf.reshape(weights_real_lower_right, [-1,int(iw-1),n_in])
#         weights_imag_lower_right = -tf.reverse(weights_imag[1:int(ih/2),1:,:],axis=[0,1])
#         weights_imag_lower_right = tf.reshape(weights_imag_lower_right, [-1,int(iw-1),n_in])

#         weights_real_lower = tf.concat([weights_real_lower_left, weights_real_lower_right], axis=1) # ih/2-1 by iw by n_in tensor
#         weights_imag_lower = tf.concat([weights_imag_lower_left, weights_imag_lower_right], axis=1)
#         weights_real = tf.concat([weights_real, weights_real_lower], axis=0) # ih by iw by n_in tensor
#         weights_imag = tf.concat([weights_imag, weights_imag_lower], axis=0)

#         mask = np.ones([ih,iw,n_in])
#         mask[0,0,:] = 0
#         mask[0,int(iw/2),:] = 0
#         mask[int(ih/2),0,:] = 0
#         mask[int(ih/2),int(iw/2),:] = 0
#         masks = tf.constant(mask, name='mask', dtype=tf.float32)
#         weights_imag = weights_imag*masks

#     return weights_real, weights_imag

def loss(logits, onehot_labels):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_labels, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
    return loss

def topKError(predictions, labels, K=5):
    correct = tf.cast(tf.nn.in_top_k(predictions, labels, K), tf.float32)
    accuracy = tf.reduce_mean(correct)
    error = 1.0 - accuracy
    return error

def averageGradients(grads):
    '''
    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    '''
    average_grads = []
    for grad_and_vars in zip(*grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
