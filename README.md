# Deep Learning in Latent Space for Video Prediction and Compression
Codes for [Deep Learning in Latent Space for Video Prediction and Compression](https://github.com/BowenL0218/BPGAN/edit/main/README.md)(CVPR 2021), a latent prediction based video compression algorithm.

# Introduction and Framework
The proposed latent domain compression of individual frames is obtained by an auto-encoder DNN trained with a generative adversarial network (GAN) framework. To exploit the temporal correlation within the video frame sequence, we employ a convolutional long short-term memory (ConvLSTM) network to predict the latent vector representation of the future frame.

![Flow chart](https://github.com/BowenL0218/BPGAN/blob/main/Images/flow_chart.png)

## Architecture
The detailed neural network structure of our predictor model.
![Predictor architecture](https://github.com/BowenL0218/Video_Compression/blob/main/Images/predictor.png)
The detailed neural network structure of our decoder model.
![Decoder architecture](https://github.com/BowenL0218/Video_Compression/blob/main/Images/decoder.png)

## Datasets
In order to use the datasets used in the paper, please download the [UVG dataset](https://media.withyoutube.com/), the [Kinetics dataset](https://deepmind.com/research/open-source/kinetics), the [VTL dataset](http://trace.eas.asu.edu/index.html), and the [UVG dataset](http://ultravideo.fi/).

- The UVG and Kinetics dataset are used for training the prediction network. 
- The VTL and UVG datasets are implemented for testing the performance.
- Note that we use the learning based image compression algorithm ([Liu et al](https://arxiv.org/pdf/1912.03734.pdf)) as the intra compression for one single frame. 
- The compressed latents are the input for the prediction network. 

## ADMM quantization
To further reduce the bitrate of the compressed video, we applied ADMM quantization for the residual from latent prediction incorporated in the proposed video compression framework. 

## Arithmetic Coding
To use the entropy coding method in this paper, download the general code library in python with [arithmetic coding](https://github.com/ahmedfgad/ArithmeticEncodingPython). 

## Test pretrained model
To tested the result without ADMM quantization,
```sh
$ python test.py
```

To test the result with ADMM quantization
```sh
$ python Compression_ADMM.py
```

## Citation
Please cite our paper if you find our paper useful for your research. 
