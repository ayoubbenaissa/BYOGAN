# byogan

byo-gan product for configuring, training and interacting with GANs.

## Build Setup

** Use Docker [link](https://docs.docker.com/get-docker/):
```
install docker & docker-compose
run: docker-compose up
```

```
install python dependencies:
```
cd  api
pip install requirements.txt
```
or
```
cd api
conda env create -f env.yml
```

### Gan Presentation
A [Generative Adversarial Network](https://arxiv.org/abs/1406.2661) GAN is a neural network based model that aims to learn the Data Distribution of a given Dataset in order to sample from this learned distribution and generate realistic looking Data, in other words after a successful learning process a GAN can effitiently produce Data that is hardly indistinguishable from real Data.
A GAN is composed of two competing networks: a **Generator** and a **Discriminator**
+ the Generator uses input from a Latent Vector and tries to produce realistic Data
+ the Discriminator, having access to real Data, tries to classify input into two classes: Real data and Fake Data.

Since the introduction of GANs, [many models](https://machinelearningmastery.com/tour-of-generative-adversarial-network-models/) have been implemented based on this architecture, but less effort has been put to understand parameters affecting GANs. Also GANs remain a difficult subject to beginners.

### Product Presentation
That is why we propose this product which allows advanced and customised configuration of GANs, helps track the learning of the configured model via the visualization of the generated samples as well as via the evolution of the network losses.

The product allows:
+ configure a training Dataset:
   choose from the knwon Datasets (MNIST, FashionMNIST ...) or select an Image Folder
   ,define the batch size
   ,define the image properties (size, RGB vs Gray)
+ configure models:
   select the preferred arhctiecture (Vanilla, Wassertein, Deep Convolutional)
   ,configure the noetwork parameters (number of layers, add Drop Out/BatchNormalization)
   ,select the network initialization method
+ configure network Losses (Binary Cross Entropy) and Optimizers (Adam, RMS Prop, SGD)
+ configure the Latent Vector
+ apply some training tricks
+ Visualization
