1. Start by configuring the Dataset (chose either a defined Dataset namely: MNIST/FashionMNIST, or a local dataset: image folder/CSV file). Then configure the path:
    i/ In case of a defined Dataset, this path should point to the Dataset and if it is not found there it will be downloaded by the tool.
    ii/ In case of a local Dataset, the path should point to the Dataset and if it is wrobgly configured this may lead to errors.
Afterwards, the batch-size (number of images per learning iteration) can be configured.
Then image information can be configured: image shape & image channels (exp: 1 for gray images, 3 for RGB).

2. Configuration of the Generator:
    i/ start by defining the desired architecture (GAN, WGAN, DCGAN)
    ii/ configure relevant parameters (input channels size, number of hidden layers, negative-slope in case of using LeakyReLU, applying drop-outs, using GPU for faster calculations)
    iii/ definine the initialization: Default, Uniform, Normal, Xavier-Uniform, Xavier-Normal, Kaiming-Uniform, Kaiming-Normal.

3. Configuration of the Discriminator:
    i/ can be applied using the configuration of the Generator (so both networks will have "symmetric" configuration)
    ii/ can be independently configued.

4. Loss Functions (Binary Cross Entropy, Mean Square Error)

5. Optimizers: Adam, RMS-Prop, SGD.
relevent parameters can be configured such as learning rate...

6. Latent Vector: can use Uniform, Gaussian or Multivariate Gaussian distribution.

7. Training Tricks: apply some training enhancements

8. Define the number of batches to be processed. After processing this number of batches, the tool gives back visualization metrcis to track the learning process.
Define the number of images (generated and real samples) to be displayed.
IMPORTANT: the number of images can NOT be greater than the square of the batch size, because this number of iamges will be used to display samples from the training of G and D.

9. Monitor the training through the metrics (samples, losses evolution, KL/JS divergences, reduced data, Discriminator metrics...)
