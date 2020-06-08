
#reshape data into vector:
def reshape(modelD, real, fake, batch_size, img_size, n_channels):
    #if we have Fully-Connected (Linear) layers => transform images into vectors
    if (modelD != 'DCGAN'):
        real = real.view(batch_size, n_channels*img_size*img_size)
        fake = fake.view(batch_size, n_channels*img_size*img_size)
    #or prepare data to fit into Conv layers => 4d tensor (batch_size, img_size, _img_size, channels)
    else:
        real = real.view(batch_size, n_channels, img_size, img_size)
        fake = fake.view(batch_size, n_channels, img_size, img_size)
    return real, fake

#used to transform RGB image to Gray (useful for Reducing the Data)
def rgb_to_gray(real_sample, fake_sample, img_size, n_channels):
    # transform RGB image to Gray-Scale
    if (n_channels == 3):
        real_sample = real_sample.view(n_channels, img_size, img_size)
        fake_sample = fake_sample.view(n_channels, img_size, img_size)
        real_sample = 0.2989 * real_sample[0, :, :] + 0.5870 * real_sample[1, :, :] + 0.1140 * real_sample[2, :, :]
        fake_sample = 0.2989 * fake_sample[0, :, :] + 0.5870 * fake_sample[1, :, :] + 0.1140 * fake_sample[2, :, :]
    if (n_channels == 1):
        real_sample = real_sample.view(img_size, img_size)
        fake_sample = fake_sample.view(img_size, img_size)
    return real_sample, fake_sample

#initialization of the training necessary elements:
def initNecessaryElements():
    #will contain all necessary elements for GAN training (dataloader, models, optimizers, losses)
    necessary_elements = {}
    necessary_elements['index_batch'] = 0
    necessary_elements['epoch_number'] = 0
    necessary_elements['training'] = True
    necessary_elements['flip'] = False
    necessary_elements['smooth'] = False
    necessary_elements['apply_gp'] = False
    necessary_elements['lambda_gp'] = False
    return necessary_elements

#calculates metrics for Discriminator:
def calc_metrics(d_real, d_fake, Precision, Recall, F1_score):
    pp = 0
    pn = 0
    np = 0
    nn = 0
    for x in d_real:
        if (x.item() > 0.5):
            pp = pp + 1
        else: pn = pn + 1
    for x in d_fake:
        if (x.item() < 0.5):
            nn = nn +1
        else: np = np +1
    precision = (pp/(pp+pn+1))
    recall = (pp/(pp+np+1))
    Precision.append(precision)
    Recall.append(recall)
    F1_score.append((2*precision*recall)/(precision + recall + 0.00001))