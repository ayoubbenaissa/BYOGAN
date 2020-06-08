from DataSet.DataSet import Dataloader
from Discriminator.DCDiscriminator.DCDiscriminator import DCDiscriminator
from Discriminator.VDiscriminator.VDiscriminator import VDiscriminator
from Discriminator.WDiscriminator.WDiscriminator import WDiscriminator
from Generator.DCGenerator.DCGenerator import DCGenerator
from Generator.VGenerator.VGenerator import VGenerator
from Generator.WGenerator.WGenerator import WGenerator
from LatentVector.LatentVector import DCLatentVector, VLatentVector
from ModelsInit import init_model, get_init_method_description
from utils import reshape, rgb_to_gray, initNecessaryElements, calc_metrics
from loss import loss_function
from optimizer import model_optimizer
from TrainGAN import real_data, fake_data, train_generator, train_discriminator
from flask import Flask, request, jsonify, json
from flask_cors import CORS
import numpy as np 
from io import BytesIO
from skimage.io import imsave
import base64
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn.functional as F
from scipy.spatial import distance
import time
from torchvision.models import inception_v3

app = Flask(__name__)
#CORS is used to accept requests from the vue-server
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#will contain all necessary elements for GAN training (dataloader, models, optimizers, losses)
necessary_elements = initNecessaryElements()
#will be propagated from Generator to latentVector
LVmodel_generator = {}
LVchannelsG = {}
#elements to be sent back after training
result_elements = {}
D_Loss = []
D_Loss_real_min = []
D_Loss_real_mean = []
D_Loss_real_max = []
D_Loss_fake_min = []
D_Loss_fake_mean = []
D_Loss_fake_max = []
D_Loss_fake = {}
G_Loss = []
KL_div = []
JS_div = []
unrolling_real_batch = []

#PCA (used to reduce Data):
pca_real2D = PCA(n_components=2)
pca_generated2D = PCA(n_components=2)
pca_test2D = PCA(n_components=2)
reduced_real_data2D = []
reduced_generated_data2D = []
pca_real3D = PCA(n_components=3)
pca_generated3D = PCA(n_components=3)
reduced_real_data3D = []
reduced_generated_data3D = []

#endpoint to instantiate the Dataset/DataLoader
@app.route('/dataset', methods=['POST'])
def create_dataset():
    if request.method=='POST':
            #necessary_elements = initNecessaryElements()
            req_data = request.get_json()
            name = req_data['name'] #when working with predefined dataset (MNIST, FashionMNIST ...)
            path = req_data['path'] #folder/file path containing the training Data
            #Dataset/DataLoader properties
            necessary_elements['batch_size'] = int(req_data['batch_size'])
            necessary_elements['img_size'] = int(req_data['img_size'])
            necessary_elements['channels'] = int(req_data['channels'])
            #instantiate the Dataset/DataLoader
            dataloader = Dataloader(name, path, necessary_elements['batch_size'], necessary_elements['img_size'], necessary_elements['channels'])
            if(dataloader != None):
                #successful instantiation of the Dataset/DataLoader
                necessary_elements['dataloader'] = dataloader
                necessary_elements['loaderIteraor'] = iter(necessary_elements['dataloader'])
                return jsonify(
                        message = 'dataloader created successfully',
                        name = name,
                        path = path, 
                        batch_size = necessary_elements['batch_size'],
                        img_size = necessary_elements['img_size'],
                        channels = necessary_elements['channels'],
                        length= len(dataloader)
                 
                    )
            else:
                #failure in the instantiation of the Dataset/DataLoader
                return app.response_class(
                        response = json.dumps(dataloader),
                        status = 400,
                        mimetype='application/json'
                    )


                #inception = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor).eval()
                #print('inception: ', inception)

#endpoint to create the Discriminator Network
@app.route('/discriminator', methods=['POST'])
def instantiate_discriminator():
    if request.method=='POST':
            req_data = request.get_json()
            model_discriminator = req_data['type'] #types: CNN, ANN and Wasserstein (ANN + BatchNorm)
            drop_out = req_data['drop_out']
            leaky_relu = req_data['leaky_relu']
            n_layers = req_data['n_layers']
            #batch-normalization proeprties
            batch_norm_array = req_data['batch_norm']
            eps_array = req_data['eps']
            momentum_array = req_data['momentum']
            #in cased of ANN-based network, this will contain the number of units per hidden layer
            layers = req_data['layers']

            deviceDiscriminator = req_data['device']
            #whether to use Sigmoid(recommended) or TanH on the output of the Discriminator
            necessary_elements['symmetric_labels'] = req_data['out_activation']

            necessary_elements['model_discriminator'] = model_discriminator
            #initialization method of the network parameters
            necessary_elements['initD'] = req_data['init']

            if (model_discriminator == 'DCGAN'):
                DCdiscriminator = DCDiscriminator(
                    necessary_elements['channels'], leaky_relu, drop_out, n_layers,
                    batch_norm_array, eps_array, momentum_array,
                    necessary_elements['symmetric_labels'],
                    necessary_elements['img_size']).to(torch.device(deviceDiscriminator))
                necessary_elements['discriminator'] = DCdiscriminator

            elif (model_discriminator == 'VGAN'):
                Vdiscriminator = VDiscriminator(
                    layers, necessary_elements['channels']*necessary_elements['img_size']*necessary_elements['img_size'],
                    leaky_relu, drop_out, n_layers,
                    necessary_elements['symmetric_labels']).to(torch.device(deviceDiscriminator))
                necessary_elements['discriminator'] = Vdiscriminator

            elif (model_discriminator == 'WGAN'):
                Wdiscriminator = WDiscriminator(
                    layers, necessary_elements['channels'] * necessary_elements['img_size'] * necessary_elements['img_size'],
                    leaky_relu, drop_out, n_layers,
                    necessary_elements['symmetric_labels'],
                    batch_norm_array, eps_array, momentum_array
                ).to(torch.device(deviceDiscriminator))
                necessary_elements['discriminator'] = Wdiscriminator

            init_model(necessary_elements['discriminator'], necessary_elements['initD'])
            if (necessary_elements['discriminator'] != None):
                #successful creation of the Discriminator
                necessary_elements['deviceDiscriminator'] = deviceDiscriminator
                necessary_elements['model_discriminator'] = model_discriminator
                summary = str(necessary_elements['discriminator'])
                #init the model:
                initDescription = get_init_method_description(necessary_elements['initD'])
                return jsonify(
                        message = 'discriminator model created successfully',
                        model= summary,
                        device= necessary_elements['deviceDiscriminator'],
                        init= necessary_elements['initD'],
                        initDescription= initDescription
                    )

            else:
                #unsuccessful creation of the Discriminator
                return jsonify(
                    dumps= json.dumps(necessary_elements['discriminator']),
                    response= 'wrong model type',
                    status=500,
                    hint= 'try VGAN/DCGAN or WGAN or refer to documentation'
                )

#endpoint used to import a pre-trained Discriminator
#NOTE THAT: the user has to provide correct Network configuration then initialize the network then import the weight of the trained model
@app.route('/traineddiscriminator', methods=['POST'])
def trained_discriminator():
    if request.method=='POST':
        req_data = request.get_json()
        path = req_data['path']
        try:
            necessary_elements['discriminator'].load_state_dict(torch.load(path))
            necessary_elements['discriminator'].to(torch.device(necessary_elements['deviceDiscriminator']))
            summary = str(necessary_elements['discriminator'])
            return jsonify(
                message = 'trained discriminator model imported successfully',
                model= summary,
                device= necessary_elements['deviceDiscriminator']
            )
        except:
            return jsonify(
                    response= 'wrong configuration',
                    status=500
                )

#endpoint to save the parameters/weights of the trained Discriminator
@app.route('/savediscriminator', methods=['POST'])
def save_discriminator():
    req_data = request.get_json()
    path = req_data['path']
    device_cpu = torch.device('cpu')
    try:
        netD_cpu = necessary_elements['discriminator'].to(device_cpu)
        torch.save(netD_cpu.state_dict(), path)
        summary = str(netD_cpu)
        return jsonify(
                message = 'trained discriminator model saved successfully',
                model= summary
            )
    except:
            return jsonify(
                    response= 'failed to save trained discriminator',
                    status=500
                )

#endpoint used to configure the Loss function for the Discriminator:
@app.route('/discriminatorloss', methods=['POST'])
def discriminator_loss():
    if request.method=='POST':
        req_data = request.get_json()
        #loss function:
        in_discriminator_loss = req_data['loss']
        #whether the loss is used for the first time or updated:
        loss_fn_state = req_data['state']
        discriminator_loss_function = loss_function(in_discriminator_loss)
        if (discriminator_loss_function != None):
            necessary_elements['discriminator_loss_function'] = discriminator_loss_function
            return jsonify(
                response= 'discriminator loss function ' + str(loss_fn_state) + ' successfully',
                loss_function= str(discriminator_loss_function),
                mimetype='application/json'
            )
        else:
            return jsonify(
                response= 'request failed',
                status=500,
                hint= 'try BCE or MSE or refer to documentation'
                )

#endpoint to configure the Optimizer for the Discriminator:
@app.route('/discriminatoroptimizer', methods=['POST'])
def discriminator_optimizer():
    if (request.method == 'POST'):
        req_data = request.get_json()
        optimizer_name = req_data['name'] #possibilities: Adam, RMP_Prop, SGD
        learning_rate = req_data['learning_rate']
        #in case of Adam, user can chose values for Beta1 & Beta2
        beta1 = req_data['beta1']
        beta2 = req_data['beta2']
        #whether the optimizer is used for the first time or is being updated:
        optimizer_state = req_data['state']
        #advanced configuration of the optimizer:
        epsilon = req_data['epsilon']
        weight_decay = req_data['weight_decay']
        ams_grad = req_data['ams_grad']
        momentum = req_data['momentum']
        alpha = req_data['alpha']
        centered_rms = req_data['centered_rms']
        nosterov = req_data['nosterov']
        optimizerD = model_optimizer(optimizer_name, necessary_elements['discriminator'], learning_rate, (beta1, beta2),
                                     epsilon, weight_decay, ams_grad, momentum, alpha, centered_rms, nosterov)
        if (optimizerD != None):
            necessary_elements['optimizerD'] = optimizerD
            return jsonify(
                response= 'discriminator optimizer ' + str(optimizer_state) + ' successfully',
                optimizer= str(optimizerD),
                mimetype='application/json'
            )
        else:
            return jsonify(
                response= 'request failed',
                status=500,
                hint= 'try SGD or Adam or RMS or refer to documentation'
                )

# IMPORTANT: must instantiate this before latent vector !!
#endpoint to configure the Generator
@app.route('/generator', methods=['POST'])
def instantiate_generator():
    if request.method =='POST':
            req_data = request.get_json()
            model_generator = req_data['type'] #type of the network (ANN, CNN or Wasserstein)
            LVmodel_generator['model_generator'] = model_generator
            drop_out = req_data['drop_out']
            leaky_relu = req_data['leaky_relu']
            n_layers = req_data['n_layers']
            #size of the Latent Vector (generally it is 120)
            channels_generator = req_data['input_channels']
            LVchannelsG['channels_generator'] = channels_generator
            batch_norm_array = req_data['batch_norm']
            eps_array = req_data['eps']
            momentum_array = req_data['momentum']
            #in cased of ANN-based network, this will contain the number of units per hidden layer:
            layers = req_data['layers']
            deviceGenerator = req_data['device']
            necessary_elements['initG'] = req_data['init']
            if (model_generator == 'DCGAN'):
                DCgenerator = DCGenerator(
                    channels_generator, leaky_relu, drop_out, n_layers,
                    batch_norm_array, eps_array, momentum_array,
                    necessary_elements['channels'], necessary_elements['img_size']).to(torch.device(deviceGenerator))
                necessary_elements['generator'] = DCgenerator

            elif (model_generator == 'VGAN'):
                Vgenerator = VGenerator(
                    layers, channels_generator, necessary_elements['channels'] * necessary_elements['img_size'] * necessary_elements['img_size'],
                    leaky_relu, drop_out, n_layers).to(torch.device(deviceGenerator))
                necessary_elements['generator'] = Vgenerator

            elif (model_generator == 'WGAN'):
                Wgenerator = WGenerator(
                    layers, channels_generator, necessary_elements['channels'] * necessary_elements['img_size'] * necessary_elements['img_size'],
                    leaky_relu, drop_out, n_layers, batch_norm_array, eps_array, momentum_array
                )
                necessary_elements['generator'] = Wgenerator

            init_model(necessary_elements['generator'], necessary_elements['initG'])
            if (necessary_elements['generator'] != None):
                summary = str(necessary_elements['generator'])
                necessary_elements['deviceGenerator'] = deviceGenerator

                initDescription = get_init_method_description(necessary_elements['initG'])
                return jsonify(
                        message = 'generator model created successfully',
                        model= summary,
                        device= necessary_elements['deviceGenerator'],
                        init= necessary_elements['initG'],
                        initDescription= initDescription
                    )

            else:
                return jsonify(
                    dumps= json.dumps(necessary_elements['generator']),
                    response= 'wrong model type',
                    status=500,
                    hint= 'try VGAN/DCGAN or WGAN or refer to documentation'
                )

#endpoint to import trained Generator weights/parameters
#NOTE THAT: user has to provide correct configuration of the Generator in order to import the trained model
@app.route('/trainedgenerator', methods=['POST'])
def trained_generator():
    if request.method=='POST':
        req_data = request.get_json()
        path = req_data['path']
        try:
            necessary_elements['generator'].load_state_dict(torch.load(path))
            necessary_elements['generator'].to(torch.device(necessary_elements['deviceGenerator']))
            summary = str(necessary_elements['generator'])
            return jsonify(
                message = 'trained generator model imported successfully',
                model= summary,
                device= necessary_elements['deviceGenerator']
            )
        except:
            return jsonify(
                    response= 'wrong configuration',
                    status=500
                )

#endpoint to save the trained Generator
@app.route('/savegenerator', methods=['POST'])
def save_generator():
    req_data = request.get_json()
    #path of the file which will conatin the trained model (generally this file is .pth)
    path = req_data['path']
    device_cpu = torch.device('cpu')
    try:
        netG_cpu = necessary_elements['generator'].to(device_cpu)
        torch.save(netG_cpu.state_dict(), path)
        summary = str(netG_cpu)
        return jsonify(
                message = 'trained generator model saved successfully',
                model= summary
            )
    except:
            return jsonify(
                    response= 'failed to save trained Generator',
                    status=500
                )

#enpoint to configure the Loss function for the Generator:
@app.route('/generatorloss', methods=['POST'])
def generator_loss():
    if request.method=='POST':
        req_data = request.get_json()
        in_generator_loss = req_data['loss']
        loss_fn_state = req_data['state']
        generator_loss_function = loss_function(in_generator_loss)
        if (generator_loss_function != None):
            necessary_elements['generator_loss_function'] = generator_loss_function
            return jsonify(
                response= 'generator loss function ' + str(loss_fn_state) + ' successfully',
                loss_function= str(generator_loss_function),
                mimetype='application/json'
            )
        else:
            return jsonify(
                response= 'request failed',
                status=500,
                hint= 'try BCE or MSE or refer to documentation'
                )

#endpoint to configure the Generator Optimizer:
@app.route('/generatoroptimizer', methods=['POST'])
def generator_optimizer():
    if (request.method == 'POST'):
        req_data = request.get_json()
        optimizer_name = req_data['name'] #Adam, RMS or SGD
        learning_rate = req_data['learning_rate']
        #in case of Adam, user can chose Beta values:
        beta1 = req_data['beta1']
        beta2 = req_data['beta2']
        optimizer_state = req_data['state']
        #advanced parameters for the Optimizer:
        epsilon = req_data['epsilon']
        weight_decay = req_data['weight_decay']
        ams_grad = req_data['ams_grad']
        momentum = req_data['momentum']
        alpha = req_data['alpha']
        centered_rms = req_data['centered_rms']
        nosterov = req_data['nosterov']
        optimizerG = model_optimizer(optimizer_name, necessary_elements['generator'], learning_rate, (beta1, beta2),
                                     epsilon, weight_decay, ams_grad, momentum, alpha, centered_rms, nosterov)
        if (optimizerG != None):
            necessary_elements['optimizerG'] = optimizerG
            return jsonify(
                response= 'generator optimizer ' + str(optimizer_state) + ' successfully',
                optimizer= str(optimizerG),
                mimetype='application/json'
            )
        else:
            return jsonify(
                response= 'request failed',
                status=500,
                hint= 'try SGD or Adam or RMS or refer to documentation'
                )

#endpoint to configure the Latent Vector, which will be fed to Generator:
@app.route('/latentvector', methods=['POST'])
def create_latent_vector():
    if request.method=='POST':
            req_data = request.get_json()
            # this will be equal to batch_size:
            in_size = necessary_elements['batch_size']
            latent_vector_state = req_data['state']
            noise_type = req_data['noise_type'] #Unform/ Gaussian or Multi-modal Gaussian (recommended is Gaussian)
            # latent vector depends on generator model/architecture
            if (LVmodel_generator['model_generator'] == 'DCGAN'):
                DClatentVec = DCLatentVector(in_size, LVchannelsG['channels_generator'], noise_type, necessary_elements['deviceGenerator'])
                if(len(DClatentVec.size()) != 0):
                    necessary_elements['latentvector'] = DClatentVec
                    return jsonify(
                            message = 'latent Vector ' + str(latent_vector_state) + ' successfully',
                            shape = DClatentVec.shape,
                            type = noise_type,
                            in_size = in_size, 
                            out_size = LVchannelsG['channels_generator'],
                            device = necessary_elements['deviceGenerator']
                    
                        )
                else:
                    return app.response_class(
                            response = json.dumps(DClatentVec),
                            status = 400,
                            mimetype='application/json'
                        )
            
            elif (LVmodel_generator['model_generator'] == 'VGAN' or LVmodel_generator['model_generator'] == 'WGAN'):
                VlatentVec = VLatentVector(in_size, LVchannelsG['channels_generator'], noise_type, necessary_elements['deviceGenerator'])
                if(len(VlatentVec.size()) != 0):
                    necessary_elements['latentvector'] = VlatentVec
                    return jsonify(
                            message = 'latent Vector ' + str(latent_vector_state) + ' successfully',
                            shape = VlatentVec.shape,
                            type = noise_type,
                            in_size = in_size, 
                            out_size = LVchannelsG['channels_generator'],
                            device = necessary_elements['deviceGenerator']
                    
                        )
                else:
                    return app.response_class(
                            response = json.dumps(VlatentVec),
                            status = 400,
                            mimetype='application/json'
                        )

            else:
                return jsonify(
                    response= 'wrong model type',
                    status=500,
                    hint= 'try Gaussian/Uniform Latent Vector or refer to documentation'
                )

#endpoint to configure the training tricks:
@app.route('/traintricks', methods=['POST'])
def tricks():
    if request.method=='POST':
        req_data = request.get_json()
        necessary_elements['flip'] = req_data['flip'] #flip labels (real => 0, fake =>1)
        necessary_elements['smooth'] = req_data['smooth'] #smooth the labels (exp: real : values between 0.9-1)
        #gradient Penalty (still experimental)
        necessary_elements['apply_gp'] = req_data['apply_gp']
        necessary_elements['lambda_gp'] = req_data['lambda_gp']
        #occasional flip to add some noise (better than full-flip)
        necessary_elements['apply_occasional_flip'] = req_data['apply_occasional_flip']
        necessary_elements['clip_d'] = req_data['clip_d']
        necessary_elements['apply_clip_d'] = req_data['apply_clip_d']
        if(necessary_elements['apply_occasional_flip']):
            necessary_elements['occasional_flip'] = int(req_data['occasional_flip'])
        #use feature matching:
        necessary_elements['apply_feature_matching'] = req_data['feature_matching']
        necessary_elements['apply_divide_d_cost'] = req_data['apply_divide_d_cost']
        return jsonify(
            message= 'training tricks have been added',
            status= 200,
            mimetype='application/json')

#used to stop the training iteration:
@app.route('/stoptraining', methods=['POST'])
def stopTraining():
    necessary_elements['training'] = False
    return jsonify(
            message= 'training has been interrupted',
            status= 200,
            mimetype='application/json')

def vector_to_image(vec, viz):
    return vec.view(viz,
    necessary_elements['channels'],
    necessary_elements['img_size'],
    necessary_elements['img_size'])

#endpoint to train the competing networks:
@app.route('/train', methods=['POST'])
def trainGAN():
    req_data = request.get_json()
    size_viz = req_data['visualization_size']
    nb_batches = req_data['nb_batches']
    train_more_Discriminator = req_data['train_more_Discriminator']
    train_more_Generator = req_data['train_more_Generator']
    unrolling_step = req_data['unrolling_step']
    nb_batches = nb_batches
    necessary_elements['flip'] = necessary_elements['flip'] if (necessary_elements['flip']) else False
    necessary_elements['smooth'] = necessary_elements['smooth'] if (necessary_elements['smooth']) else False
    D_Loss_real_min = []
    D_Loss_real_mean = []
    D_Loss_real_max = []
    D_Loss_fake_min = []
    D_Loss_fake_mean = []
    D_Loss_fake_max = []
    fake_generated = []
    track_convergence_DS = []
    Precision = []
    Recall = []
    F1_score = []
    #Inception_Score = []
    generated_bytes = None
    real_bytes = None
    necessary_elements['discriminator'].to(torch.device(necessary_elements['deviceDiscriminator']))
    necessary_elements['generator'].to(torch.device(necessary_elements['deviceGenerator']))
    necessary_elements['training'] = True
    best_KL = -30
    worst_KL = 30
    start_time = time.time()
    for i in range(nb_batches):
        if (unrolling_step != 0):
            unrollingIterator = iter(necessary_elements['dataloader'])
            for tmpIndex in range(unrolling_step):
                unrollingIterItem = next(unrollingIterator)
                if(len(unrollingIterItem) == 1):
                    unrolling_real_batch_element = unrollingIterItem[0]
                else: unrolling_real_batch_element,_ = unrollingIterItem
                if (necessary_elements['model_discriminator'] != 'DCGAN'):
                    unrolling_real_batch_element = unrolling_real_batch_element.view(unrolling_real_batch_element.shape[0], necessary_elements['channels']*necessary_elements['img_size']*necessary_elements['img_size'])
                else: unrolling_real_batch_element = unrolling_real_batch_element.view(unrolling_real_batch_element.shape[0], necessary_elements['channels'], necessary_elements['img_size'], necessary_elements['img_size'])
                unrolling_real_batch.append(unrolling_real_batch_element)

        if (necessary_elements['index_batch'] == 0) or (necessary_elements['index_batch'] == len(necessary_elements['dataloader'])-1):
            necessary_elements['index_batch'] = 0
            necessary_elements['loaderIteraor'] = iter(necessary_elements['dataloader'])
            necessary_elements['epoch_number'] = necessary_elements['epoch_number'] + 1
        iterItem = next(necessary_elements['loaderIteraor'])
        necessary_elements['index_batch'] = necessary_elements['index_batch'] + 1
        if(len(iterItem) == 1):
            real_batch = iterItem[0]
        else: real_batch,_ = iterItem
        if (necessary_elements['apply_occasional_flip'] and necessary_elements['index_batch'] % necessary_elements['occasional_flip'] == 0):
            necessary_elements['flip'] = True
        # train Discriminator:
        fake_batch = necessary_elements['generator'](necessary_elements['latentvector']).detach()
        real_batch, fake_batch = reshape(necessary_elements['model_discriminator'],real_batch, fake_batch,
                                        necessary_elements['batch_size'], necessary_elements['img_size'], necessary_elements['channels'])
        real_batch = real_batch.to(necessary_elements['deviceDiscriminator'])
        fake_batch = fake_batch.to(necessary_elements['deviceDiscriminator'])
        d_error, d_real, d_fake, gradient_penalty = train_discriminator(necessary_elements['optimizerD'], real_batch,
                                                    fake_batch, necessary_elements['discriminator'], 
                                                    necessary_elements['discriminator_loss_function'],
                                                    necessary_elements['deviceDiscriminator'],
                                                    necessary_elements['flip'], necessary_elements['smooth'],
                                                    necessary_elements['symmetric_labels'],
                                                    necessary_elements['apply_gp'], necessary_elements['lambda_gp'],
                                                    LVmodel_generator['model_generator'], necessary_elements['clip_d'],
                                                    necessary_elements['apply_clip_d'], necessary_elements['apply_divide_d_cost'])
        D_Loss.append(d_error.item())
        d_real_squeezed = torch.squeeze(d_real)
        d_fake_squeezed = torch.squeeze(d_fake)
        d_real_squeezed = d_real_squeezed.data.cpu()
        d_fake_squeezed = d_fake_squeezed.data.cpu()
        D_Loss_real_mean.append(torch.mean(d_real_squeezed).item())
        D_Loss_real_min.append(torch.min(d_real_squeezed).item())
        D_Loss_real_max.append(torch.max(d_real_squeezed).item())
        D_Loss_fake_mean.append(torch.mean(d_fake_squeezed).item())
        D_Loss_fake_min.append(torch.min(d_fake_squeezed).item())
        D_Loss_fake_max.append(torch.max(d_fake_squeezed).item())
        calc_metrics(d_real_squeezed, d_fake_squeezed, Precision, Recall, F1_score)
        #when we want to train D more:
        if (train_more_Discriminator > 0):
            for index in range(train_more_Discriminator):
                fake_batch = necessary_elements['generator'](necessary_elements['latentvector']).detach()
                _, fake_batch = reshape(necessary_elements['model_discriminator'],real_batch, fake_batch,
                                            necessary_elements['batch_size'], necessary_elements['img_size'], necessary_elements['channels'])
                fake_batch = fake_batch.to(necessary_elements['deviceDiscriminator'])
                d_error, d_real, d_fake, gradient_penalty = train_discriminator(necessary_elements['optimizerD'], real_batch,
                                                            fake_batch, necessary_elements['discriminator'], 
                                                            necessary_elements['discriminator_loss_function'],
                                                            necessary_elements['deviceDiscriminator'],
                                                            necessary_elements['flip'], necessary_elements['smooth'],
                                                            necessary_elements['symmetric_labels'],
                                                            necessary_elements['apply_gp'], necessary_elements['lambda_gp'],
                                                            LVmodel_generator['model_generator'], necessary_elements['clip_d'],
                                                            necessary_elements['apply_clip_d'], necessary_elements['apply_divide_d_cost'])
                D_Loss.append(d_error.item())
                d_real_squeezed = torch.squeeze(d_real)
                d_fake_squeezed = torch.squeeze(d_fake)
                d_real_squeezed = d_real_squeezed.data.cpu()
                d_fake_squeezed = d_fake_squeezed.data.cpu()
                D_Loss_real_mean.append(torch.mean(d_real_squeezed).item())
                D_Loss_real_min.append(torch.min(d_real_squeezed).item())
                D_Loss_real_max.append(torch.max(d_real_squeezed).item())
                D_Loss_fake_mean.append(torch.mean(d_fake_squeezed).item())
                D_Loss_fake_min.append(torch.min(d_fake_squeezed).item())
                D_Loss_fake_max.append(torch.max(d_fake_squeezed).item())                                           
        
        #train Generator:
        fake_generated = necessary_elements['generator'](necessary_elements['latentvector'])
        _, fake_generated = reshape(necessary_elements['model_discriminator'],real_batch, fake_generated,
                                        necessary_elements['batch_size'], necessary_elements['img_size'], necessary_elements['channels'])
        g_error = train_generator(necessary_elements['optimizerG'], fake_generated, 
                                necessary_elements['discriminator'], necessary_elements['generator_loss_function'], 
                                necessary_elements['deviceGenerator'], necessary_elements['deviceDiscriminator'],
                                necessary_elements['flip'], necessary_elements['smooth'],
                                necessary_elements['symmetric_labels'],
                                unrolling_step, necessary_elements['optimizerD'],
                                unrolling_real_batch, necessary_elements['discriminator_loss_function'],
                                real_batch, necessary_elements['apply_feature_matching'],
                                necessary_elements['batch_size'])
        G_Loss.append(g_error.item())
        kl_div_item = (F.kl_div(fake_generated, real_batch)).item()
        KL_div.append(kl_div_item)
        if (kl_div_item > best_KL):
            best_KL = kl_div_item
            best_generated = fake_generated
        if (kl_div_item < worst_KL):
            worst_KL = kl_div_item
            worst_generated = fake_generated
        tmpJS = 0.5 * (real_batch + fake_generated)
        JS_div.append((0.5*(F.kl_div(real_batch, tmpJS) + F.kl_div(fake_generated, tmpJS))).item())

        #when we want to train G more:
        if (train_more_Generator > 0):
            for index in range(train_more_Generator):
                fake_generated = necessary_elements['generator'](necessary_elements['latentvector'])
                _, fake_batch = reshape(necessary_elements['model_discriminator'],real_batch, fake_batch,
                                            necessary_elements['batch_size'], necessary_elements['img_size'], necessary_elements['channels'])
                g_error = train_generator(necessary_elements['optimizerG'], fake_generated, 
                                        necessary_elements['discriminator'], necessary_elements['generator_loss_function'], 
                                        necessary_elements['deviceGenerator'], necessary_elements['deviceDiscriminator'],
                                        necessary_elements['flip'], necessary_elements['smooth'],
                                        necessary_elements['symmetric_labels'],
                                        unrolling_step, necessary_elements['optimizerD'],
                                        unrolling_real_batch, necessary_elements['discriminator_loss_function'],
                                        real_batch, necessary_elements['apply_feature_matching'],
                                        necessary_elements['batch_size'])
                G_Loss.append(g_error.item())
                KL_div.append((F.kl_div(real_batch, fake_generated)).item())
                tmpJS = 0.5 * (real_batch + fake_generated)
                JS_div.append((0.5*(F.kl_div(real_batch, tmpJS) + F.kl_div(fake_generated, tmpJS))).item())
        if (necessary_elements['apply_occasional_flip'] and necessary_elements['index_batch'] % necessary_elements['occasional_flip'] == 0):
            necessary_elements['flip'] = False
        track_convergence_DS.extend([(F.kl_div(real_batch, fake_generated)).item(),
                                    g_error.item(), d_error.item(),
                                    torch.min(d_real_squeezed).item(), torch.max(d_real_squeezed).item(),
                                    torch.min(d_fake_squeezed).item(), torch.max(d_fake_squeezed).item()])
    # loop done:
    result_elements['d_error'] = d_error.tolist()
    result_elements['d_real'] = d_real.tolist()
    result_elements['d_fake'] = d_fake.tolist()
    # 2D data:
    best_js_sample = 0
    fake_sample = worst_generated[0]
    real_sample = real_batch[0]
    for real_sample_item in real_batch :
        tmpJS_sample = 0.5 * (real_sample_item + fake_sample)
        js_div_item_sample = (0.5*(F.kl_div(real_sample_item, tmpJS_sample) + F.kl_div(fake_sample, tmpJS_sample))).item()
        if (js_div_item_sample > best_js_sample):
            best_js_sample = js_div_item_sample
            real_sample = real_sample_item

    real_sample, fake_sample = rgb_to_gray(real_sample, fake_sample, necessary_elements['img_size'], necessary_elements['channels'])

    result_elements['pca_real2D'] = pca_real2D.fit_transform(real_sample.data.cpu())
    result_elements['pca_generated2D'] = pca_generated2D.fit_transform(fake_sample.data.cpu())
    # 3D data:
    result_elements['pca_real3D'] = pca_real3D.fit_transform(real_sample.data.cpu())
    result_elements['pca_generated3D'] = pca_generated3D.fit_transform(fake_sample.data.cpu())
    
    result_elements['g_error'] = g_error.tolist()
    tmpFakeGenerated = best_generated[:size_viz * size_viz]
    tmpWorstGenerated = worst_generated[:size_viz * size_viz]
    tmpFakeGenerated = vector_to_image(tmpFakeGenerated, size_viz * size_viz).data.cpu()
    tmpWorstGenerated = vector_to_image(tmpWorstGenerated, size_viz * size_viz).data.cpu()
    if type(tmpFakeGenerated) == np.ndarray:
        tmpFakeGenerated = torch.from_numpy(tmpFakeGenerated)
    if type(tmpWorstGenerated) == np.ndarray:
        tmpWorstGenerated = torch.from_numpy(tmpWorstGenerated)
    gridGenerated = vutils.make_grid(tmpFakeGenerated, nrow=size_viz, normalize=True, scale_each=True)
    gridWorstGenerated = vutils.make_grid(tmpWorstGenerated, nrow=size_viz, normalize=True, scale_each=True)
    tmpRealImages = real_batch[:size_viz * size_viz]
    tmpRealImages = vector_to_image(tmpRealImages, size_viz * size_viz).data.cpu()
    if type(tmpRealImages) == np.ndarray:
        tmpRealImages = torch.from_numpy(tmpRealImages)
    gridReal = vutils.make_grid(tmpRealImages, nrow=size_viz, normalize=True, scale_each=True)  
    tmpFakeGenerated = np.transpose(gridGenerated.cpu(), (1, 2, 0))
    strIO = BytesIO()
    imsave(strIO, tmpFakeGenerated, plugin='pil', format_str='png')
    strIO.seek(0)
    generated_bytes = base64.b64encode(strIO.getvalue())

    tmpWorstGenerated = np.transpose(gridWorstGenerated.cpu(), (1, 2, 0))
    strIO = BytesIO()
    imsave(strIO, tmpWorstGenerated, plugin='pil', format_str='png')
    strIO.seek(0)
    worst_generated_bytes = base64.b64encode(strIO.getvalue())

    tmpRealImages = np.transpose(gridReal.cpu(), (1, 2, 0))
    strIO = BytesIO()
    imsave(strIO, tmpRealImages, plugin='pil', format_str='png')
    strIO.seek(0)
    real_bytes = base64.b64encode(strIO.getvalue())
    end_time = time.time()
    elapsed_time = end_time - start_time

    if (result_elements != None):
        return jsonify(
            d_error= D_Loss[(len(D_Loss) - 40):],
            g_error= G_Loss[(len(G_Loss) - 40):],
            size_generated_images= fake_generated[:size_viz * size_viz].shape,
            d_Loss_real_min= D_Loss_real_min,
            d_Loss_real_mean= D_Loss_real_mean,
            d_Loss_real_max= D_Loss_real_max,
            d_Loss_fake_min= D_Loss_fake_min,
            d_Loss_fake_mean= D_Loss_fake_mean,
            d_Loss_fake_max= D_Loss_fake_max,
            generated_bytes= generated_bytes.decode('ascii'),
            worst_generated_bytes= worst_generated_bytes.decode('ascii'),
            real_bytes= real_bytes.decode('ascii'),
            real_2d= [result_elements['pca_real2D'][:, 0].tolist() , result_elements['pca_real2D'][:, 1].tolist()],
            fake_2d= [result_elements['pca_generated2D'][:, 0].tolist() , result_elements['pca_generated2D'][:, 1].tolist() ],
            real_3d= [result_elements['pca_real3D'][:, 0].tolist() , result_elements['pca_real3D'][:, 1].tolist(), result_elements['pca_real3D'][:, 2].tolist()],
            fake_3d= [result_elements['pca_generated3D'][:, 0].tolist() , result_elements['pca_generated3D'][:, 1].tolist(), result_elements['pca_generated3D'][:, 2].tolist()],
            kl_div= KL_div[(len(KL_div) - 40):],
            js_div= JS_div[(len(JS_div) - 40):],
            precision= Precision,
            recall= Recall,
            f1_score= F1_score,
            training= necessary_elements['training'],
            track_convergence_DS= track_convergence_DS,
            elapsed_time= elapsed_time,
            index_batch= necessary_elements['index_batch'],
            epoch_number= necessary_elements['epoch_number'],
            status= 200,
            mimetype='application/json'
        )
    else:
        return app.response_class(
                response = json.dumps(result_elements),
                status = 400,
                mimetype='application/json'
                    )

#endpoint to reset all the training session:
@app.route('/reset', methods=['POST'])
def reset():
    req_data = request.get_json()
    reset_train = req_data['reset']
    if (reset_train == 'reset'):
        necessary_elements = {}
        necessary_elements['index_batch'] = 0
        necessary_elements['epoch_number'] = 0
        return jsonify(
            message= 'train session has been reset',
            train_status= necessary_elements,
            status= 200,
            mimetype='application/json')

#if __name__ == '__main__':
    #app.run(host='127.0.0.1')