#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzh
"""
import os
from src.model import gcn_model, disentangle_model_vae_id, disentangle_model_vae_exp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', action='store', dest="gpu", default='0',
                    help="gpu id")
parser.add_argument('-e', action='store', dest="epoch", default=2,
                    help="training epoch")
parser.add_argument('-lr', action='store', dest="l_rate", default=0.0001,
                    help="learning rate")
parser.add_argument('-m', action='store', dest="mode", default= 'gcn_vae_exp',
                    help="training mode")
parser.add_argument('-l', action='store_true', dest="load",
                    help="load pretrained model")
parser.add_argument('-t', action='store_false', dest="train",
                    help="use -t to switch to testing step")
parser.add_argument('-s', action='store', dest="suffix",default='',
                    help="suffix of filename")
parser.add_argument('-p', action='store', dest="test_people_id",default=142,
                    help="id of test people")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
epoch = int(args.epoch)
l_rate = float(args.l_rate)
load = args.load
train = args.train
mode = args.mode
suffix = args.suffix
people_id = int(args.test_people_id)
# filename prefix
prefix = mode
if not train:
    try:
        os.makedirs('data/mesh')
    except:
        pass
#if not train:
if True:
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    sess = tf.Session(config=config) 
    KTF.set_session(sess)
    
if people_id>140:
    filename = 'test'
else:
    filename = 'train'
    
print('Config: Load pretrained<-{}, suffix<-{}, epoch<-{}, lr<-{}'.format(load, suffix, epoch, l_rate))

        
if mode == 'fusion_dr':
    
    # dim of input vector
    input_dim = 11510*9
    # dim of per feature
    feature_dim = 9
    latent_dim_id = 50
    latent_dim_exp = 25
    #25,5#75,50
    prefix = 'disentangle'
    print('Loading {} model'.format(mode))
    if suffix == '':
        suffix = 'fusion_dr'
        #suffix = 'fusion7550'
        #suffix = 'fusion_no'
    # VAE network
    
    if train:
        net = gcn_model(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2)
        #suffix = '50'
        suffix = 'gcn_vae_exp'
        exp_net = disentangle_model_vae_exp(input_dim, prefix, suffix, l_rate, True, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2, latent_dim_exp = latent_dim_exp, kl_weight = 0.00001)

        suffix = 'gcn_vae_id'
        id_net = disentangle_model_vae_id(input_dim, prefix, suffix, l_rate, True, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2, latent_dim_id = latent_dim_id,kl_weight = 0.00001)

        #net.train_fusion(id_net, exp_net, epoch)
        net.end_to_end(id_net, exp_net, epoch)
    else:
        net = gcn_model(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2)
        suffix = 'gcn_vae_exp'
        exp_net = disentangle_model_vae_exp(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2, latent_dim_exp = latent_dim_exp)

        suffix = 'gcn_vae_id'
        id_net = disentangle_model_vae_id(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2, latent_dim_id = latent_dim_id)

        #net.test_whole(id_net, exp_net,people_id=people_id)
        #net.test_interpolation(id_net, exp_net)
        net.test_change(id_net, exp_net)


if mode == 'gcn_vae_id':
    
    # dim of input vector
    input_dim = 11510*9
    # dim of per feature
    feature_dim = 9
    latent_dim_id = 50
    prefix = 'disentangle'
    print('Loading {} model'.format(mode))
    if suffix == '':
        suffix = 'gcn_vae_id'
    # AE network
    
    if train:
        net = disentangle_model_vae_id(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2, kl_weight = 0.00001,latent_dim_id = latent_dim_id)
        #net.special_train(epoch)
        net.train(epoch)
        
        
    else:
        net = disentangle_model_vae_id(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2,latent_dim_id = latent_dim_id)
        net.test(people_id= people_id, filename=filename)
        #net.test_whole()
        #net.code_bp(epoch)       

        
        
if mode == 'gcn_vae_exp':
    
    # dim of input vector
    input_dim = 11510*9
    # dim of per feature
    feature_dim = 9
    latent_dim_exp = 25
    prefix = 'disentangle'
    print('Loading {} model'.format(mode))
    if suffix == '':
        suffix = 'gcn_vae_exp'
    # AE network
    
    if train:
        net = disentangle_model_vae_exp(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2, kl_weight = 0.00001, latent_dim_exp = latent_dim_exp)
        net.train(epoch)
    else:
        net = disentangle_model_vae_exp(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2, latent_dim_exp = latent_dim_exp)
        net.test()
#        net.test(people_id= people_id, filename=filename)
#        net.test_whole()
#        suffix = 'gcn_vae_id'
#        suffix = 'special'
#        id_net = disentangle_model_vae_id(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2)
#        suffix = 'fusion_dr'
#        fusion_net = gcn_model(input_dim, prefix, suffix, l_rate, load, feature_dim = feature_dim, batch_size=1, MAX_DEGREE=2)
#        net.test_fusion(id_net)
#        net.test_change(id_net)


if train:
    import matplotlib.pyplot as plt
    import numpy as np
    log = np.load('log.npy')
    test_log = np.load('testlog.npy')

    plt.switch_backend('agg')
    plt.plot(log)
    plt.plot(test_log)
    plt.savefig('loss')
