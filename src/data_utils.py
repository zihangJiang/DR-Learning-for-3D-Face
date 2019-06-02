# -*- coding: utf-8 -*-
"""
@author: jzh
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
import openmesh as om
import scipy.sparse as sp
from tqdm import tqdm
# Preprocessing config
change_length = 11510
#change_length = 3931
unit = 9
delta = np.array([1,0,0,1,0,1,0,0,0])
cross_id = np.tile(delta, change_length)
epsilon = 1e-4
def batch_change(data):
    '''
    to change the data by divide it to some 'unit'
    and reduce 'delta' in each unit
    the whole change step will take 'change_length' steps
    meaning only use data[: change_length * unit] 
    '''
    dim = data.shape[0]
    length = int(dim/9)
    cross = np.tile(delta, length)
    return data - cross

def data_interpolation(data_array, alpha = 0.5):
    '''
    interpolate data
    'alpha' is the constant of interpolation
    '''
    num,dim = data_array.shape
    whole_data = np.zeros((int(num*(num-1)/2 + num), dim))
    k = 0
    print((num,dim,int(num*(num-1)/2)))
    for i in range(num):
        print(i)
        for j in range(i,num):
            whole_data[k] = alpha * data_array[i] + (1-alpha) * data_array[j]
            k = k+1
    return whole_data

def load_data_fromfile(filename, M_list, m_list, **kwargs):
     data = batch_change(np.fromfile(filename),change_length,unit,delta)
     if kwargs=={}:
         data = normalize_fromfile(data[np.newaxis,:], M_list, m_list)
     else:
         data = normalize_fromfile(data[kwargs['filter_data']][np.newaxis,:], M_list, m_list)
     
     return data



def data_recover(data):
    '''
    recover data from minused data
    '''
    return data + cross_id
    

def normalize(data, a = 0.9 ,epsilon = 10e-6):
    num,dim = data.shape
    for i in range(dim):
        M = np.max(data[:,i])
        m = np.min(data[:,i])
        if M-m < epsilon:
            M = M+epsilon
            m = m-epsilon
            
        data[:,i] = 2 * a * (data[:,i]-m) / (M-m) -a
    return data

def save_normalize_list(array, data, suffix = 'disentangle', a = 0.9 ,epsilon = 10e-6):
    num,dim = data.shape
    M_list = np.zeros_like(array)
    m_list = np.zeros_like(array)
    for i in range(dim):
        M = np.max(data[:,i])
        m = np.min(data[:,i])
        if M-m < epsilon:
            M = M+epsilon
            m = m-epsilon
        M_list[i] = M
        m_list[i] = m
    np.save('../data/{}/max_data'.format(suffix),M_list)
    np.save('../data/{}/min_data'.format(suffix),m_list)
    return array
    
def denormalize_fromfile(array, M_list, m_list, a = 0.9):
    num,dim = array.shape
    for i in range(dim):
        M = M_list[i]
        m = m_list[i]
        array[:, i] = (array[:,i]+a)*(M-m)/(2*a)+m
    return array

def normalize_fromfile(array, M_list, m_list, a = 0.9):
    num,dim = array.shape
    for i in range(dim):
        M = M_list[i]
        m = m_list[i]
        array[:, i] = 2 * a * (array[:,i]-m) / (M-m) -a
    return array

def deduce_mean(tup, data_array):
    data = data_array.copy()
    for i in range(len(tup)-1):
        data[tup[i]:tup[i+1]] -= np.mean(data[tup[i]:tup[i+1]], axis = 0)
        
    return data
def draw(color, tup, embed):
    n = min(len(color), len(tup)-1)
    plt.savefig('pict.png')
    for i in range(n):
        plt.plot(embed[tup[i]:tup[i+1],0],embed[tup[i]:tup[i+1],1],color[i])
    
    plt.savefig('draw_embed')
    #plt.show()

def reduce_normalize_list(dir_name):
    epsilon = 10e-6
    max_data = np.fromfile(os.path.join(dir_name,'max_data.dat')) - cross_id
    min_data = np.fromfile(os.path.join(dir_name,'min_data.dat')) - cross_id
    for i in range(min_data.shape[0]):
        if abs(min_data[i] - max_data[i]) < epsilon:
            min_data[i] -= epsilon
            max_data[i] += epsilon
    np.save(os.path.join(dir_name,'max_data'), max_data)        
    np.save(os.path.join(dir_name,'min_data'), min_data)  

def concate_data(data_path,data_format):
    train_data = []
    test_data = []
    if not os.path.exists('../data/disentangle/train_data'):
        os.mkdir('../data/disentangle/train_data')
    if not os.path.exists('../data/disentangle/test_data'):
        os.mkdir('../data/disentangle/test_data')

    for j in range(1,141):
        print('dealing with Feature{}'.format(j))
        whole_data = np.vstack((np.fromfile(os.path.join(data_path, data_format.format(j, i))) - cross_id for i in range(47)))
        np.save('../data/disentangle/train_data/Feature{}'.format(j),whole_data)
        train_data.append(whole_data)
    np.save('../data/disentangle/train_data',np.vstack(train_data))

    for j in range(141,151):
        print('dealing with Feature{}'.format(j))
        whole_data = np.vstack((np.fromfile(os.path.join(data_path, data_format.format(j, i))) - cross_id for i in range(47)))
        np.save('../data/disentangle/test_data/Feature{}'.format(j),whole_data)
        test_data.append(whole_data)
    np.save('../data/disentangle/test_data',np.vstack(test_data))
    train_data.extend(test_data)
    return np.vstack(train_data)
     
def polar_weights(D, bound = (0.5,1.2)):
    r = np.random.uniform(bound[0], bound[1])
    weights = np.empty(D)
    theta = np.empty(D-1)
    for i in range(D):
        weights[i] = r

    for i in range(D-1):
        theta[i] = np.random.uniform(epsilon,math.pi/2.0)
        weights[i] = weights[i]*math.cos(theta[i])

    for i in range(D):
        for j in range(0,i):
            weights[i] = weights[i]*math.sin(theta[j])
    return weights

def interpolate(data, Dimension, Interpolate_num=1):
    Num = len(data)
    Index = np.random.randint(0,Num,Dimension)
    meta_data = []
    interpolate_data = []
    for i in range(Dimension):
        meta_data.append(data[Index[i]])

    for i in range(Interpolate_num):
        weights = polar_weights(Dimension)
        a=np.zeros(len(data[Index[0]]))
        for d in range(Dimension):
            a = a+meta_data[d]*weights[d]

        interpolate_data.append(a)
    
    return interpolate_data


if __name__ == '__main__':
    data_path = '../data/FWH'
    data_format = 'Tester_{}/face_{}.dat'
    data = concate_data(data_path, data_format)
    save_normalize_list(data[0],data)
    # meanface_data_format = '../data/Meanface/face_{}.dat'
    # meanface_data = np.vstack((np.fromfile(meanface_data_format.format(i)) - cross_id for i in range(47)))
    # np.save('../data/disentangle/MeanFace_data', meanface_data)
    #data_format = 'Tester_{}/shape_{}.obj'

    # train_data = np.load('../data/disentangle/train_data.npy')
    # test_data = np.load('../data/disentangle/test_data.npy')
    # inter_data = np.vstack(np.fromfile('../data/disentangle/Interpolated_results/interpolated_{}.dat'.format(i))-cross_id for i in range(5000))
    # whole_data = np.vstack((train_data, test_data, inter_data))
    # whole_data = np.vstack((train_data, test_data))
    # save_normalize_list(whole_data[0],whole_data)