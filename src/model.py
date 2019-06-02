# -*- coding: utf-8 -*-
"""
@author: jzh
"""
import numpy as np, keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from src.VAE import get_gcn, get_gcn_vae_id, get_gcn_vae_exp
from src.data_utils import normalize_fromfile, denormalize_fromfile, data_recover, batch_change
from src.get_mesh import get_mesh
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from src.mesh import V2M2
ref_name = 'data/disentangle/Mean_Face.obj'

'''
GCN code was inspired by https://github.com/tkipf/keras-gcn
'''
def get_general_laplacian(adj):
    return (sp.diags(np.power(np.array(adj.sum(1)), 1).flatten(), 0) - adj) * sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = (sp.eye(adj.shape[0], dtype=np.float32)) - adj_normalized
    return laplacian

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = (eigsh(laplacian, 1, which='LM', return_eigenvectors=False))[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = 2.0 / largest_eigval * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian

def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print(('Calculating Chebyshev polynomials up to order {}...').format(k))
    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k + 1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    T_k = [i.astype(np.float32) for i in T_k]
    return T_k


        
        
class gcn_dis_model(object):

    def __init__(self, input_dim, prefix, suffix, lr, load, feature_dim=9, latent_dim_id=50, latent_dim_exp=25, kl_weight=0.000005,weight_decay = 0.00001, batch_size=1, MAX_DEGREE=2):
        self.input_dim = input_dim
        self.prefix = prefix
        self.suffix = suffix
        self.load = load
        self.latent_dim_id = latent_dim_id
        self.latent_dim_exp = latent_dim_exp
        self.feature_dim = feature_dim
        self.v = int(input_dim / feature_dim)
        self.hidden_dim = 300
        self.lr = lr
        self.kl_weight = K.variable(kl_weight)
        self.M_list = np.load(('data/{}/max_data.npy').format(self.prefix))
        self.m_list = np.load(('data/{}/min_data.npy').format(self.prefix))
        self.batch_size = batch_size
        self.weight_decay = K.variable(weight_decay)
        self.build_model(MAX_DEGREE)

class disentangle_model_vae_id(gcn_dis_model):

    def build_model(self, MAX_DEGREE):
        SYM_NORM = True
        A = sp.load_npz(('data/{}/FWH_adj_matrix.npz').format(self.prefix))
        L = normalized_laplacian(A, SYM_NORM)
        T_k = chebyshev_polynomial(rescale_laplacian(L), MAX_DEGREE)
        support = MAX_DEGREE + 1
        self.kl_loss, self.encoder, self.decoder, self.gcn_vae_id = get_gcn_vae_id(T_k, support, batch_size=self.batch_size, feature_dim=self.feature_dim, v=self.v, input_dim=self.input_dim, latent_dim = self.latent_dim_id)
        self.neutral_face = Input(shape=(self.input_dim,))
        real = self.gcn_vae_id.get_input_at(0)
        ratio = K.variable(self.M_list - self.m_list)
        if self.feature_dim == 9:
            self.id_loss = K.mean(K.abs((self.neutral_face - self.gcn_vae_id(real)) * ratio))/1.8
        else:
            ori_mesh = K.reshape(((self.neutral_face - self.gcn_vae_id(real)) * ratio), (self.batch_size, -1, 3))
            self.id_loss = K.mean(K.sqrt(K.sum(K.square(ori_mesh) ,axis=-1)))/1.8
            
        
        
        weights = self.gcn_vae_id.trainable_weights#+[self.scalar]
        self.regularization_loss = 0
        for w in weights:
            #print(w)
            if self.feature_dim == 9:
                self.regularization_loss += self.weight_decay*  K.sum(K.square(w))
            else:
                self.regularization_loss += 0.00002*  K.sum(K.square(w))
        self.loss = self.id_loss + self.kl_weight * self.kl_loss + self.regularization_loss
        self.opt = Adam(lr=self.lr)
        training_updates = (self.opt).get_updates(weights, [], self.loss)
        self.train_func = K.function([real, self.neutral_face], [self.id_loss, self.loss, self.kl_loss, self.regularization_loss], training_updates)
        self.test_func = K.function([real, self.neutral_face], [self.id_loss, self.loss, self.kl_loss, self.regularization_loss])
        if self.load:
            self.load_models()

    def save_models(self):
        self.gcn_vae_id.save_weights(('model/gcn_vae_id_model/gcn_vae_id{}{}.h5').format(self.prefix, self.suffix))
        self.encoder.save_weights(('model/gcn_vae_id_model/encoder_id_{}{}.h5').format(self.prefix, self.suffix))
        self.decoder.save_weights(('model/gcn_vae_id_model/decoder_id_{}{}.h5').format(self.prefix, self.suffix))
    def load_models(self):
        self.gcn_vae_id.load_weights(('model/gcn_vae_id_model/gcn_vae_id{}{}.h5').format(self.prefix, self.suffix))

        
    def code_bp(self, epoch):
        #test_array = np.vstack(batch_change(np.fromfile('data/disentangle/real_data/{}.dat'.format(i))) for i in range(287))
        test_array = np.load('data/{}/test_data.npy'.format(self.prefix))[47*np.arange(10)]
        frt = np.loadtxt('src/front_part_v.txt', dtype = int)
        mask = np.zeros(11510)
        mask[frt] = 1
        normalize_fromfile(test_array, self.M_list, self.m_list)
        num = 0
        target_feature = test_array[num:num+1]
        #x = [0,2,6,7,8]
        K.set_learning_phase(0)
        start = self.encoder.predict(target_feature, batch_size = self.batch_size)
        code = K.variable(start[0])

        target_feature_holder = Input(shape=(self.input_dim, ))
        mask = K.variable(np.repeat(mask, 9))

        ratio = K.variable(self.M_list - self.m_list)
        cross_id = K.variable(np.tile(np.array([1,0,0,1,0,1,0,0,0]), 11510))


        target = self.decoder(code)
        loss = K.mean(K.abs(ratio*(target - target_feature_holder)))/1.8
        lr = self.lr
        for circle in range(10):
            training_updates = (Adam(lr=lr)).get_updates([code], [], loss)
            
            bp_func = K.function([target_feature_holder], [loss, target], training_updates)
            for i in range(epoch):
                err, result_mesh = bp_func([target_feature])
                print('Epoch: {}, loss: {}'.format(i,err))
            lr = input('learning rate change?   ')
            lr = float(lr)
            if lr == 0:
                break
        start_norm = self.decoder.predict(start[0],batch_size = self.batch_size)
        start_id = denormalize_fromfile(start_norm, self.M_list, self.m_list)
        result_id = denormalize_fromfile(result_mesh, self.M_list, self.m_list)
        denormalize_fromfile(target_feature, self.M_list, self.m_list)

        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        V2M2(get_mesh(ref_name, data_recover(start_id)), 'data/mesh/start_id.obj')
        V2M2(get_mesh(ref_name, data_recover(result_id)), 'data/mesh/result_id.obj')
        V2M2(get_mesh(ref_name, data_recover(target_feature)), 'data/mesh/target_id.obj')
        
        
    def train(self, epoch):
        def get_interpolate_data(prefix, num = 2000):
            if prefix == 'disentangle':
                #interpolate_data = np.vstack(batch_change(np.fromfile('data/{}/real_data/{}.dat'.format(prefix, i))) for i in range(num))
                interpolate_data = np.vstack(batch_change(np.fromfile('data/{}/Interpolated_results/interpolated_{}.dat'.format(prefix, i))) for i in range(num))
            else:
                interpolate_data = np.vstack(np.fromfile('data/{}/Interpolated_results/interpolated_{}.dat'.format(prefix, i)) for i in range(num))
                mean_inter = np.mean(interpolate_data, axis = 0)
                interpolate_data = interpolate_data - mean_inter
            return interpolate_data
        inter_array = get_interpolate_data(self.prefix, 4000)
        data_array = np.load(('data/{}/train_data.npy').format(self.prefix))
        test_array = np.load(('data/{}/test_data.npy').format(self.prefix))
        mean_exp = np.load(('data/{}/MeanFace_data.npy').format(self.prefix))

        normalize_fromfile(test_array, self.M_list, self.m_list)
        normalize_fromfile(mean_exp, self.M_list, self.m_list)
        normalize_fromfile(data_array, self.M_list, self.m_list)
        normalize_fromfile(inter_array, self.M_list, self.m_list)

        ITS = data_array.shape[0]//self.batch_size
        log = np.zeros((epoch*ITS,))
        test_log = np.zeros((epoch*ITS,))
        constant_list = np.arange(data_array.shape[0])
        inter_list = np.arange(inter_array.shape[0])
        display_step = 50
        for i in range(epoch):

            np.random.shuffle(constant_list)
            np.random.shuffle(inter_list)
            # for index, j in enumerate(constant_list):
            for index, j in enumerate(zip(*[iter(constant_list)]*self.batch_size)):
                # l = np.random.randint(0, 47)
                l = np.random.randint(0,47,self.batch_size)
                inter_sample = np.random.randint(0,inter_array.shape[0],self.batch_size)
                #l = 1
                j = np.array(j)
                C_exp = j % 47
                C_neutral = j - C_exp
                people_with_emotion = data_array[j]
                people_neutral_face = data_array[C_neutral]
                C_int = inter_list[(index*self.batch_size) %inter_array.shape[0]: (index * self.batch_size)%inter_array.shape[0]+self.batch_size]
                inter_people = inter_array[C_int]
                m = np.random.randint(0, 47, inter_people.shape[0])
                inter_people_emotion = inter_people + mean_exp[m] + 0.9*(self.M_list + self.m_list)/(self.M_list - self.m_list)
                K.set_learning_phase(1)
                K.set_value(self.opt.lr, self.lr*10)
                err_re_inter, err_total_inter, err_kl, err_regular = self.train_func([inter_people_emotion, inter_people])
                K.set_value(self.opt.lr, self.lr*0.1)
                err_re_emoti, err_total_emoti, err_kl, err_regular = self.train_func([people_with_emotion, people_neutral_face])
                
                err_re = err_re_emoti#(err_re_inter + err_re_emoti)/2
                err_total = (err_total_inter + err_total_emoti)/2
                
                k = np.random.randint(0, 10*47,self.batch_size)
                test_emotion = test_array[k]
                test_neutral = test_array[k-(k%47)]
                K.set_learning_phase(0)
                eval_re, eval_total, eval_kl, eval_regular = self.test_func([test_emotion, test_neutral])
                if index%display_step == 0:
                    print(('Epoch: {:3}, total_loss: {:8.4f}, re_loss: {:8.4f}, kl_loss: {:8.4f}, regular: {:8.4f}, eval: {:8.4f}, eval_re: {:8.4f}, eval_kl: {:8.4f}').format(i, err_total, err_re, err_kl, err_regular, eval_total, eval_re, eval_kl))
                log[i*ITS + index] += err_re
                test_log[i*ITS + index] += eval_re
            np.save('log', log)
            np.save('testlog', test_log)
        self.save_models()
    def special_train(self, epoch):
        def get_interpolate_data(prefix, num = 2000):
            if prefix == 'disentangle':
                #interpolate_data = np.vstack(batch_change(np.fromfile('data/{}/real_data/{}.dat'.format(prefix, i))) for i in range(num))
                interpolate_data = np.vstack(batch_change(np.fromfile('data/{}/Interpolated_results/interpolated_{}.dat'.format(prefix, i))) for i in range(num))

            else:
                interpolate_data = np.vstack(np.fromfile('data/{}/real_data/{}.dat'.format(prefix, i)) for i in range(num))
                mean_inter = np.mean(interpolate_data, axis = 0)
                interpolate_data = interpolate_data - mean_inter
            return interpolate_data
        data_array = np.load(('data/{}/train_data.npy').format(self.prefix))[47*np.arange(140)]
        test_array = np.load(('data/{}/test_data.npy').format(self.prefix))[47*np.arange(10)]
        inter_array = get_interpolate_data(self.prefix)
        normalize_fromfile(inter_array, self.M_list, self.m_list)
        normalize_fromfile(data_array, self.M_list, self.m_list)
        normalize_fromfile(test_array, self.M_list, self.m_list)
        data_array = np.concatenate([data_array, inter_array])

        log = np.zeros((epoch,))
        test_log = np.zeros((epoch,))
        constant_list = np.arange(data_array.shape[0])
        
        
        display_step = 50
        
        for i in range(epoch):

            np.random.shuffle(constant_list)
            for index, j in enumerate(zip(*[iter(constant_list)]*self.batch_size)):
                test_idx = np.random.randint(0,10,self.batch_size)
                test_emotion = test_array[test_idx]
                people_with_emotion = data_array[np.array(j)]
                K.set_learning_phase(1)
                err_re, err_total, err_kl, err_regular = self.train_func([people_with_emotion, people_with_emotion])
                K.set_learning_phase(0)
                eval_re, eval_total, eval_kl, eval_regular = self.test_func([test_emotion, test_emotion])
                if index%display_step == 0:
                    print(('Epoch: {:3}, total_loss: {:8.4f}, re_loss: {:8.4f}, kl_loss: {:8.4f}, regular: {:8.4f}, eval: {:8.4f}, eval_re: {:8.4f}, eval_kl: {:8.4f}').format(i, err_total, err_re, err_kl, err_regular, eval_total, eval_re, eval_kl))
                log[i] += err_total
                test_log[i] += eval_total
            np.save('log', log)
            np.save('testlog', test_log)

        self.save_models()
        
    def test(self, limit=5, filename='test', people_id=142):
        data = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, filename, people_id))
        data_array = data.copy()
        normalize_fromfile(data_array, self.M_list, self.m_list)
        err_re, err_total, err_kl, _ = self.test_func([data_array[24:25], data_array[:1]])
        print(err_re)
        feature_id = denormalize_fromfile(self.gcn_vae_id.predict(data_array, batch_size=self.batch_size), self.M_list, self.m_list)
        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        for i in (0, 1, 2, 22, 24, 25, 37, 39):
            V2M2(get_mesh(ref_name, data_recover(feature_id[i])), ('data/mesh/id_{}_{}.obj').format(self.prefix, i))
            V2M2(get_mesh(ref_name, data_recover(data[i])), ('data/mesh/ori_{}_{}.obj').format(self.prefix, i))
            
            


class disentangle_model_vae_exp(gcn_dis_model):

    def build_model(self, MAX_DEGREE):
        SYM_NORM = True
        A = sp.load_npz(('data/{}/FWH_adj_matrix.npz').format(self.prefix))
        L = normalized_laplacian(A, SYM_NORM)
        T_k = chebyshev_polynomial(rescale_laplacian(L), MAX_DEGREE)
        support = MAX_DEGREE + 1
        self.kl_loss, self.encoder, self.decoder, self.gcn_vae_exp = get_gcn_vae_exp(T_k, support, batch_size=self.batch_size, feature_dim=self.feature_dim, v=self.v, input_dim=self.input_dim, latent_dim = self.latent_dim_exp)
        self.mean_exp = Input(shape=(self.input_dim,))
        real = self.gcn_vae_exp.get_input_at(0)
        ratio = K.variable(self.M_list - self.m_list)
        # L2 when xyz, L1 when rimd
        if self.feature_dim == 9:
            #self.away_loss = 0.001/K.mean(K.abs(0.9*s- ( self.gcn_vae_exp(real)) * ratio))
            self.exp_loss = K.mean(K.abs((self.mean_exp - self.gcn_vae_exp(real)) * ratio )) / 1.8 #+ self.away_loss
        
        else:
            self.exp_loss = K.mean(K.square((self.mean_exp - self.gcn_vae_exp(real)) * ratio )) * 100
        
        self.loss = self.exp_loss + self.kl_weight * self.kl_loss
        weights = self.gcn_vae_exp.trainable_weights
        training_updates = (Adam(lr=self.lr)).get_updates(weights, [], self.loss)
        self.train_func = K.function([real, self.mean_exp], [self.exp_loss, self.loss, self.kl_loss], training_updates)
        self.test_func = K.function([real, self.mean_exp], [self.exp_loss, self.loss, self.kl_loss])
        if self.load:
            self.load_models()

    def save_models(self):
        self.gcn_vae_exp.save_weights(('model/gcn_vae_exp_model/gcn_vae_exp{}{}.h5').format(self.prefix, self.suffix))
        self.encoder.save_weights(('model/gcn_vae_exp_model/encoder_exp_{}{}.h5').format(self.prefix, self.suffix))
        self.decoder.save_weights(('model/gcn_vae_exp_model/decoder_exp_{}{}.h5').format(self.prefix, self.suffix))
        
    def load_models(self):
        self.gcn_vae_exp.load_weights(('model/gcn_vae_exp_model/gcn_vae_exp{}{}.h5').format(self.prefix, self.suffix))

    def train(self, epoch):
        data_array = np.load(('data/{}/train_data.npy').format(self.prefix))
        test_array = np.load(('data/{}/test_data.npy').format(self.prefix))
        mean_exp = np.load(('data/{}/MeanFace_data.npy').format(self.prefix))
        normalize_fromfile(mean_exp, self.M_list, self.m_list)
        normalize_fromfile(data_array, self.M_list, self.m_list)
        normalize_fromfile(test_array, self.M_list, self.m_list)
        log = np.zeros((epoch,))
        test_log = np.zeros((epoch,))
        constant_list = np.arange(6580)
        for i in range(epoch):
            k = np.random.randint(1, 11)
            test_emotion = test_array[k * 47 - 47:k * 47]
            np.random.shuffle(constant_list)
            for j in constant_list:
                l = np.random.randint(0, 47)
                C_exp = j % 47
                people_with_emotion = data_array[j:j + 1]
                exp = mean_exp[C_exp:C_exp+1]
                err_re, err_total, err_kl = self.train_func([people_with_emotion, exp])
                eval_re, eval_total, eval_kl = self.test_func([test_emotion[l:l + 1], mean_exp[l:l+1]])
                print(('Epoch: {:3}, people: {:4}, total_loss: {:8.6f}, re_loss: {:8.6f}, kl_loss: {:8.4f}, eval: {:8.6f}, eval_re: {:8.6f}, eval_kl: {:8.4f}').format(i, j, err_total, err_re, err_kl, eval_total, eval_re, eval_kl))
                log[i] += err_total
                test_log[i] += eval_total
        np.save('log', log)
        np.save('testlog', test_log)
        self.save_models()

    def test(self, limit=5, filename='test', people_id=142):
        data = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, filename, people_id))
        data_array = data.copy()
        normalize_fromfile(data_array, self.M_list, self.m_list)
        feature_exp = denormalize_fromfile(self.gcn_vae_exp.predict(data_array, batch_size=self.batch_size), self.M_list, self.m_list)
        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        for i in (0, 1, 2, 22, 37, 39):
            V2M2(get_mesh(ref_name, data_recover(feature_exp[i])), ('data/mesh/exp_{}_{}.obj').format(self.prefix, i))
            V2M2(get_mesh(ref_name, data_recover(data[i])), ('data/mesh/ori_{}_{}.obj').format(self.prefix, i))
            
            
    def test_fusion(self, id_net):
        filename = 'test'
        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        os.mkdir('data/mesh/id')
        os.mkdir('data/mesh/exp')
        os.mkdir('data/mesh/rec')
        os.mkdir('data/mesh/ori')
        for people_id in range(141, 151):
            os.mkdir(('data/mesh/id/Feature{}').format(people_id))
            os.mkdir(('data/mesh/exp/Feature{}').format(people_id))
            os.mkdir(('data/mesh/rec/Feature{}').format(people_id))
            os.mkdir(('data/mesh/ori/Feature{}').format(people_id))
            data = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, filename, people_id))
            data_array = data.copy()
            normalize_fromfile(data_array, self.M_list, self.m_list)

            feature_id = denormalize_fromfile(id_net.decoder.predict(id_net.encoder.predict(data_array, batch_size=self.batch_size)[0],batch_size=self.batch_size), self.M_list, self.m_list)
            feature_exp = denormalize_fromfile(self.decoder.predict(self.encoder.predict(data_array, batch_size=self.batch_size)[0],batch_size=self.batch_size), self.M_list, self.m_list)
            for i in range(47):
                V2M2(get_mesh(ref_name, data_recover(data[i])), ('data/mesh/ori/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(data[i]-feature_id[i])), ('data/mesh/id/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_exp[i])), ('data/mesh/exp/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_exp[i] + data[i]-feature_id[i])), ('data/mesh/rec/Feature{}/{}.obj').format(people_id, i))
                
    def test_training_pose(self, id_net, fusion_net):
        from src.measurement import write_align_mesh
        filename = '/raid/jzh/alignpose/RimdFeature1024/gathered_feature'
        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        os.mkdir('data/mesh/exp')
        os.mkdir('data/mesh/id')
        os.mkdir('data/mesh/rec_plus')
        os.mkdir('data/mesh/rec_fusion')
        for people_id in range(141, 151):
            os.mkdir(('data/mesh/exp/Feature{}').format(people_id))
            os.mkdir(('data/mesh/id/Feature{}').format(people_id))
            os.mkdir(('data/mesh/rec_plus/Feature{}').format(people_id))
            os.mkdir(('data/mesh/rec_fusion/Feature{}').format(people_id))
            data = np.load(('{}/Feature{}.npy').format(filename, people_id))
            data_array = data.copy()
            normalize_fromfile(data_array, self.M_list, self.m_list)
            
            #norm_id = data_array -id_net.decoder.predict(id_net.encoder.predict(data_array, batch_size=self.batch_size)[0],batch_size=self.batch_size) - 0.9*( self.M_list+ self.m_list)/( self.M_list- self.m_list) 
            norm_id =id_net.gcn_vae_id.predict(data_array, batch_size=self.batch_size)
            norm_exp = self.decoder.predict(self.encoder.predict(data_array, batch_size=self.batch_size)[0],batch_size=self.batch_size)
            norm_rec = fusion_net.gcn_comp.predict([norm_id, norm_exp],batch_size = self.batch_size)+ 0.9*( self.M_list+ self.m_list)/( self.M_list- self.m_list) 
            
            feature_id = denormalize_fromfile(norm_id, self.M_list, self.m_list)
            feature_exp = denormalize_fromfile(norm_exp, self.M_list, self.m_list)
            feature_rec = denormalize_fromfile(norm_rec, self.M_list, self.m_list)
            
            for i in range(20):
                V2M2(get_mesh(ref_name, data_recover(feature_exp[i])), ('data/mesh/exp/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_id[i])), ('data/mesh/id/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_id[i] + feature_exp[i])), ('data/mesh/rec_plus/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_rec[i])), ('data/mesh/rec_fusion/Feature{}/{}.obj').format(people_id, i))
                write_align_mesh(('data/mesh/rec_plus/Feature{}/{}.obj').format(people_id, i),'/raid/jzh/alignpose/Tester_{}/AlignPose/pose_{}.obj'.format(people_id, i),('data/mesh/rec_fusion/Feature{}/aligned_{}.obj').format(people_id, i))
                write_align_mesh(('data/mesh/rec_fusion/Feature{}/{}.obj').format(people_id, i),'/raid/jzh/alignpose/Tester_{}/AlignPose/pose_{}.obj'.format(people_id, i),('data/mesh/rec_plus/Feature{}/aligned_{}.obj').format(people_id, i))
                
                
    def test_change(self, id_net):
        from src.measurement import write_align_mesh
        filename = '/raid/jzh/alignpose/RimdFeature1024/gathered_feature'
        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        data_array141 = np.load(('{}/Feature{}.npy').format(filename, 141))
        data_array142 = np.load(('{}/Feature{}.npy').format(filename, 142))
        data141 = data_array141.copy()
        data142 = data_array142.copy()
        normalize_fromfile(data_array141, self.M_list, self.m_list)
        normalize_fromfile(data_array142, self.M_list, self.m_list)
        
        feature_id_141 = denormalize_fromfile(id_net.decoder.predict(id_net.encoder.predict(data_array141, batch_size=self.batch_size)[0],batch_size=self.batch_size), self.M_list, self.m_list)
        feature_exp_141 = denormalize_fromfile(self.decoder.predict(self.encoder.predict(data_array141, batch_size=self.batch_size)[0],batch_size=self.batch_size), self.M_list, self.m_list)
        feature_id_142 = denormalize_fromfile(id_net.decoder.predict(id_net.encoder.predict(data_array142, batch_size=self.batch_size)[0],batch_size=self.batch_size), self.M_list, self.m_list)
        feature_exp_142 = denormalize_fromfile(self.decoder.predict(self.encoder.predict(data_array142, batch_size=self.batch_size)[0],batch_size=self.batch_size), self.M_list, self.m_list)

        for i in range(20):
            V2M2(get_mesh(ref_name, data_recover(data141[0] - feature_id_141[0] + feature_exp_142[i])), ('data/mesh/141_id_142_exp_{}.obj'.format(i)))
            V2M2(get_mesh(ref_name, data_recover(data142[0] - feature_id_142[0] + feature_exp_141[i])), ('data/mesh/142_id_141_exp_{}.obj'.format(i)))
            write_align_mesh('data/mesh/141_id_142_exp_{}.obj'.format(i),'/raid/jzh/alignpose/Tester_{}/AlignPose/pose_{}.obj'.format(141, i),'data/mesh/alighed_141_id_142_exp_{}.obj'.format(i))
            write_align_mesh('data/mesh/142_id_141_exp_{}.obj'.format(i),'/raid/jzh/alignpose/Tester_{}/AlignPose/pose_{}.obj'.format(142, i),'data/mesh/alighed_142_id_141_exp_{}.obj'.format(i))

class gcn_model(object):

    def __init__(self, input_dim, prefix, suffix, lr, load, feature_dim=9, batch_size=1, MAX_DEGREE=1):
        self.input_dim = input_dim
        self.prefix = prefix
        self.suffix = suffix
        self.lr = lr
        self.load = load
        self.M_list = np.load(('data/{}/max_data.npy').format(self.prefix))
        self.m_list = np.load(('data/{}/min_data.npy').format(self.prefix))
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.v = int(input_dim / feature_dim)
        ratio = K.variable(self.M_list - self.m_list)
        s = K.variable(self.m_list + self.M_list)
        SYM_NORM = True
        A = sp.load_npz(('data/{}/FWH_adj_matrix.npz').format(self.prefix))
        L = normalized_laplacian(A, SYM_NORM)
        T_k = chebyshev_polynomial(rescale_laplacian(L), MAX_DEGREE)
        support = MAX_DEGREE + 1
        self.gcn_comp = get_gcn(T_k, support, batch_size=self.batch_size, feature_dim=self.feature_dim, v=self.v, input_dim=self.input_dim)
        self.real = Input(shape=(self.input_dim,))
        self.neutral_face = Input(shape=(self.input_dim,))
        self.mean_exp = Input(shape=(self.input_dim,))
        if feature_dim == 9:
            self.loss = K.mean(K.abs((self.real - self.gcn_comp([self.neutral_face, self.mean_exp])) * ratio - 0.9 * s))
        else:
            self.loss = K.mean(K.square((self.real - self.gcn_comp([self.neutral_face, self.mean_exp])) * ratio- 0.9 * s))
        self.weights = self.gcn_comp.trainable_weights
        #print(self.weights)
        self.training_updates = (Adam(lr=lr)).get_updates(self.weights, [], self.loss)
        self.train_func = K.function([self.real, self.mean_exp, self.neutral_face], [self.loss], self.training_updates)
        self.test_func = K.function([self.real, self.mean_exp, self.neutral_face], [self.loss])
        if not load:
            self.load_models()

    def save_models(self):
        self.gcn_comp.save_weights(('model/gcn_comp_{}{}.h5').format(self.prefix, self.suffix))

    def load_models(self):
        #self.gcn_comp.load_weights(('model/gcn_comp_{}{}.h5').format(self.prefix, self.suffix))
        self.gcn_comp.load_weights(('model/gcn_comp_{}{}.h5').format(self.prefix, 'fusion_rimd'))

    def train(self, epoch):
        data_array = np.load(('data/{}/train_data.npy').format(self.prefix))
        mean_exp = np.load(('data/{}/MeanFace_data.npy').format(self.prefix))
        normalize_fromfile(data_array, self.M_list, self.m_list)
        normalize_fromfile(mean_exp, self.M_list, self.m_list)
        log = np.zeros((epoch,))
        test_log = np.zeros((epoch,))
        batch_size = self.batch_size
        train_people_num = 130
        constant_list = np.arange(train_people_num * 47)
        for i in range(epoch):
            k = np.random.randint(1, 11)
            test_emotion = data_array[train_people_num * 47 + k * 47 - 47:train_people_num * 47 + k * 47]
            np.random.shuffle(constant_list)
            for j in range(int(train_people_num * 47 / batch_size)):
                batch_index = constant_list[j * batch_size:j * batch_size + batch_size]
                exp_index = batch_index % 47
                neutral_index = batch_index - exp_index
                people_with_emotion = data_array[batch_index]
                people_neutral_face = data_array[neutral_index]
                people_exp = mean_exp[exp_index]

                err_total = self.train_func([people_with_emotion, people_exp, people_neutral_face])
                if err_total[0] > 0.04:
                    for _ in range(5):
                        err_total =self.train_func([people_with_emotion, people_exp, people_neutral_face])
                    if err_total[0] > 0.06:
                        print('bad data')
                        for _ in range(10):
                            err_total =self.train_func([people_with_emotion, people_exp, people_neutral_face])
                t = np.random.randint(1, 46)
                eval_total = self.test_func([test_emotion[t:t+self.batch_size], mean_exp[t:t+self.batch_size], np.repeat(test_emotion[:1], self.batch_size, axis=0)])
                print(('Epoch: {:3}, people: {:4}, total_loss: {:8.6f}, eval: {:8.6f}').format(i, j, err_total[0], eval_total[0]))
                log[i] += err_total[0]
                test_log[i] += eval_total[0]
                
                
        np.save('test_log', test_log)
        np.save('log', log)
        self.save_models()
    def train_fusion(self, id_net, exp_net, epoch):
        def get_interpolate_data(prefix, num = 2000):
            if prefix == 'disentangle':
                interpolate_data = np.vstack(batch_change(np.fromfile('data/{}/Interpolated_results/interpolated_{}.dat'.format(prefix, i))) for i in range(num))
            else:
                interpolate_data = np.vstack(np.fromfile('data/{}/Interpolated_results/interpolated_{}.dat'.format(prefix, i)) for i in range(num))
                mean_inter = np.mean(interpolate_data, axis = 0)
                interpolate_data = interpolate_data - mean_inter
            return interpolate_data
        def reshape_feature(x):
            return K.reshape(x, (self.batch_size, -1, 3))
        data_array = np.load(('data/{}/train_data.npy').format(self.prefix))
        test_array = np.load(('data/{}/test_data.npy').format(self.prefix))
        mean_exp = np.load(('data/{}/MeanFace_data.npy').format(self.prefix))
        inter_array = get_interpolate_data(self.prefix)
        
        normalize_fromfile(inter_array, self.M_list, self.m_list)
        normalize_fromfile(test_array, self.M_list, self.m_list)
        normalize_fromfile(data_array, self.M_list, self.m_list)
        normalize_fromfile(mean_exp, self.M_list, self.m_list)
        
        data_array = np.concatenate([data_array, mean_exp, inter_array])
        
        train_people_num = data_array.shape[0]
        
        log = np.zeros((epoch * train_people_num,))
        test_log = np.zeros((epoch * train_people_num,))
        batch_size = self.batch_size
        
        constant_list = np.arange(train_people_num)
        
        id_mesh = id_net.decoder(id_net.encoder(self.real)[0])
        exp_mesh = exp_net.decoder(exp_net.encoder(self.real)[0])
        rec_mesh = self.gcn_comp([id_mesh,exp_mesh])
        ratio = K.variable(self.M_list - self.m_list)
        s = K.variable(self.M_list + self.m_list)
        self.regularization_loss = 0
        for w in self.weights:
            #print(w)
            if self.feature_dim == 9:
                self.regularization_loss += 0.00001*  K.sum(K.abs(w))
            else:
                self.regularization_loss += 0.00001*  K.sum(K.square(w))
        if self.feature_dim == 9:
            loss_func = K.mean(K.abs((self.real - rec_mesh)* ratio - 0.9* s))+ self.regularization_loss
        else:
            loss_func = K.mean(K.sqrt(K.sum(K.square(reshape_feature((self.real - rec_mesh)* ratio - 0.9* s)) ,axis=-1)))/1.8+ self.regularization_loss
        
        training_updates = (Adam(lr=self.lr)).get_updates(self.weights, [], loss_func)
        train_function = K.function([self.real], [loss_func, self.regularization_loss], training_updates)
        test_function = K.function([self.real], [loss_func, self.regularization_loss])
        
        display_step = 50
        
        for i in range(epoch):

            np.random.shuffle(constant_list)
            for j in range(int(train_people_num/batch_size)):
                batch_index = constant_list[j * batch_size:j * batch_size + batch_size]
                #exp_index = batch_index % 47
                #neutral_index = batch_index - exp_index
                people_with_emotion = data_array[batch_index]
                #people_neutral_face = data_array[neutral_index]
                #people_exp = mean_exp[exp_index]
                k = np.random.randint(test_array.shape[0])
                test_emotion = test_array[k :k + 1]
                err_total = train_function([people_with_emotion])
                eval_total = test_function([test_emotion])
                
                log[i*train_people_num + j] += err_total[0]
                test_log[i*train_people_num + j] += eval_total[0]
                if j%display_step ==0:
                    print(('Epoch: {:3}, people: {:4}, total_loss: {:8.6f}, regular_loss: {:8.6f}, eval: {:8.6f}').format(i, j, err_total[0],err_total[1], eval_total[0]))
                    np.save('testlog', test_log)
                    np.save('log', log)
        np.save('testlog', test_log)
        np.save('log', log)
        self.save_models()
        
    def end_to_end(self, id_net, exp_net, epoch):
        #---------------------
        # future part
        #---------------------
        
        ratio = K.variable(self.M_list - self.m_list)
        s = K.variable(self.m_list + self.M_list)
        # id operator
        def I(z):
            return id_net.decoder(id_net.encoder(z)[2])
        def E(z):
            return exp_net.decoder(exp_net.encoder(z)[2])
        def F(y,x):
            return self.gcn_comp([y,x])
        def reshape_feature(x):
            return K.reshape(x, (self.batch_size, -1, 3))
            #ori_mesh = K.reshape(((self.neutral_face - self.gcn_vae_id(real)) * ratio), (self.batch_size, -1, 3))
            #self.id_loss = K.mean(K.sqrt(K.sum(K.square(ori_mesh) ,axis=-1)))/1.8
        '''
         1. I(z_{i,k}) - y_i = 0
         2. E(z_{i,k}) - x_k = 0
         3. E(I(z_{i,k})) = I(E(z_{i,k})) = 0
         4. F(E(z_{i,k}),I(z_{i,k}) - z_{i,k} = 0
        '''
        z = self.real
        x = self.mean_exp
        y = self.neutral_face
        
        if self.feature_dim == 9:
            loss_id = K.mean(K.abs((I(z) - y)*ratio))+ id_net.kl_weight * id_net.kl_loss
            loss_exp = K.mean(K.abs((E(z) - x)*ratio))+ exp_net.kl_weight * exp_net.kl_loss
            loss_disentangle = K.mean(K.abs(E(I(z))*ratio + 0.9*s)) + K.mean(K.abs(I(E(z))*ratio + 0.9*s))
            loss_rec = K.mean(K.abs(F(I(z),E(z))*ratio+ 0.9*s - z*ratio))
        else:
            loss_id = K.mean(K.sqrt(K.sum(K.square(reshape_feature((I(z) - y)*ratio)) ,axis=-1)))/1.8 + id_net.kl_weight * id_net.kl_loss
            loss_exp = K.mean(K.sqrt(K.sum(K.square(reshape_feature((E(z) - x)*ratio)) ,axis=-1)))/1.8+ exp_net.kl_weight * exp_net.kl_loss
            loss_disentangle = K.mean(K.sqrt(K.sum(K.square(reshape_feature(E(I(z))*ratio + 0.9*s)) ,axis=-1)))/1.8 + K.mean(K.sqrt(K.sum(K.square(reshape_feature(I(E(z))*ratio + 0.9*s)) ,axis=-1)))/1.8 
            loss_rec = K.mean(K.sqrt(K.sum(K.square(reshape_feature(F(I(z),E(z))*ratio+ 0.9*s - z*ratio)) ,axis=-1)))/1.8 
        
        
        weights_I =  id_net.encoder.trainable_weights + id_net.decoder.trainable_weights
        weights_E =  exp_net.encoder.trainable_weights + exp_net.decoder.trainable_weights
        weights_F =  self.gcn_comp.trainable_weights
        
        regular_loss = 0
        for weight in weights_I:
            if self.feature_dim == 9:
                regular_loss += 0.00003*K.sum(K.square(weight))
            else:
                regular_loss += 0.00003*K.sum(K.square(weight))
        for weight in weights_E:
            if self.feature_dim == 9:
                regular_loss += 0.000001*K.sum(K.square(weight))
            else:
                regular_loss += 0.000001*K.sum(K.square(weight))
        for weight in weights_F:
            if self.feature_dim == 9:
                regular_loss += 0.00001* K.sum(K.square(weight))
            else:
                regular_loss += 0.00001* K.sum(K.square(weight))
        total_loss = loss_id + loss_exp + loss_disentangle + loss_rec + regular_loss
        our_model = Model(z,[I(z), E(z), F(I(z),E(z))])


        def load_models():
            our_model.load_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))
        def save_models():
            our_model.save_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))
            
        id_z = id_net.encoder.get_input_at(0)
        exp_z = exp_net.encoder.get_input_at(0)
        if self.load:
            load_models()
            
        training_updates = (Adam(lr=self.lr)).get_updates(weights_I+weights_E+weights_F, [], total_loss)
        train_func = K.function([id_z, exp_z, z, x, y], [total_loss, loss_id, loss_exp, loss_disentangle, loss_rec,regular_loss], training_updates)
        test_func = K.function([id_z, exp_z, z, x, y], [total_loss, loss_id, loss_exp, loss_disentangle, loss_rec])
        

        def get_interpolate_data(prefix, num = 2000):
            if prefix == 'disentangle':
                interpolate_data = np.vstack(batch_change(np.fromfile('data/{}/real_data/{}.dat'.format(prefix, i))) for i in range(num))
            else:
                interpolate_data = np.vstack(np.fromfile('data/{}/Interpolated_results/interpolated_{}.dat'.format(prefix, i)) for i in range(num))
                mean_inter = np.mean(interpolate_data, axis = 0)
                interpolate_data = interpolate_data - mean_inter
            return interpolate_data
        
        data_array = np.load(('data/{}/train_data.npy').format(self.prefix))
        test_array = np.load(('data/{}/test_data.npy').format(self.prefix))
        mean_exp = np.load(('data/{}/MeanFace_data.npy').format(self.prefix))
        inter_array = get_interpolate_data(self.prefix,1)
        
        normalize_fromfile(inter_array, self.M_list, self.m_list)
        normalize_fromfile(test_array, self.M_list, self.m_list)
        normalize_fromfile(data_array, self.M_list, self.m_list)
        normalize_fromfile(mean_exp, self.M_list, self.m_list)
        
        data_array = np.concatenate([data_array, mean_exp, inter_array])
        
        train_people_num = data_array.shape[0]
        
        log = np.zeros((epoch * train_people_num,))
        test_log = np.zeros((epoch * train_people_num,))
        
        display_step = 50
        constant_list = np.arange(train_people_num)
        for i in range(epoch):
            k = np.random.randint(1, 11)
            test_emotion = test_array[k * 47 - 47: k * 47]
            np.random.shuffle(constant_list)
            for j in range(train_people_num):
                batch_index = constant_list[j]
                if batch_index < 141*47:
                    exp_index = batch_index % 47
                    neutral_index = batch_index - exp_index
                else:
                    exp_index = 0
                    neutral_index = batch_index
                
                
                people_with_emotion = data_array[batch_index:batch_index+1]
                people_neutral_face = data_array[neutral_index:neutral_index +1]
                people_exp = mean_exp[exp_index:exp_index+1]

                err_total, err_id, err_exp, err_dis, err_rec, err_regular  = train_func([people_with_emotion,people_with_emotion,people_with_emotion, people_exp, people_neutral_face])
                t = np.random.randint(0, 46)
                eval_total, eval_id, eval_exp, eval_dis, eval_rec = test_func([test_emotion[t:t+1],test_emotion[t:t+1],test_emotion[t:t+1], mean_exp[t:t+1], test_emotion[:1]])

                if j%display_step == 0:
                    print(('Epoch: {:3}, people: {:4}, total_loss: {:8.6f}, eval: {:8.6f}').format(i, j, err_total, eval_total))
                    print('ERROR: id: {:8.6f}, exp:{:8.6f}, disentangle: {:8.6f}, rec:{:8.6f}, regular: {:8.6f}'.format(err_id, err_exp, err_dis, err_rec, err_regular))
                    print('EVAL: id: {:8.6f}, exp:{:8.6f}, disentangle: {:8.6f}, rec:{:8.6f}'.format( eval_id, eval_exp, eval_dis, eval_rec))

                log[i] += err_total
                test_log[i] += eval_total
                
                
        np.save('test_log', test_log)
        np.save('log', log)
        save_models()
        
    def test(self, id_net, exp_net, filename='test', people_id=142):
        # id operator
        def I(z):
            return id_net.decoder(id_net.encoder(z)[2])
        def E(z):
            return exp_net.decoder(exp_net.encoder(z)[2])
        def F(y,x):
            return self.gcn_comp([y,x])
        
        '''
         1. I(z_{i,k}) - y_i = 0
         2. E(z_{i,k}) - x_k = 0
         3. E(I(z_{i,k})) = I(E(z_{i,k})) = 0
         4. F(E(z_{i,k}),I(z_{i,k}) - z_{i,k} = 0
        '''
        z = self.real

        our_model = Model(z,[I(z), E(z), F(I(z),E(z))])


        def load_models():
            our_model.load_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))
        def save_models():
            our_model.save_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))

        if self.load:
            load_models()
        #self.load_models()
        data = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, filename, people_id))
        data_array = data.copy()
        normalize_fromfile(data_array, self.M_list, self.m_list)
        
        
        exp_code = exp_net.encoder.predict(data_array, batch_size = self.batch_size)[0]
        id_code = id_net.encoder.predict(data_array, batch_size = self.batch_size)[0]
        
        norm_exp = exp_net.decoder.predict(exp_code, batch_size = self.batch_size)
        norm_id = id_net.decoder.predict(id_code, batch_size = self.batch_size)
        

        feature_rec = denormalize_fromfile((self.gcn_comp.predict([norm_id, norm_exp], batch_size=self.batch_size)+ 0.9 * (self.m_list + self.M_list) / (self.M_list - self.m_list)) , self.M_list, self.m_list)
        feature_exp = denormalize_fromfile(norm_exp , self.M_list, self.m_list)
        feature_id = denormalize_fromfile(norm_id , self.M_list, self.m_list)

        
        feature_plus = feature_exp + feature_id
        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        for i in (0, 1, 2, 22, 37, 39):
            V2M2(get_mesh(ref_name, data_recover(feature_rec[i])), ('data/mesh/rec_{}_{}.obj').format(self.prefix, i))
            V2M2(get_mesh(ref_name, data_recover(data[i])), ('data/mesh/ori_{}_{}.obj').format(self.prefix, i))
            V2M2(get_mesh(ref_name, data_recover(feature_plus[i])), ('data/mesh/plus_{}_{}.obj').format(self.prefix, i))
            V2M2(get_mesh(ref_name, data_recover(feature_exp[i])), ('data/mesh/exp_{}_{}.obj').format(self.prefix, i))
            V2M2(get_mesh(ref_name, data_recover(feature_id[i])), ('data/mesh/id_{}_{}.obj').format(self.prefix, i))
            
    def test_change(self, id_net, exp_net, filename='test', people_id=142):
        # id operator
        def I(z):
            return id_net.decoder(id_net.encoder(z)[2])
        def E(z):
            return exp_net.decoder(exp_net.encoder(z)[2])
        def F(y,x):
            return self.gcn_comp([y,x])
        
        '''
         1. I(z_{i,k}) - y_i = 0
         2. E(z_{i,k}) - x_k = 0
         3. E(I(z_{i,k})) = I(E(z_{i,k})) = 0
         4. F(E(z_{i,k}),I(z_{i,k}) - z_{i,k} = 0
        '''
        z = self.real

        our_model = Model(z,[I(z), E(z), F(I(z),E(z))])


        def load_models():
            our_model.load_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))
        def save_models():
            our_model.save_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))

        if self.load:
            load_models()

        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        os.mkdir('data/mesh/people')
        os.mkdir('data/mesh/transfer')
        data_array141 = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, filename, 141))
        data_array142 = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, filename, 142))


        data142 = data_array142.copy()
        normalize_fromfile(data_array141, self.M_list, self.m_list)
        normalize_fromfile(data_array142, self.M_list, self.m_list)
        
        exp_code = exp_net.encoder.predict(data_array142, batch_size = self.batch_size)[0]
        id_code = id_net.encoder.predict(data_array141, batch_size = self.batch_size)[0]
        
        bias = 0.9 * (self.m_list + self.M_list) / (self.M_list - self.m_list)
        
        norm_exp = exp_net.decoder.predict(exp_code, batch_size = self.batch_size)
        norm_id = id_net.decoder.predict(id_code, batch_size = self.batch_size)
        id_used = np.repeat(norm_id[0:1], 47, axis = 0)
        
        norm_transfer = self.gcn_comp.predict([id_used, norm_exp], batch_size = self.batch_size) + bias
        
        feature_id_141 = denormalize_fromfile(id_used, self.M_list, self.m_list)
        feature_exp_142 = denormalize_fromfile(norm_exp, self.M_list, self.m_list)
        feature_transfer = denormalize_fromfile(norm_transfer, self.M_list, self.m_list)
        
        key_frames = (0,1,2,22,37,39,2,1,23,12)
        inter = 20
        for index,i in enumerate(key_frames[:-1]):
            for j in range(inter):
                transfer_feature = (1-j/(inter - 1))*feature_transfer[key_frames[index]] + j/(inter-1)*feature_transfer[key_frames[index + 1]]
                people_feature = (1-j/(inter - 1))*data142[key_frames[index]] + j/(inter - 1)*data142[key_frames[index + 1]]
                V2M2(get_mesh(ref_name, data_recover(people_feature)), ('data/mesh/people/exp_{}_frame_{}.obj').format(index, j))
                V2M2(get_mesh(ref_name, data_recover(transfer_feature)), ('data/mesh/transfer/exp_{}_frame_{}.obj').format(index, j))

            
    def test_whole(self, id_net, exp_net, filename = 'test',people_id = 141):
        # id operator
        def I(z):
            return id_net.decoder(id_net.encoder(z)[2])
        def E(z):
            return exp_net.decoder(exp_net.encoder(z)[2])
        def F(y,x):
            return self.gcn_comp([y,x])
        
        '''
         1. I(z_{i,k}) - y_i = 0
         2. E(z_{i,k}) - x_k = 0
         3. E(I(z_{i,k})) = I(E(z_{i,k})) = 0
         4. F(E(z_{i,k}),I(z_{i,k}) - z_{i,k} = 0
        '''
        z = self.real

        our_model = Model(z,[I(z), E(z), F(I(z),E(z))])


        def load_models():
            our_model.load_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))
        def save_models():
            our_model.save_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))

        if self.load:
            load_models()

        
        def get_optimized_mesh(target_feature, epoch = 50, lr = 0.4):
            id_start = id_net.encoder.predict(target_feature)
            exp_start = exp_net.encoder.predict(target_feature)
            id_code = K.variable(id_start[0])
            exp_code = K.variable(exp_start[0])
            target_feature_holder = Input(shape=(self.input_dim, ))
            target_id = id_net.decoder(id_code)
            target_exp = exp_net.decoder(exp_code)
            target_plus = target_id + target_exp
            target_rec = self.gcn_comp([target_id, target_exp])
            ratio = K.variable(self.M_list - self.m_list)
            s = K.variable(self.M_list + self.m_list)
            plus_loss = K.mean(K.abs(ratio*( target_feature_holder - target_plus) - 0.9*s))/1.8
            rec_loss = K.mean(K.abs(ratio*( target_feature_holder - target_rec) - 0.9*s))/1.8
            
            training_updates_plus = (Adam(lr=lr)).get_updates([id_code, exp_code], [], plus_loss)
            training_updates_rec = (Adam(lr=lr)).get_updates([id_code, exp_code], [], rec_loss)
            
            bp_func_plus = K.function([target_feature_holder], [plus_loss, target_plus], training_updates_plus)
            bp_func_rec = K.function([target_feature_holder], [rec_loss, target_rec], training_updates_rec)
            for i in range(epoch):
                err, result_mesh_plus = bp_func_plus([target_feature])
            print('Plus Error:{}'.format(err))
            for i in range(epoch):
                err, result_mesh_rec = bp_func_rec([target_feature])
            print('Rec Error:{}'.format(err))
            return result_mesh_plus, result_mesh_rec
            
        
        
        
        import shutil, os
        # commit following lines for the second test
        ###
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        os.mkdir('data/mesh/id')
        os.mkdir('data/mesh/exp')
        os.mkdir('data/mesh/rec')
        os.mkdir('data/mesh/ori')
        os.mkdir('data/mesh/plus')
        os.mkdir('data/mesh/bp_plus')
        os.mkdir('data/mesh/bp_rec')
        ###
        # commit above lines for the second test
        for people_id in range(people_id, people_id+1):
            os.mkdir(('data/mesh/id/Feature{}').format(people_id))
            os.mkdir(('data/mesh/exp/Feature{}').format(people_id))
            os.mkdir(('data/mesh/rec/Feature{}').format(people_id))
            os.mkdir(('data/mesh/ori/Feature{}').format(people_id))
            os.mkdir(('data/mesh/plus/Feature{}').format(people_id))
            os.mkdir(('data/mesh/bp_plus/Feature{}').format(people_id))
            os.mkdir(('data/mesh/bp_rec/Feature{}').format(people_id))
            
            #data = np.load(('{}/Feature{}.npy').format(filename, people_id))
            data = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, filename, people_id))
            data_array = data.copy()
            normalize_fromfile(data_array, self.M_list, self.m_list)
            
            
            exp_code = exp_net.encoder.predict(data_array, batch_size = self.batch_size)[0]
            id_code = id_net.encoder.predict(data_array, batch_size = self.batch_size)[0]
            
            norm_exp = exp_net.decoder.predict(exp_code, batch_size = self.batch_size)
            norm_id = id_net.decoder.predict(id_code, batch_size = self.batch_size)
            
    
            feature_rec = denormalize_fromfile((self.gcn_comp.predict([norm_id, norm_exp], batch_size=self.batch_size)+ 0.9 * (self.m_list + self.M_list) / (self.M_list - self.m_list)) , self.M_list, self.m_list)
            feature_exp = denormalize_fromfile(norm_exp , self.M_list, self.m_list)
            feature_id = denormalize_fromfile(norm_id , self.M_list, self.m_list)
            f_plus = np.zeros_like(feature_rec)
            f_rec = np.zeros_like(feature_rec)
            bias = 0.9 * (self.m_list + self.M_list) / (self.M_list - self.m_list)
            for i in range(47):
                f_plus_i, f_rec_i = get_optimized_mesh(data_array[i:i+1])
                f_plus[i] = f_plus_i + bias
                f_rec[i] = f_rec_i+ bias
                
            bp_plus = denormalize_fromfile(f_plus , self.M_list, self.m_list)
            bp_rec = denormalize_fromfile(f_rec , self.M_list, self.m_list)            
            
            for i in range(47):
                
                
                V2M2(get_mesh(ref_name, data_recover(bp_plus[i])), ('data/mesh/bp_plus/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(bp_rec[i])), ('data/mesh/bp_rec/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(data[i])), ('data/mesh/ori/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_rec[i])), ('data/mesh/rec/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_id[i])), ('data/mesh/id/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_exp[i])), ('data/mesh/exp/Feature{}/{}.obj').format(people_id, i))
                V2M2(get_mesh(ref_name, data_recover(feature_exp[i] + feature_id[i])), ('data/mesh/plus/Feature{}/{}.obj').format(people_id, i))
                
                

    def test_interpolation(self, id_net, exp_net):
        
        exp_1 = batch_change(np.fromfile('/home/jzh/CVPR2019/Exploring/extrapolated_144.dat')).reshape(1,-1)
        exp_2 = batch_change(np.fromfile('/home/jzh/CVPR2019/Exploring/extrapolated_167.dat')).reshape(1,-1)
        id_1 = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, 'whole', 43))[:1]
        id_2 = np.load(('data/{}/{}_data/Feature{}.npy').format(self.prefix, 'whole', 134))[:1]
        normalize_fromfile(exp_1, self.M_list, self.m_list)
        normalize_fromfile(exp_2, self.M_list, self.m_list)
        normalize_fromfile(id_1, self.M_list, self.m_list)
        normalize_fromfile(id_2, self.M_list, self.m_list)
        
        # id operator
        def I(z):
            return id_net.decoder(id_net.encoder(z)[2])
        def E(z):
            return exp_net.decoder(exp_net.encoder(z)[2])
        def F(y,x):
            return self.gcn_comp([y,x])
        
        '''
         1. I(z_{i,k}) - y_i = 0
         2. E(z_{i,k}) - x_k = 0
         3. E(I(z_{i,k})) = I(E(z_{i,k})) = 0
         4. F(E(z_{i,k}),I(z_{i,k}) - z_{i,k} = 0
        '''
        z = self.real


        our_model = Model(z,[I(z), E(z), F(I(z),E(z))])
        
        code_id = Input(shape = (50,))
        code_exp = Input(shape = (25,))

        corespondent_mesh = self.gcn_comp([id_net.decoder(code_id),exp_net.decoder(code_exp)])
        
        code2mesh = Model([code_id, code_exp], corespondent_mesh)

        def load_models():
            our_model.load_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))
        def save_models():
            our_model.save_weights(('model/our_model/our_model{}{}.h5').format(self.prefix, self.suffix))
            
            
        def get_optimized_code(target_feature, epoch = 50, lr = 0.5):
            id_start = id_net.encoder.predict(target_feature)
            exp_start = exp_net.encoder.predict(target_feature)
            id_code = K.variable(id_start[0])
            exp_code = K.variable(exp_start[0])
            target_feature_holder = Input(shape=(self.input_dim, ))
            target_id = id_net.decoder(id_code)
            target_exp = exp_net.decoder(exp_code)

            target_rec = self.gcn_comp([target_id, target_exp])
            ratio = K.variable(self.M_list - self.m_list)
            s = K.variable(self.M_list + self.m_list)
            rec_loss = K.mean(K.abs(ratio*( target_feature_holder - target_rec) - 0.9*s))/1.8

            training_updates_rec = (Adam(lr=lr)).get_updates([id_code, exp_code], [], rec_loss)
            bp_func_rec = K.function([target_feature_holder], [rec_loss, target_rec, id_code, exp_code], training_updates_rec)
            for i in range(epoch):
                 err, result_mesh_rec, code_id, code_exp  = bp_func_rec([target_feature])
            print('Rec Error:{}'.format(err))
            return  result_mesh_rec, code_id, code_exp
        
    
        if self.load:
            load_models()
        import shutil, os
        shutil.rmtree('data/mesh')
        os.mkdir('data/mesh')
        
        _, _, exp_code_1 = get_optimized_code(exp_1)
        _, _, exp_code_2 = get_optimized_code(exp_2)
        _, id_code_1, _ = get_optimized_code(id_1)
        _, id_code_2, _ = get_optimized_code(id_2)
        
        bias = 0.9 * (self.m_list + self.M_list) / (self.M_list - self.m_list)
        
        n = 20
 
        for id_go in range(-2,n+1):
            for exp_go in range(-2,n+1):
                new_code_id = id_go/(n-1)*id_code_1 + (1 - id_go/(n-1))*id_code_2
                new_code_exp = exp_go/(n-1)*exp_code_1 + (1 - exp_go/(n-1))*exp_code_2
                norm_rec = code2mesh.predict([new_code_id, new_code_exp],batch_size = self.batch_size) + bias

                
                feature_rec = denormalize_fromfile(norm_rec , self.M_list, self.m_list)
                
                V2M2(get_mesh(ref_name, data_recover(feature_rec[0])), ('data/mesh/id_{}_exp_{}.obj').format(id_go, exp_go))
        

                
                
if __name__ == '__main__':
    start = True

