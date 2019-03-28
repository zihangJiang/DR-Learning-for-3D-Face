import numpy as np
import scipy
from scipy import linalg
import scipy.sparse as sparse
from scipy.sparse import linalg
import openmesh
import tensorflow as tf
try:
    import pyigl as igl
except:
    import src.pyigl as igl
import os
import time
from multiprocessing import Pool as ThreadPool # use multiprocess
from functools import partial

'''
This part use iglhelper function to fulfill the target that transfer between eigen and numpy
'''
def p2e(m):
    if isinstance(m, np.ndarray):
        if not (m.flags['C_CONTIGUOUS'] or m.flags['F_CONTIGUOUS']):
            raise TypeError('p2e support either c-order or f-order')
        if m.dtype.type in [np.int32, np.int64]:
            return igl.eigen.MatrixXi(m.astype(np.int32))
        elif m.dtype.type in [np.float64, np.float32]:
            return igl.eigen.MatrixXd(m.astype(np.float64))
        elif m.dtype.type == np.bool:
            return igl.eigen.MatrixXb(m)
        raise TypeError("p2e only support dtype float64/32, int64/32 and bool")
    if sparse.issparse(m):
        # convert in a dense matrix with triples
        coo = m.tocoo()
        triplets = np.vstack((coo.row, coo.col, coo.data)).T

        triples_eigen_wrapper = igl.eigen.MatrixXd(triplets)

        if m.dtype.type == np.int32:
            t = igl.eigen.SparseMatrixi()
            t.fromcoo(triples_eigen_wrapper)
            return t
        elif m.dtype.type == np.float64:
            t = igl.eigen.SparseMatrixd()
            t.fromCOO(triples_eigen_wrapper)
            return t

    raise TypeError("p2e only support numpy.array or scipy.sparse")

def e2p(m):
    if isinstance(m, igl.eigen.MatrixXd):
        return np.array(m, dtype='float64', order='C')
    elif isinstance(m, igl.eigen.MatrixXi):
        return np.array(m, dtype='int32', order='C')
    elif isinstance(m, igl.eigen.MatrixXb):
        return np.array(m, dtype='bool', order='C')
    elif isinstance(m, igl.eigen.SparseMatrixd):
        coo = np.array(m.toCOO())
        I = coo[:, 0]
        J = coo[:, 1]
        V = coo[:, 2]
        return sparse.coo_matrix((V,(I,J)), shape=(m.rows(),m.cols()), dtype='float64')
    elif isinstance(m, igl.eigen.SparseMatrixi):
        coo = np.array(m.toCOO())
        I = coo[:, 0]
        J = coo[:, 1]
        V = coo[:, 2]
        return sparse.coo_matrix((V,(I,J)), shape=(m.rows(),m.cols()), dtype='int32')
'''
This part use iglhelper function to fulfill the target that transfer between eigen and numpy
'''
prefix = ''
if __name__ == '__main__':
    prefix = '../'
# change some certain position to make the feature additive
change_length = 11510
unit = 9
delta = np.array([1,0,0,1,0,1,0,0,0])
cross_id = np.tile(delta, change_length)
### max_feature list and min_feature list, saved as text file
max_feature = np.load(prefix+"data/disentangle/max_data.npy") + cross_id
min_feature = np.load(prefix+"data/disentangle/min_data.npy") + cross_id
ref_mesh_filename = prefix+"data/disentangle/Mean_Face.obj"  # here is the filename of reference mesh
V = int(max_feature.shape[0]/9)

substract_feature = np.subtract(max_feature, min_feature)


def num2zeroone(x):
    # return x/1.8+0.5
    return (x+0.9)/1.8

def norm_2_ori(filename, normalized = True):
    if isinstance(filename, str):
        if filename[-4:]=='.txt':
            feature = np.loadtxt(filename)
        elif filename[-4:]=='.dat':
            feature = np.fromfile(filename)
        else:
            feature = np.load(filename)
        if normalized:
            feature = np.array(list(map(num2zeroone, feature)))
            feature = np.multiply(feature, substract_feature)
            feature = np.add(feature, min_feature)
        return feature
    else: # feature directly
#        feature = np.array(list(map(lambda x: (x+0.9)/1.8, filename)))
#        feature = np.multiply(feature, substract_feature)
#        feature = np.add(feature, min_feature)
        return filename

def feature_2_matrix(f):
    """
    input: a 9-dim array 
    output: logR and S matrix
    """

    # return np.matrix([ [0,-f[2], f[1]], [f[2],0,-f[0]],[-f[1],f[0],0]]), np.matrix([[f[3],f[4],f[5]],[f[4],f[6],f[7]],[f[5],f[7],f[8]]])   # qianyi's format
    return np.matrix([ [0,f[6], f[7]], [-f[6],0,f[8]],[-f[7],-f[8],0]]), np.matrix([[f[0],f[1],f[2]],[f[1],f[3],f[4]],[f[2],f[4],f[5]]])   # keyu's format

def expm(A):
    """
    compute expm of matrix A
    input: matrix A
    ouptut: expm matrix of A
    """
    return scipy.linalg.expm(A)

def compute_T(A):
    """
    compute T_matrix by dot product
    input : R and S matrix
    output: T_matrix
    """
    return np.dot(A[0], A[1])

def global_para(ref_mesh_filename):
    """
    return some global parameters used in generating mesh
    including : Mesh Face list, Laplacian matrix of face(two formats of sparse matrix), reference mesh
    """
    # get Laplacian matrix
    global F, A, B, ref_mesh
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readOBJ(ref_mesh_filename, V, F)  
    L = igl.eigen.SparseMatrixd() # Laplacian Matrix
    igl.cotmatrix(V, F, L)
    A = e2p(L)
    c = A.col
    r = A.row
    adj = sparse.coo_matrix((np.ones(c.shape[0]), (c, r)),
                        shape=(A.shape[0], A.shape[0]), dtype=np.float32).tocsr()-sparse.eye(A.shape[0])
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    A = sparse.diags(np.power(np.array(adj.sum(1)), 1).flatten(), 0) - adj
    B = A.tocsr()

    ref_mesh = openmesh.read_trimesh(ref_mesh_filename)


def prepare_T(filename, normalized = True):
    feature = norm_2_ori(filename, normalized)
    t1 = time.time()
    # feature saved in format
    # (wx, wy, wz, s00, s01, s02, s11, s12, s22,.....)   qianyi's format
    # (s00, s01, s02, s11, s12, s22, -wz, wy, -wx, ....)   keyu's format

    logR_array = []
    S_array = []
    for i in range(V):
        logR_matrix, S_matrix = feature_2_matrix(feature[9*i:9*i+9])
        logR_array.append(logR_matrix)
        S_array.append(S_matrix)

    pool = ThreadPool(12)
    R_array = list(pool.map(expm, logR_array))

    global T_array
    T_array = list(pool.map(compute_T, zip(R_array, S_array)))
    pool.close()
    #    print("logR S array time cost %.2fs"%(time.time()-t1))
    return T_array

def compute_temp(vh_idx):
    vh = ref_mesh.vertex_handle(vh_idx)
    temp = np.array([0,0,0])
    for vv_it in ref_mesh.vv(vh):
        pi = ref_mesh.point(vh)
        pj = ref_mesh.point(vv_it)
        pij = pi-pj
        eij = np.array(pi-pj)
        temp = temp+B[vh.idx(), vv_it.idx()]*np.dot(T_array[vh.idx()]+T_array[vv_it.idx()], eij)
    return 0.5*np.array([temp[0, 0], temp[0, 1], temp[0, 2]])



def generate_mesh(filename, output_file , normalized = True):
    """
    input: normalized feature && output file
    output: output mesh
    """

    global_para(ref_mesh_filename)
    t0 = time.time()
    prepare_T(filename, normalized)

    pool = ThreadPool()
    b_ = list(pool.map(compute_temp,range(V)))
    pool.close()
    P = scipy.sparse.linalg.spsolve(A.tocsc(), scipy.sparse.csc_matrix(b_, dtype=float))

    V_new = p2e(-P.tocsr().todense())
    igl.writeOBJ(output_file, V_new, F)
    print("write mesh DONE")
def V2M(array, filename):
    def p2e(m):
        if isinstance(m, np.ndarray):
            if not (m.flags['C_CONTIGUOUS'] or m.flags['F_CONTIGUOUS']):
                raise TypeError('p2e support either c-order or f-order')
            if m.dtype.type in [np.int32, np.int64]:
                return igl.eigen.MatrixXi(m.astype(np.int32))
            elif m.dtype.type in [np.float64, np.float32]:
                return igl.eigen.MatrixXd(m.astype(np.float64))
            elif m.dtype.type == np.bool:
                return igl.eigen.MatrixXb(m)
            raise TypeError("p2e only support dtype float64/32, int64/32 and bool")
        if sparse.issparse(m):
            # convert in a dense matrix with triples
            coo = m.tocoo()
            triplets = np.vstack((coo.row, coo.col, coo.data)).T
    
            triples_eigen_wrapper = igl.eigen.MatrixXd(triplets)
    
            if m.dtype.type == np.int32:
                t = igl.eigen.SparseMatrixi()
                t.fromcoo(triples_eigen_wrapper)
                return t
            elif m.dtype.type == np.float64:
                t = igl.eigen.SparseMatrixd()
                t.fromCOO(triples_eigen_wrapper)
                return t
    
        raise TypeError("p2e only support numpy.array or scipy.sparse")

    # read reference mesh
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readOBJ('data/disentangle/Mean_Face.obj',V,F)
    #igl.readOBJ('/home/jzh/Featuredisentangle/src/Mean_Face.obj',V,F)
    # read dat file and add to V matrix, then write mesh
    tar_v = array.copy()#np.fromfile('0_1008.dat')
    tar_v = tar_v.reshape(11510,3)
    new_v = p2e(tar_v)
    igl.writeOBJ(filename,new_v+V,F)
    
def V2M2(array, filename, ref_name = 'data/disentangle/Mean_Face.obj', v_num = 11510):
    def p2e(m):
        if isinstance(m, np.ndarray):
            if not (m.flags['C_CONTIGUOUS'] or m.flags['F_CONTIGUOUS']):
                raise TypeError('p2e support either c-order or f-order')
            if m.dtype.type in [np.int32, np.int64]:
                return igl.eigen.MatrixXi(m.astype(np.int32))
            elif m.dtype.type in [np.float64, np.float32]:
                return igl.eigen.MatrixXd(m.astype(np.float64))
            elif m.dtype.type == np.bool:
                return igl.eigen.MatrixXb(m)
            raise TypeError("p2e only support dtype float64/32, int64/32 and bool")
        if sparse.issparse(m):
            # convert in a dense matrix with triples
            coo = m.tocoo()
            triplets = np.vstack((coo.row, coo.col, coo.data)).T
    
            triples_eigen_wrapper = igl.eigen.MatrixXd(triplets)
    
            if m.dtype.type == np.int32:
                t = igl.eigen.SparseMatrixi()
                t.fromcoo(triples_eigen_wrapper)
                return t
            elif m.dtype.type == np.float64:
                t = igl.eigen.SparseMatrixd()
                t.fromCOO(triples_eigen_wrapper)
                return t
    
        raise TypeError("p2e only support numpy.array or scipy.sparse")

    # read reference mesh
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readOBJ(ref_name,V,F)
    #igl.readOBJ('/home/jzh/Featuredisentangle/src/Mean_Face.obj',V,F)
    # read dat file and add to V matrix, then write mesh
    tar_v = array.copy()#np.fromfile('0_1008.dat')
    tar_v = tar_v.reshape(v_num,3)
    new_v = p2e(tar_v)
    igl.writeOBJ(filename,new_v,F)


def obj2dat_vertex(filename):
    ref_name = 'data/disentangle/Mean_Face.obj'
    V= igl.eigen.MatrixXd()
    F= igl.eigen.MatrixXi()
    igl.readOBJ(ref_name,V,F)
    
    V1= igl.eigen.MatrixXd()
    igl.readOBJ(filename, V1, F)
    
    
    # to dat
    e2p(V1 - V).tofile(filename[:-4]+'.dat')


if __name__ == '__main__':
    """
    Testing code
    """
    #generate_mesh("Normalized_feature/4.txt", "generate_1.obj")
    start = True
    ref_name = '../data/disentangle/Mean_Face.obj'
    from get_mesh import get_mesh


