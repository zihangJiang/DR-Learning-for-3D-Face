import pyigl as igl
import numpy as np
import scipy
import scipy.sparse as sp

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
        return sp.coo_matrix((V,(I,J)), shape=(m.rows(),m.cols()), dtype='float64')
    elif isinstance(m, igl.eigen.SparseMatrixi):
        coo = np.array(m.toCOO())
        I = coo[:, 0]
        J = coo[:, 1]
        V = coo[:, 2]
        return sp.coo_matrix((V,(I,J)), shape=(m.rows(),m.cols()), dtype='int32')

V = igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()
igl.readOBJ('../data/disentangle/Mean_Face.obj', V, F)

L = igl.eigen.SparseMatrixd()
igl.cotmatrix(V, F, L)
A = e2p(L)
print(A)
c = A.col
r = A.row
adj = sp.coo_matrix((np.ones(c.shape[0]), (c, r)),
                    shape=(A.shape[0], A.shape[0]), dtype=np.float32).tocsr()-sp.eye(A.shape[0])
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
sp.save_npz('../data/disentangle/FWH_adj_matrix',adj)
temp = sp.load_npz('../data/disentangle/FWH_adj_matrix.npz')
print(temp)
