from get_dr import *
from get_mesh import *
from mesh import V2M2
from measurement import write_align_mesh
import numpy as np
import os

import numpy as np
import openmesh as om
try:
    import pyigl as igl
except:
    import src.pyigl as igl
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

# def generate_mean_face():
    # print("generating mean face shapes")
    # V = igl.eigen.MatrixXd()
    # F = igl.eigen.MatrixXi()
    # igl.readOBJ('../data/disentangle/Mean_Face.obj', V, F)
    # try:
        # os.makedirs('../data/FWH/Mean_Face')
    # except:
        # pass
    
    # pool = ThreadPool()
    # for i in tqdm(range(47)):
        # def process(j):
            # return om.read_trimesh('../data/FaceWarehouse_Data/Tester_{}/Blendshape/shape_{}.obj'.format(j,i)).points()
        # gather_mesh = pool.map(process, range(1,151))
        # mean_mesh = igl.eigen.MatrixXd((sum(gather_mesh)/len(gather_mesh)).astype(np.float64))
        # igl.writeOBJ('../data/FWH/Mean_Face/shape_{}.obj'.format(i), mean_mesh, F)
		# write_align_mesh('../data/FWH/Mean_Face/shape_{}.obj'.format(i),
		# '../data/disentangle/Mean_Face.obj',
		# '../data/FWH/Mean_Face/shape_{}.obj'.format(i),
		# index = np.loadtxt('front_part_v.txt',dtype=int))
def generate_mean_face():
    print("generating mean face shapes")
    mean_exp = np.load(('../data/{}/MeanFace_data.npy').format('disentangle'))
    try:
        os.makedirs('../data/FWH/Mean_Face')
    except:
        pass
    for i in range(47):
        V2M2(get_mesh('../data/disentangle/Mean_Face.obj', mean_exp[i]), '../data/FWH/Mean_Face/shape_{}.obj'.format(i))
		write_align_mesh('../data/FWH/Mean_Face/shape_{}.obj'.format(i),
		'../data/disentangle/Mean_Face.obj',
		'../data/FWH/Mean_Face/shape_{}.obj'.format(i),
		index = np.loadtxt('front_part_v.txt',dtype=int))
	
	
def generate_dr_feature():
    print("generating dr feature")
    for j in range(1,151):
        try:
            os.makedirs('../data/FWH/Tester_{}'.format(j))
        except:
            pass
        for i in range(47):
            print('dealing with Tester_{}, Shape_{}'.format(j,i))
            write_align_mesh('../data/FaceWarehouse_Data/Tester_{}/Blendshape/shape_{}.obj'.format(j,i),
            '../data/FWH/Mean_Face/shape_{}.obj'.format(i), #'../data/disentangle/Mean_Face.obj',
            '../data/FWH/Tester_{}/shape_{}.obj'.format(j,i),
            index = np.loadtxt('front_part_v.txt',dtype=int))
            dr_feature = get_dr('../data/disentangle/Mean_Face.obj',
                '../data/FWH/Tester_{}/shape_{}.obj'.format(j,i))
            dr_feature.tofile('../data/FWH/Tester_{}/face_{}.dat'.format(j,i))

if __name__=="__main__":
    '''
    put raw facewarehouse data in ../data/FaceWarehouse_Data
    generate dr feature and aligned data in ../data/FWH
    '''
    generate_mean_face()
    generate_dr_feature()

    '''
    generate meanface.obj
    '''
    # pool = ThreadPool()
    # V = igl.eigen.MatrixXd()
    # F = igl.eigen.MatrixXi()
    # igl.readOBJ('../data/disentangle/Mean_Face.obj', V, F)
    # def process(i):
    #     return om.read_trimesh('../data/FWH/Mean_Face/shape_{}.obj'.format(i)).points()
    # gather_mesh = pool.map(process, range(47))
    # mean_mesh = igl.eigen.MatrixXd((sum(gather_mesh)/len(gather_mesh)).astype(np.float64))
    # igl.writeOBJ('../data/FWH/Mean_Face/Mean.obj', mean_mesh, F)