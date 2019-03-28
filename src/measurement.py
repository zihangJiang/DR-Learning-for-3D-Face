## this file used to compute some measurement 
import sys
import os

import numpy as np
import openmesh as om
try:
    import pyigl as igl
except:
    import src.pyigl as igl
# rigid registeration of 2 meshes
def rigid_registeration(src_array, tar_array, index = None):
	'''
	apply rigid registeration between 2 meshes.
	'''
	if index is not None:
		src_array = src_array[index]
		tar_array = tar_array[index]
	M = np.matmul((src_array - np.mean(src_array, axis=0)).T, (tar_array - np.mean(tar_array, axis=0)))
	u,s,v= np.linalg.svd(M)
	# print(np.dot(np.dot(u,np.diag(s)), v))
	sig = np.ones(s.shape)
	sig[-1] = np.linalg.det(np.dot(u,v))
	R = np.matmul(np.matmul(v.T, np.diag(sig)), u.T)
	t = np.mean(tar_array, axis=0) - np.matmul(R,np.mean(src_array, axis=0))
	return R, t

def compute_distance_whole(src, tar, index):
	'''
	compute the L2 distance of average distance between 2 meshes on each vertice
	'''
	src_mesh = om.read_trimesh(src)
	tar_mesh = om.read_trimesh(tar)
	point_array_src = src_mesh.points()
	point_array_tar = tar_mesh.points()
	R, t = rigid_registeration(point_array_src, point_array_tar, index)
	register_src = np.dot(R, point_array_src.T).T + np.tile(t,point_array_src.shape[0]).reshape(-1,3)
	# distance = np.mean(np.square(point_array_tar - register_src ))
	distance_mean = np.mean(np.sqrt(np.sum(np.square(point_array_tar-register_src),axis=1)))
	distance_max = np.max(np.sqrt(np.sum(np.square(point_array_tar-register_src),axis=1)))
	distance_no_rigid = np.mean(np.sqrt(np.sum(np.square(point_array_tar-point_array_src),axis=1)))
	return distance_mean, distance_max,distance_no_rigid

def compute_distance(src, tar, index = None):#np.loadtxt('src/front_part_v.txt', dtype=int)):
	'''
	compute the L2 distance of average distance between 2 meshes on each vertice
	'''
	src_mesh = om.read_trimesh(src)
	tar_mesh = om.read_trimesh(tar)
	point_array_src = src_mesh.points()
	point_array_tar = tar_mesh.points()
	R, t = rigid_registeration(point_array_src, point_array_tar, index)
	register_src = np.dot(R, point_array_src.T).T + np.tile(t,point_array_src.shape[0]).reshape(-1,3)
	# distance = np.mean(np.square(point_array_tar - register_src ))\

 	## new way for front face distance computation
	diff_array = (point_array_tar-register_src)[index]
	#print(np.shape(diff_array))
	distance_mean = np.mean(np.sqrt(np.sum(np.square(diff_array),axis=1)))
	distance_max = np.max(np.sqrt(np.sum(np.square(diff_array),axis=1)))
	distance_no_rigid = np.mean(np.sqrt(np.sum(np.square(diff_array),axis=1)))
	return distance_mean, distance_max,distance_no_rigid

def compute_variance(data_array):
	'''
	compute variance of data_array, Var(x) = sqrt(E(x-E(x))^2))
	data_array size: feature_num * feature_dim 
	'''
	return np.mean(np.std(data_array, axis=0))


def write_align_mesh(src, tar, filename, index = None):
	src_mesh=om.read_trimesh(src)
	tar_mesh=om.read_trimesh(tar)
	point_array_src = src_mesh.points()
	point_array_tar = tar_mesh.points()
	R, t = rigid_registeration(point_array_src, point_array_tar, index)
	register_array_src = np.dot(R, point_array_src.T).T + np.tile(t,point_array_src.shape[0]).reshape(-1,3)
	new_V = igl.eigen.MatrixXd(register_array_src.astype(np.float64))
	V = igl.eigen.MatrixXd()
	F = igl.eigen.MatrixXi()
	igl.readOBJ(src, V,F)
	igl.writeOBJ(filename, new_V, F)
	# om.write_mesh(filename, src_mesh)

def cal_distance_in_file(src_file_format, tar_file_format, vis = False):
    avg_dis=[]
    index=np.loadtxt('src/front_part_v.txt',dtype=int)
    for i in range(141,151):
        for j in range(47):
            if j in []:
                continue
            dis,_,_ = compute_distance(src_file_format.format(i, j), tar_file_format.format(i, j), index)
            avg_dis.append(dis)
            if vis:
                print('Loading Mesh: {}, Expression: {}, vertice distacnce: {}'.format(i,j, dis))

    
    print('Avg distance: {}'.format(np.mean(avg_dis)))
    print('extreme differ: {}'.format(np.max(np.abs(avg_dis - np.mean(avg_dis)))))
    print('median distance: {}'.format(np.median(avg_dis)))
    print('Max distance: {}'.format(np.max(avg_dis)))
    return avg_dis

def cal_id_disentanglement_in_file(file_format, use_registeration = True, vis = True):
    index = None
    loss_log = []
    for i in range(141,151):
        data_array=[]
        for j in range(0, 47):
            mesh=om.read_trimesh(file_format.format(i,j))
            point_array=mesh.points()
            if use_registeration:
                if j > 0:
                    R, t = rigid_registeration(point_array, fix, index)
                    register_src = np.dot(R, point_array.T).T + np.tile(t,point_array.shape[0]).reshape(-1,3)
                    point_array = register_src
                else:
                    fix = point_array
            data_array.append(point_array.reshape(1,-1))
        data_array=np.concatenate(data_array,axis=0)
        # store id std to a file
        dis = np.sqrt(np.sum(np.square(np.std(data_array.reshape((47,11510,3)), axis = 0)),axis = -1))
        #dis.tofile('id_var_for_people_{}.dat'.format(i))
        loss_log.append(np.mean(dis))
        #loss_log.append(compute_variance(data_array))
        if vis:
            print('Exp: {}, var: {}'.format(i, loss_log[-1]))
    print('our id Average variance: {}'.format(np.mean(loss_log)))
    print('our id Median variance: {}'.format(np.median(loss_log)))
    print('extreme var: {}'.format(max(np.max(loss_log)-np.mean(loss_log), np.mean(loss_log)- np.min(loss_log))))
    return loss_log

def cal_exp_disentanglement_in_file(file_format, use_registeration = True, vis = True):
    index = None
    loss_log = []
    for j in range(0, 47):
        data_array=[]
        for i in range(141,151):
            mesh=om.read_trimesh(file_format.format(i,j))
            point_array=mesh.points()
            if use_registeration:
                if j > 0:
                    R, t = rigid_registeration(point_array, fix, index)
                    register_src = np.dot(R, point_array.T).T + np.tile(t,point_array.shape[0]).reshape(-1,3)
                    point_array = register_src
                else:
                    fix = point_array
            data_array.append(point_array.reshape(1,-1))
        data_array=np.concatenate(data_array,axis=0)
        # store exp std to a file
        dis = np.sqrt(np.sum(np.square(np.std(data_array.reshape((10,11510,3)), axis = 0)),axis = -1))
        #dis.tofile('exp_var_for_expression_{}.dat'.format(j))
        loss_log.append(np.mean(dis))
        #loss_log.append(compute_variance(data_array))
        if vis:
            print('Exp: {}, var: {}'.format(j, loss_log[-1]))
    print('our exp Average variance: {}'.format(np.mean(loss_log)))
    print('our exp Median variance: {}'.format(np.median(loss_log)))
    print('extreme var: {}'.format(max(np.max(loss_log)-np.mean(loss_log), np.mean(loss_log)- np.min(loss_log))))
    return loss_log

if __name__ == '__main__':
    avg_dis=[]
    count=0
    meshpath_root = '/home/jzh/coma_mesh'
    test_root = '/home/jzh/coma_mesh'
    
    index=None
    for i in range(12):
        for j in range(1,len(os.listdir(os.path.join(meshpath_root,'ori/Feature{}/'.format(i))))):
            tar_path=os.path.join(meshpath_root,'ori/Feature{}/{}.obj'.format(i,j))
            src_path=os.path.join(test_root,'rec/Feature{}/{}.obj'.format(i,j))
            #print(src_path)
            #print(tar_path)
            dis,_,_ = compute_distance_whole(src_path, tar_path, index)
            avg_dis.append(dis)
            print('Loading Mesh: {}, Expression: {}, vertice distacnce: {}'.format(i,j, dis))
            count=count+1
            
    print('Avg distance: {}'.format(np.mean(avg_dis)))
    print('median distance: {}'.format(np.median(avg_dis)))
    print('extreme var: {}'.format(np.max(np.abs(avg_dis - np.mean(avg_dis)))))

