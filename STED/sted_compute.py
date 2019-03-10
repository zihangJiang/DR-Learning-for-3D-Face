'''
This file include some functions for STED distance computation
'''

import numpy as np 
import pickle
import math
import time
import openmesh as om
with open('edgelist.pkl', 'rb') as f:
	edge_list = pickle.load(f)
with open('velist.pkl', 'rb') as f:
	vertex_edge_list = pickle.load(f)
def sted_compute(src_point_array, tar_point_array):
	'''
	point_array: store the coordinate of point, size should be nver*3
		-- src_point_array: source point array / origin mesh
		-- tar_point_array: target point array / distorted mesh
	'''
	with open('edgelist.pkl', 'rb') as f:
		edge_list = pickle.load(f)
	src_el=[]
	tar_el=[]
	t = time.time()
	# compute edge length of src_mesh and tar_mesh.
	# This part should be accerated
	src_el = np.array([np.sqrt(np.sum(np.square(src_point_array[ele[0], :]-src_point_array[ele[1], :]))) for ele in edge_list])
	tar_el = np.array([np.sqrt(np.sum(np.square(tar_point_array[ele[0], :]-tar_point_array[ele[1], :]))) for ele in edge_list])
	# for ele in edge_list: 
	# 	src_el.append(np.sqrt(np.sum(np.square(src_point_array[ele[0], :]-src_point_array[ele[1], :]))))
	# 	tar_el.append(np.sqrt(np.sum(np.square(tar_point_array[ele[0], :]-tar_point_array[ele[1], :]))))
	# src_el= np.array(src_el)
	# tar_el= np.array(tar_el)
	# print('edge time cost {}'.format(time.time()-t))
	
	# compute relative edge difference, ed
	ed = np.abs(src_el-tar_el)/src_el

	# compute weights of edge
	with open('velist.pkl', 'rb') as f:
		vertex_edge_list = pickle.load(f)

	dev=0
	for ve in vertex_edge_list:
		weight_array = src_el[ve]
		sum_el = np.sum(weight_array)
		weight_array = weight_array/sum_el
		sub_ed_array = ed[ve]
		avged = np.average(sub_ed_array, weights=weight_array)
		vared = np.square(sub_ed_array-avged)
		dev= dev+math.sqrt(np.average(vared, weights= weight_array))

	return dev/src_point_array.shape[0]


def sted_compute_advanced_back(src_point_array, tar_point_array):
	'''
	point_array: store the coordinate of point, size should be nver*3
		-- src_point_array: source point array / origin mesh
		-- tar_point_array: target point array / distorted mesh
	'''
#	with open('edgelist.pkl', 'rb') as f:
#		edge_list = pickle.load(f)
	src_el=[]
	tar_el=[]
	# compute edge length of src_mesh and tar_mesh.
	# This part should be accerated
	src_el = np.array([np.sqrt(np.sum(np.square(src_point_array[ele[0], :]-src_point_array[ele[1], :]))) for ele in edge_list])
	tar_el = np.array([np.sqrt(np.sum(np.square(tar_point_array[ele[0], :]-tar_point_array[ele[1], :]))) for ele in edge_list])
	# for ele in edge_list: 
	# 	src_el.append(np.sqrt(np.sum(np.square(src_point_array[ele[0], :]-src_point_array[ele[1], :]))))
	# 	tar_el.append(np.sqrt(np.sum(np.square(tar_point_array[ele[0], :]-tar_point_array[ele[1], :]))))
	# src_el= np.array(src_el)
	# tar_el= np.array(tar_el)
	# print('edge time cost {}'.format(time.time()-t))
	
	# compute relative edge difference, ed
	ed = np.abs(src_el-tar_el)/src_el

	# compute weights of edge
#	with open('velist.pkl', 'rb') as f:
#		vertex_edge_list = pickle.load(f)

	dev=0
	for ve in vertex_edge_list:
		weight_array = src_el[ve]
		sum_el = np.sum(weight_array)
		weight_array = weight_array/sum_el
		sub_ed_array = ed[ve]
		avged = np.average(sub_ed_array, weights=weight_array)
		vared = np.square(sub_ed_array-avged)
		dev= dev+math.sqrt(np.average(vared, weights= weight_array))

	return dev/src_point_array.shape[0]


def sted_compute_advanced(src_point_array, tar_point_array):
    '''
        point_array: store the coordinate of point, size should be nver*3
        -- src_point_array: source point array / origin mesh
        -- tar_point_array: target point array / distorted mesh
    '''
#	with open('edgelist.pkl', 'rb') as f:
#		edge_list = pickle.load(f)
    src_el=[]
    tar_el=[]
    # compute edge length of src_mesh and tar_mesh.
    # This part should be accerated
    src_el = np.array([np.sqrt(np.sum(np.square(src_point_array[ele[0], :]-src_point_array[ele[1], :]))) for ele in edge_list])
    tar_el = np.array([np.sqrt(np.sum(np.square(tar_point_array[ele[0], :]-tar_point_array[ele[1], :]))) for ele in edge_list])
    # for ele in edge_list: 
    # 	src_el.append(np.sqrt(np.sum(np.square(src_point_array[ele[0], :]-src_point_array[ele[1], :]))))
    # 	tar_el.append(np.sqrt(np.sum(np.square(tar_point_array[ele[0], :]-tar_point_array[ele[1], :]))))
    # src_el= np.array(src_el)
    # tar_el= np.array(tar_el)
    # print('edge time cost {}'.format(time.time()-t))
    	
    per_vertex_sted=[]
	# compute relative edge difference, ed
    ed = np.abs(src_el-tar_el)/src_el

	# compute weights of edge
#	with open('velist.pkl', 'rb') as f:
#		vertex_edge_list = pickle.load(f)

    dev=0
    for ve in vertex_edge_list:
        weight_array = src_el[ve]
        sum_el = np.sum(weight_array)
        weight_array = weight_array/sum_el
        sub_ed_array = ed[ve]
        avged = np.average(sub_ed_array, weights=weight_array)
        vared = np.square(sub_ed_array-avged)
        vertex_sted=math.sqrt(np.average(vared, weights= weight_array))
        per_vertex_sted.append(vertex_sted)
        #dev= dev+math.sqrt(np.average(vared, weights= weight_array))
        dev= dev+vertex_sted

    return dev/src_point_array.shape[0], per_vertex_sted


def cal_sted_loss_in_file(tar_file_format, src_file_format = '/raid/jzh/CVPR2019/alignpose/Tester_{}/AlignPose/pose_{}.obj',vis = False):
    average_p_loss = []
    for j in range(141,151):
        for i in range(47):
            tar_mesh = om.read_trimesh(tar_file_format.format(j,i))

            src_mesh = om.read_trimesh(src_file_format.format(j,i))
            src_point=src_mesh.points()
            tar_point=tar_mesh.points()
            p_loss, _ = sted_compute_advanced(src_point, tar_point)
            #print(np.array(sted_array).astype(np.float64)[:3])
            #np.savetxt((tar_file_format[:-4]+'.txt').format(j,i), sted_array)
            average_p_loss.append(p_loss)
            if vis:
                print('people:{} exp: {}, STED {:9.6f}'.format(j, i, p_loss))
    print('Average loss: {}'.format(np.mean(average_p_loss)))
    print('median loss: {}'.format(np.median(average_p_loss)))
    print('extreme differ: {}'.format(np.max(np.abs(np.array(average_p_loss) - np.mean(average_p_loss)))))
    return average_p_loss


