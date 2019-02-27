# This file used to process mesh and return some useful data for measurement computation.
# Dependence: openmesh(python)
import openmesh as om
import pickle as pkl

class data_buffer(object):
	
	def __init__(self, filename):
		self.mesh = om.read_trimesh(filename)

	def halfedges_list(self):
		'''
		input: mesh
		output: halfedge list,
		'''
		self.halfedges=[]
		for he in self.mesh.halfedges():
			self.halfedges.append((self.mesh.from_vertex_handle(he).idx(),self.mesh.to_vertex_handle(he).idx()))
		return self.halfedges

	def v_vertex_list(self):
		'''
		input: mesh
		output: vertex_vertex_list
		'''
		self.vv_list=[]
		for v in self.mesh.vertices():
			n_v=[]
			for vv in self.mesh.vv(v):
				n_v.append(vv.idx())
			self.vv_list.append(n_v)
		return self.vv_list

	def edge_list(self):
		'''
		input: mesh
		output: edgelist
		'''
		self.edges=[]
		for e in self.mesh.edges():
			he=self.mesh.halfedge_handle(e, 0)
			self.edges.append((self.mesh.from_vertex_handle(he).idx(), self.mesh.to_vertex_handle(he).idx()))
		return self.edges

	def v_edge_list(self):
		'''
		return vetex_edge_list
		'''
		self.ve_list=[]
		for v in self.mesh.vertices():
			v_e=[]
			for ve in self.mesh.ve(v):
				v_e.append(ve.idx())
			self.ve_list.append(v_e)
		return self.ve_list

	def export_data(self, data, outfile):
		with open(outfile, 'wb') as f:
			pkl.dump(data, f)

	def import_data(self, inputfile):
		with open(inputfile, 'rb') as f:
			data = pkl.load(f)
		return data









