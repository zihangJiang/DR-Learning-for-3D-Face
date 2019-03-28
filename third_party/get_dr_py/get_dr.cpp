#include "util_3drotation_log_exp.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>;

namespace py=pybind11;

struct TriTraits : public OpenMesh::DefaultTraits
{
  /// Use double precision points
  typedef OpenMesh::Vec3d Point;
  /// Use double precision Normals
  typedef OpenMesh::Vec3d Normal;
  /// Use double precision TexCood2D
  typedef OpenMesh::Vec2d TexCoord2D;

  /// Use RGBA Color
  typedef OpenMesh::Vec4f Color;

    /// Status
    VertexAttributes(OpenMesh::Attributes::Status);
    FaceAttributes(OpenMesh::Attributes::Status);
    EdgeAttributes(OpenMesh::Attributes::Status);

//    VertexAttributes(
//            OpenMesh::Attributes::
//            )
};

/// Simple Name for Mesh
typedef OpenMesh::TriMesh_ArrayKernelT<TriTraits>  TriMesh;

class DR_feature{
public:
    DR_feature();
	void read_ref_mesh(std::string ref_mesh_name);
	void read_defor_mesh(std::string defor_mesh_name);
	std::vector<double> get_feature(std::string ref_mesh_name, std::string defor_mesh_name);


private:
	TriMesh ref_mesh_;
	TriMesh defor_mesh_;

	OpenMesh::EPropHandleT<double> LB_weights;
	OpenMesh::VPropHandleT<Eigen::Matrix3d> T_matrixs;
private:
	void compute_ref_LB_weight();
    void compute_Ti(TriMesh::VertexHandle v_it,TriMesh::VertexHandle v_to_it);
};

DR_feature::DR_feature(){
	ref_mesh_.add_property(LB_weights);
	defor_mesh_.add_property(T_matrixs);
}

void DR_feature::read_ref_mesh(std::string ref_mesh_name){
	if(!OpenMesh::IO::read_mesh(ref_mesh_, ref_mesh_name)){
        std::cout<<"read_ref_mesh_ : read file wrong!!!"<<std::endl;
        return ;
    }

    compute_ref_LB_weight();
}

void DR_feature::compute_ref_LB_weight(){
	TriMesh::EdgeIter e_it, e_end(ref_mesh_.edges_end());
    TriMesh::HalfedgeHandle    h0, h1, h2;
    TriMesh::VertexHandle      v0, v1;
    TriMesh::Point             p0, p1, p2, d0, d1;
    TriMesh::Scalar w;
    for (e_it=ref_mesh_.edges_begin(); e_it!=e_end; e_it++)
    {
        w  = 0.0;
        if(ref_mesh_.is_boundary(*e_it))
        {
            h0 = ref_mesh_.halfedge_handle(e_it.handle(),0);
            if(ref_mesh_.is_boundary(h0))
                h0 = ref_mesh_.opposite_halfedge_handle(h0);

            v0 = ref_mesh_.to_vertex_handle(h0);
            v1 = ref_mesh_.from_vertex_handle(h0);
            p0 = ref_mesh_.point(v0);
            p1 = ref_mesh_.point(v1);
            h1 = ref_mesh_.next_halfedge_handle(h0);
            p2 = ref_mesh_.point(ref_mesh_.to_vertex_handle(h1));
            d0 = (p0-p2).normalize();
            d1 = (p1-p2).normalize();
            w += 2.0 / tan(acos(std::min(0.99, std::max(-0.99, (d0|d1)))));
            w = std::max(0.0, w);

            if(std::isnan(w))
                std::cout<<"Some weight NAN"<<std::endl;
            ref_mesh_.property(LB_weights,e_it) = w;
            continue;
        }
        h0 = ref_mesh_.halfedge_handle(e_it.handle(),0);
        v0 = ref_mesh_.to_vertex_handle(h0);
        p0 = ref_mesh_.point(v0);

        h1 = ref_mesh_.opposite_halfedge_handle(h0);
        v1 = ref_mesh_.to_vertex_handle(h1);
        p1 = ref_mesh_.point(v1);

        h2 = ref_mesh_.next_halfedge_handle(h0);
        p2 = ref_mesh_.point(ref_mesh_.to_vertex_handle(h2));
        d0 = (p0 - p2).normalize();
        d1 = (p1 - p2).normalize();
        w += 1.0/ tan(acos(std::max(-1.0, std::min(1.0, dot(d1,d0) ))));

        h2 = ref_mesh_.next_halfedge_handle(h1);
        p2 = ref_mesh_.point(ref_mesh_.to_vertex_handle(h2));
        d0 = (p0 - p2).normalize();
        d1 = (p1 - p2).normalize();
        w += 1.0 / tan(acos(std::max(-1.0, std::min(1.0, dot(d1,d0)))));

        if(std::isnan(w))
            std::cout<<"Some weight is NAN"<<std::endl;
        w = std::max(0.0, w);
        ref_mesh_.property(LB_weights,e_it) = w;
    }
}

void DR_feature::read_defor_mesh(std::string _filename){
	if(!OpenMesh::IO::read_mesh(defor_mesh_,_filename))
    {
        std::cout<<"read_defor_mesh : read file wrong"<<std::endl;
        return ;
    }

    TriMesh::FaceIter f_it = defor_mesh_.faces_begin();
    for(;f_it!=defor_mesh_.faces_end();f_it++)
    {
        TriMesh::FaceHandle f_h = *f_it;
        TriMesh::FaceVertexIter fe_it0,fe_it1;
        fe_it0 = ref_mesh_.fv_iter(f_h);
        fe_it1 = defor_mesh_.fv_iter(f_h);
        int v00,v01,v02,v10,v11,v12;
        v00 = (*fe_it0).idx();fe_it0++;
        v01 = (*fe_it0).idx();fe_it0++;
        v02 = (*fe_it0).idx();fe_it0++;

        v10 = (*fe_it1).idx();fe_it1++;
        v11 = (*fe_it1).idx();fe_it1++;
        v12 = (*fe_it1).idx();fe_it1++;

        if(v00 == v10)
        {
            if(v01!=v11||v02!=v12)
            {
                std::cout<<"defor and ref are not compatible!!!"<<std::endl;
                return;
            }
        }
        else if(v00 == v11)
        {
            if(v01!=v12||v02!=v10)
            {
                std::cout<<"defor and ref are not compatible!!!"<<std::endl;
                return;
            }
        }
        else if(v00 == v12)
        {
            if(v01!=v10||v02!=v11)
            {
                std::cout<<"defor and ref are not compatible!!!"<<std::endl;
                return;
            }
        }
        else
        {
            std::cout<<"defor and ref are not compatible!!!"<<std::endl;
            return;
        }
    }

    TriMesh::VertexIter v_it, v_to_it;
    for(v_it=ref_mesh_.vertices_begin(),v_to_it=defor_mesh_.vertices_begin()
        ;v_it!=ref_mesh_.vertices_end()&&v_to_it!=defor_mesh_.vertices_end()
        ;v_it++,v_to_it++)
    {
        if((*v_it).idx()!=(*v_to_it).idx())
            std::cout<<"DR_feature::compute_ref_to_defor_Tmatrixs different topology!!!"<<std::endl;
        compute_Ti(*v_it,*v_to_it);
    }

}

void DR_feature::compute_Ti(TriMesh::VertexHandle v_it, TriMesh::VertexHandle v_to_it){
    if(v_it.idx()!=v_to_it.idx())
    	std::cout<<"compute_Ti correspond is wrong!!!"<<std::endl;
    TriMesh::VertexEdgeIter veiter=ref_mesh_.ve_iter(v_it);
    int v_id = v_it.idx();
    TriMesh::Point p0,p1;
    p0 = ref_mesh_.point(v_it);
    p1 = defor_mesh_.point(v_to_it);
    Eigen::Matrix3d L,RI;
    L.setZero();
    RI.setZero();
    double tolerance = 1.0e-6;
    TriMesh::HalfedgeHandle h_e=ref_mesh_.halfedge_handle(v_it);
    TriMesh::VertexHandle test_v = ref_mesh_.to_vertex_handle(h_e);
    TriMesh::Point tp0,tp1;
    tp0 = ref_mesh_.point(v_it); tp1 = ref_mesh_.point(test_v);
    double scale=1.0;
    if(((tp0[0]-tp1[0])*(tp0[0]-tp1[0])+(tp0[1]-tp1[1])*(tp0[1]-tp1[1])+(tp0[2]-tp1[2])*(tp0[2]-tp1[2]))<0.1)
        scale = 100;

    for(;veiter.is_valid();veiter++)
    {
        double weight = 1.0;
        int to_id;
        TriMesh::VertexHandle to_v=ref_mesh_.to_vertex_handle(ref_mesh_.halfedge_handle(*veiter, 0));
        if(to_v.idx()==v_id)
            to_v = ref_mesh_.from_vertex_handle(ref_mesh_.halfedge_handle(*veiter, 0));
        to_id = to_v.idx();
        Eigen::Vector3d eij0,eij1;
        TriMesh::Point q0,q1;
        q0 = ref_mesh_.point(to_v);
        q1 = defor_mesh_.point(to_v);
        eij0(0) = p0[0]-q0[0];
        eij0(1) = p0[1]-q0[1];
        eij0(2) = p0[2]-q0[2];
        eij0*=weight*scale;

        eij1(0) = p1[0]-q1[0];
        eij1(1) = p1[1]-q1[1];
        eij1(2) = p1[2]-q1[2];
        eij1*=weight*scale;

        L+=eij1*eij0.transpose();
        RI+=eij0*eij0.transpose();
    }
    Eigen::Matrix3d T;
    if(fabs(RI.determinant())>tolerance)
         T = L*RI.inverse();
    else
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(RI, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3d U,V;
        U=svd.matrixU();
        V=svd.matrixV();
        Eigen::Matrix3d S_inv(svd.singularValues().asDiagonal());
        for(int i=0;i<3;i++)
        {
            if(fabs(S_inv(i,i))>tolerance)
                S_inv(i,i)=1.0/S_inv(i,i);
            else
                S_inv(i,i)=0.0;
        }
        T = L*V*S_inv*U.transpose();
    }
    defor_mesh_.property(T_matrixs,v_it) = T;
}


std::vector<double> DR_feature::get_feature(std::string ref_mesh_name, std::string defor_mesh_name){
	read_ref_mesh(ref_mesh_name);
	read_defor_mesh(defor_mesh_name);

	std::vector<double> dr_feature;

	TriMesh::VertexIter v_it;
    for(v_it=defor_mesh_.vertices_begin(); v_it!=defor_mesh_.vertices_end(); v_it++)
    {
        Eigen::Matrix3d T = defor_mesh_.property(T_matrixs,*v_it);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(T, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Matrix3d U,V;
        U=svd.matrixU();
        V=svd.matrixV();
        Eigen::Matrix3d S(svd.singularValues().asDiagonal());
        Eigen::Matrix3d Temp=Eigen::Matrix3d::Identity();
        Temp(2,2) = (U*V.transpose()).determinant();
        Eigen::Matrix3d R=U*Temp*V.transpose();
        Eigen::Matrix3d Scale = V*Temp*S*V.transpose();
        Eigen::Matrix3d logR = rotation_log_exp::log(R);

        dr_feature.push_back(Scale(0, 0));
        dr_feature.push_back(Scale(0, 1));
        dr_feature.push_back(Scale(0, 2));
        dr_feature.push_back(Scale(1, 1));
        dr_feature.push_back(Scale(1, 2));
        dr_feature.push_back(Scale(2, 2));
        dr_feature.push_back(logR(0, 1));
        dr_feature.push_back(logR(0, 2));
        dr_feature.push_back(logR(1, 2));
    }
    return dr_feature;
}

py::array_t<double> get_dr(std::string ref_mesh_name, std::string defor_mesh_name){
	DR_feature feature;
	std::vector<double> temp = feature.get_feature(ref_mesh_name, defor_mesh_name);

    auto result = py::array_t<double>(temp.size());
    auto result_buffer = result.request();
    double *result_ptr = (double *)result_buffer.ptr;

    std::memcpy(result_ptr, temp.data(), temp.size()*sizeof(double));

    return result;

}

 PYBIND11_MODULE(get_dr, m){
    m.doc() = "get dr";
    m.def("get_dr", &get_dr, "get_dr"); 
 }