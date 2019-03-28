#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <string>
#include <iostream>
#include <vector>
//#include "include/pybind11/pybind11.h"
//#include "include/pybind11/numpy.h"
//#include "include/pybind11/stl.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <fstream>

struct TriTraits : public OpenMesh::DefaultTraits
{
  /// Use double precision points
  typedef OpenMesh::Vec3d Point;
  /// Use double precision Normals
  typedef OpenMesh::Vec3d Normal;
  /// Use double precision TexCood2D
  typedef OpenMesh::Vec2d TexCoord2D;
};

/// Simple Name for Mesh
typedef OpenMesh::TriMesh_ArrayKernelT<TriTraits>  TriMesh;

namespace py=pybind11;

// compute matrix exp
Eigen::Matrix3d exp(Eigen::Matrix3d angle_cross_axis){
        Eigen::Matrix3d test=angle_cross_axis+angle_cross_axis.transpose();
        if(test.norm()>1e-6)
        {
            std::cout<<"rotation_log_exp::exp3::input matrix isn't skew-matrix!!!"<<std::endl;
            return Eigen::Matrix3d::Identity();
        }

        Eigen::Vector3d angle_axis;
        angle_axis(0)=angle_cross_axis(2,1);
        angle_axis(1)=angle_cross_axis(0,2);
        angle_axis(2)=angle_cross_axis(1,0);
        double angle=sqrt(angle_axis(0)*angle_axis(0)+angle_axis(1)*angle_axis(1)+angle_axis(2)*angle_axis(2));
        if(angle<1e-6)
            return Eigen::Matrix3d::Identity();
        Eigen::Matrix3d cross_axis=angle_cross_axis/angle;

        test=cross_axis+cross_axis.transpose();
        if(test.norm()>1e-6)
        {
            std::cout<<"rotation_log_exp::exp2::input matrix isn't skew-matrix!!!"<<std::endl;
            return Eigen::Matrix3d::Identity();
        }

        return Eigen::Matrix3d::Identity()+sin(angle)*cross_axis+(1.0-cos(angle))*cross_axis*cross_axis;
}

py::array_t<double> get_mesh(std::string file, py::array_t<double> feature_input){
    TriMesh ref_mesh_;
    if(!OpenMesh::IO::read_mesh(ref_mesh_, file)){
        std::cerr<<"Read Mesh error"<<std::endl;
    }
    int nver=ref_mesh_.n_vertices();
    OpenMesh::EPropHandleT<double> LB_weights;
    ref_mesh_.add_property(LB_weights);

    // compute LB weight
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
        ref_mesh_.property(LB_weights,e_it) = w;
    }

    // prepare Sparse A_ matrix
    Eigen::SparseMatrix<double> A_;
    A_.resize(3*nver,3*nver);
    std::vector<Eigen::Triplet<double> > tripletlist;
    TriMesh::VertexIter v_it = ref_mesh_.vertices_begin();
    for(;v_it!=ref_mesh_.vertices_end();v_it++)
    {
        // fix one point
        TriMesh::VertexEdgeIter ve_iter = ref_mesh_.ve_iter(*v_it);
        int center_id = (*v_it).idx();
        double center_val[3]={0.0,0.0,0.0};
        for(;ve_iter.is_valid();ve_iter++)
        {
            double w = ref_mesh_.property(LB_weights,*ve_iter);
            TriMesh::VertexHandle to_v = ref_mesh_.to_vertex_handle(ref_mesh_.halfedge_handle(*ve_iter,0));
            if(to_v.idx() == center_id)
                to_v = ref_mesh_.from_vertex_handle(ref_mesh_.halfedge_handle(*ve_iter,0));
            for(int i =0;i<3;i++)
            {
                center_val[i]+=w;
                tripletlist.push_back(Eigen::Triplet<double>(3*center_id+i,3*to_v.idx()+i,-w));
//                outfile << 3*center_id + i << " " << 3*to_v.idx()+i << " "<< -w << '\n';
            }
        }
        for(int i =0;i<3;i++)
        {
            tripletlist.push_back(Eigen::Triplet<double>(3*center_id+i,3*center_id+i,center_val[i]));
        }
    }
    A_.setFromTriplets(tripletlist.begin(),tripletlist.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > A_solver_;
    A_solver_.compute(A_);
    // prepare T_matrix
    py::buffer_info feature_array = feature_input.request();
    std::vector<Eigen::Matrix3d> T_array;
    for(int i =0; i<ref_mesh_.n_vertices(); i++)
    {
        Eigen::Matrix3d logR;
        logR<<0,((double*)feature_array.ptr)[9*i+6],((double*)feature_array.ptr)[9*i+7],-((double*)feature_array.ptr)[9*i+6],0,((double*)feature_array.ptr)[9*i+8],-((double*)feature_array.ptr)[9*i+7],-((double*)feature_array.ptr)[9*i+8],0;
        Eigen::Matrix3d S;
        S<<((double*)feature_array.ptr)[9*i+0],((double*)feature_array.ptr)[9*i+1],((double*)feature_array.ptr)[9*i+2],((double*)feature_array.ptr)[9*i+1],((double*)feature_array.ptr)[9*i+3],((double*)feature_array.ptr)[9*i+4],((double*)feature_array.ptr)[9*i+2],((double*)feature_array.ptr)[9*i+4],((double*)feature_array.ptr)[9*i+5];
        T_array.push_back(exp(logR)*S);
    }

    // compute b_
    Eigen::VectorXd b_;
    b_.resize(3*nver);
    b_.setZero();
    v_it = ref_mesh_.vertices_begin();
    for(;v_it!=ref_mesh_.vertices_end();v_it++)
    {
        TriMesh::VertexEdgeIter ve_iter = ref_mesh_.ve_iter(*v_it);
        Eigen::Vector3d temp(0.0,0.0,0.0);
        for(;ve_iter.is_valid();ve_iter++)
        {
            TriMesh::VertexHandle v0 = ref_mesh_.to_vertex_handle(ref_mesh_.halfedge_handle(*ve_iter,0));
            if(v0.idx() == (*v_it).idx())
                v0 = ref_mesh_.from_vertex_handle(ref_mesh_.halfedge_handle(*ve_iter,0));
            OpenMesh::Vec3d Pj,Pk;
            Pj = ref_mesh_.point(*v_it);
            Pk = ref_mesh_.point(v0);
            Eigen::Vector3d e1jk(Pj[0]-Pk[0],Pj[1]-Pk[1],Pj[2]-Pk[2]);
            double cjk = ref_mesh_.property(LB_weights,*ve_iter);

            temp+= cjk*( T_array[v0.idx()] + T_array[(*v_it).idx()] )*e1jk;
        }
        temp*=0.5;
        b_.block<3,1>(3*(*v_it).idx(),0) = temp;
    }

    // solve 
    Eigen::VectorXd out;
    out.resize(3*nver);
    out = A_solver_.solve(b_);

    // std::vector<double> P_;
        // std::cout<<"dahsdas"<<std::endl;
    
    // std::memcpy(P_.data(), out.data(), out.size()*sizeof(double));
    // Eigen::VectorXd::Map(&P_[0], out.size()) = out;
    std::vector<double> P_(out.data(), out.data()+out.size());
    auto result = py::array_t<double>(P_.size());
    auto result_buffer = result.request();
    double *result_ptr = (double *)result_buffer.ptr;

    std::memcpy(result_ptr, P_.data(), P_.size()*sizeof(double));

    // std::cout<<result[0]<<std::endl;
    return result;

    // for(int i =0; i< P_.rows(); i++)
    // std::cout<<P_[i]<<std::endl;
    // std::vector<size_t> strides = {sizeof(double)};
    // std::vector<size_t> shape = {feature_array.shape[0]/3};
    // size_t ndim = 1;
    // return py::array(py::buffer_info(P_.data(), sizeof(double), py::format_descriptor<double>::value, ndim, feature_array.shape, strides));
    // return py::array(nver, P_.data());
}
 PYBIND11_MODULE(get_mesh, m){
    m.doc() = "get mesh";
    m.def("get_mesh", &get_mesh, "get_mesh"); 
 }
