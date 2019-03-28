#include "util_3drotation_log_exp.h"
#include <iostream>

Eigen::Matrix3d rotation_log_exp::exp(double angle, const Eigen::Vector3d &axis)
{
    if(fabs(axis.norm()-1.0)>log_exp_tolerance)
    {
        std::cout<<"rotation_log_exp::exp1::input axis isn't an unit direction!!!"<<std::endl;
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Matrix3d cross_axis,temp;
    temp.setZero();
    temp(2,1)=axis(0);
    temp(0,2)=axis(1);
    temp(1,0)=axis(2);
    cross_axis = temp - temp.transpose();
    return exp(angle,cross_axis);
}

Eigen::Matrix3d rotation_log_exp::exp(double angle, const Eigen::Matrix3d &cross_axis)
{
    Eigen::Matrix3d test=cross_axis+cross_axis.transpose();
    if(test.norm()>log_exp_tolerance)
    {
        std::cout<<"rotation_log_exp::exp2::input matrix isn't skew-matrix!!!"<<std::endl;
        return Eigen::Matrix3d::Identity();
    }

    return Eigen::Matrix3d::Identity()+sin(angle)*cross_axis+(1.0-cos(angle))*cross_axis*cross_axis;
}

Eigen::Matrix3d rotation_log_exp::exp(const Eigen::Matrix3d &angle_cross_axis)
{
    Eigen::Matrix3d test=angle_cross_axis+angle_cross_axis.transpose();
    if(test.norm()>log_exp_tolerance)
    {
        std::cout<<"rotation_log_exp::exp3::input matrix isn't skew-matrix!!!"<<std::endl;
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Vector3d angle_axis;
    angle_axis(0)=angle_cross_axis(2,1);
    angle_axis(1)=angle_cross_axis(0,2);
    angle_axis(2)=angle_cross_axis(1,0);
    double angle=sqrt(angle_axis(0)*angle_axis(0)+angle_axis(1)*angle_axis(1)+angle_axis(2)*angle_axis(2));
    if(angle<log_exp_tolerance)
        return Eigen::Matrix3d::Identity();
    Eigen::Matrix3d cross_axis=angle_cross_axis/angle;

    return exp(angle,cross_axis);
}

Eigen::Matrix3d rotation_log_exp::exp(const Eigen::Vector3d &angle_axis)
{
    double angle = sqrt(angle_axis(0)*angle_axis(0)+angle_axis(1)*angle_axis(1)+angle_axis(2)*angle_axis(2));
    if(angle<log_exp_tolerance)
    {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d axis = angle_axis/angle;

    Eigen::Matrix3d cross_axis,temp;
    temp.setZero();
    temp(2,1)=axis(0);
    temp(0,2)=axis(1);
    temp(1,0)=axis(2);
    cross_axis = temp - temp.transpose();
    return exp(angle,cross_axis);
}

void rotation_log_exp::log(const Eigen::Matrix3d &rotation_matrix, double &angle, Eigen::Vector3d &axis)
{
    if((rotation_matrix.transpose()*rotation_matrix-Eigen::Matrix3d::Identity()).norm()>log_exp_tolerance)
    {
        std::cout<<"rotation_log_exp::log::rotation_matrix isn't an orthogonal matrix!!!"<<std::endl;
        return;
    }
    double csin = (rotation_matrix.trace()-1.0)/2.0;
    if(csin<-1.0||csin>1.0)
    {
        if(fabs(csin-1.0)>log_exp_tolerance&&fabs(csin+1.0)>log_exp_tolerance)
        {
            std::cout<<"rotation_log_exp::log::csin "<<csin<<" range is wrong!!!"<<std::endl;
            return;
        }
        else
            csin=std::max(std::min(1.0,csin),-1.0);
    }
    double tangle = acos(csin);
    if(fabs(tangle)<log_exp_tolerance)
    {
        angle = 0.0;
        axis=Eigen::Vector3d::Zero();
        return;
    }
    if(fabs(tangle-M_PI)<log_exp_tolerance)//axis has 2 possible, return 1
    {
        angle = M_PI;
        Eigen::Matrix3d B = (rotation_matrix+Eigen::Matrix3d::Identity())/2.0;
        double k1 = sqrt(B(0,0));//here has 2 possible, + or -
        double k2,k3;
        if(k1*B(0,1)>0.0) k2 = sqrt(B(1,1));
        else k2 = -sqrt(B(1,1));
        if(k1*B(0,2)>0.0) k3 = sqrt(B(2,2));
        else k3 = -sqrt(B(2,2));
        axis(0)=k1;axis(1)=k2;axis(2)=k3;
        return ;
    }

    //tangle has 2 possible, 0~pi or pi~2pi
    Eigen::Vector3d taxis;
    taxis(0)=rotation_matrix(2,1)-rotation_matrix(1,2);
    taxis(1)=rotation_matrix(0,2)-rotation_matrix(2,0);
    taxis(2)=rotation_matrix(1,0)-rotation_matrix(0,1);
    Eigen::Vector3d t_axis=taxis/(2.0*sin(tangle));
    double sinv = sin(tangle);
    double r01 = (1.0-csin)*t_axis(0)*t_axis(1)-t_axis(2)*sinv;
    double r02 = (1.0-csin)*t_axis(0)*t_axis(2)+t_axis(1)*sinv;
    double r10 = (1.0-csin)*t_axis(0)*t_axis(1)+t_axis(2)*sinv;
    double r12 = (1.0-csin)*t_axis(1)*t_axis(2)-t_axis(0)*sinv;
    double r20 = (1.0-csin)*t_axis(0)*t_axis(2)-t_axis(1)*sinv;
    double r21 = (1.0-csin)*t_axis(1)*t_axis(2)+t_axis(0)*sinv;
    double check = (rotation_matrix(0,1)-r01)*(rotation_matrix(0,1)-r01)+(rotation_matrix(0,2)-r02)*(rotation_matrix(0,2)-r02)
            +(rotation_matrix(1,0)-r10)*(rotation_matrix(1,0)-r10)+(rotation_matrix(1,2)-r12)*(rotation_matrix(1,2)-r12)
            +(rotation_matrix(2,0)-r20)*(rotation_matrix(2,0)-r20)+(rotation_matrix(2,1)-r21)*(rotation_matrix(2,1)-r21);

    if(check<log_exp_tolerance)
    {
        angle = tangle;
        axis=t_axis;
        return;
    }
    else
    {
        std::cout<<"rotation_log_exp::log angle is large than PI"<<std::endl;
        tangle = 2*M_PI - tangle;
        t_axis = taxis/(2.0*sin(tangle));
        sinv = sin(tangle);
        r01 = (1.0-csin)*t_axis(0)*t_axis(1)-t_axis(2)*sinv;
        r02 = (1.0-csin)*t_axis(0)*t_axis(2)+t_axis(1)*sinv;
        r10 = (1.0-csin)*t_axis(0)*t_axis(1)+t_axis(2)*sinv;
        r12 = (1.0-csin)*t_axis(1)*t_axis(2)-t_axis(0)*sinv;
        r20 = (1.0-csin)*t_axis(0)*t_axis(2)-t_axis(1)*sinv;
        r21 = (1.0-csin)*t_axis(1)*t_axis(2)+t_axis(0)*sinv;
        check = (rotation_matrix(0,1)-r01)*(rotation_matrix(0,1)-r01)+(rotation_matrix(0,2)-r02)*(rotation_matrix(0,2)-r02)
                +(rotation_matrix(1,0)-r10)*(rotation_matrix(1,0)-r10)+(rotation_matrix(1,2)-r12)*(rotation_matrix(1,2)-r12)
                +(rotation_matrix(2,0)-r20)*(rotation_matrix(2,0)-r20)+(rotation_matrix(2,1)-r21)*(rotation_matrix(2,1)-r21);
        if(check>log_exp_tolerance)
            std::cout<<"rotation_log_exp::log::computation is wrong!!!"<<std::endl;
        angle = tangle;
        axis=t_axis;
        return;
    }
}

void rotation_log_exp::log(const Eigen::Matrix3d &rotation_matrix, double &angle, Eigen::Matrix3d &cross_axis)
{
    Eigen::Vector3d axis;
    log(rotation_matrix,angle,axis);

    Eigen::Matrix3d temp;
    temp.setZero();
    temp(2,1)=axis(0);
    temp(0,2)=axis(1);
    temp(1,0)=axis(2);
    cross_axis = temp - temp.transpose();
}

Eigen::Matrix3d rotation_log_exp::log(const Eigen::Matrix3d &rotation_matrix)
{
    double angle;
    Eigen::Matrix3d angle_cross_axis;
    log(rotation_matrix,angle,angle_cross_axis);
    return angle*angle_cross_axis;
}

Eigen::Matrix3d rotation_log_exp::rotation_so3(const Eigen::Vector3d &angle_axis)
{
    Eigen::Matrix3d cross_axis,temp;
    temp.setZero();
    temp(2,1)=angle_axis(0);
    temp(0,2)=angle_axis(1);
    temp(1,0)=angle_axis(2);
    cross_axis = temp - temp.transpose();
    return cross_axis;
}

Eigen::Matrix3d rotation_log_exp::rotation_so3(double angle, const Eigen::Vector3d &axis)
{
    if(fabs(axis.norm()-1.0)>log_exp_tolerance)
    {
        std::cout<<"rotation_log_exp::exp1::input axis isn't an unit direction!!!"<<std::endl;
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Matrix3d cross_axis,temp;
    temp.setZero();
    temp(2,1)=axis(0);
    temp(0,2)=axis(1);
    temp(1,0)=axis(2);
    cross_axis = temp - temp.transpose();
    cross_axis *= angle;
    return cross_axis;
}

void rotation_log_exp::so3_to_angle_axis(const Eigen::Matrix3d &r, double &angle, Eigen::Vector3d &axis)
{
    axis(0)=r(2,1);
    axis(1)=r(0,2);
    axis(2)=r(1,0);
    angle = sqrt(axis(0)*axis(0)+axis(1)*axis(1)+axis(2)*axis(2));
    if(angle<log_exp_tolerance)
        axis*=0.0;
    else
        axis/=angle;
}
