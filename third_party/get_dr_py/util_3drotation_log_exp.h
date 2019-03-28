#ifndef UTIL_3DROTATION_LOG_EXP
#define UTIL_3DROTATION_LOG_EXP
#include <Eigen/Dense>


//rotation matrix log and exp operation, reference Axis–angle representation on wiki
namespace rotation_log_exp
{
static double log_exp_tolerance = 1.0e-6;
Eigen::Matrix3d exp(const Eigen::Vector3d &angle_axis);
Eigen::Matrix3d exp(double angle,const Eigen::Vector3d &axis);
Eigen::Matrix3d exp(double angle,const Eigen::Matrix3d &cross_axis);
Eigen::Matrix3d exp(const Eigen::Matrix3d &angle_cross_axis);

void log(const Eigen::Matrix3d &rotation_matrix,double &angle,Eigen::Vector3d &axis);
void log(const Eigen::Matrix3d &rotation_matrix,double &angle,Eigen::Matrix3d &cross_axis);
Eigen::Matrix3d log(const Eigen::Matrix3d &rotation_matrix);

Eigen::Matrix3d rotation_so3(const Eigen::Vector3d &angle_axis);
Eigen::Matrix3d rotation_so3(double angle,const Eigen::Vector3d &axis);

void so3_to_angle_axis(const Eigen::Matrix3d &r, double &angle, Eigen::Vector3d &axis);
}


#endif // ROTATION_H
