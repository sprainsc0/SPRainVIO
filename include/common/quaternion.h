#pragma once

#include <cmath>
#include <Eigen/Eigen>
#include <Eigen/Geometry>

namespace math_lib
{
class Quaternion {
public:
    double        qx, qy, qz, qw;

    // constructor creates a quaternion equivalent
    // to roll=0, pitch=0, yaw=0
    Quaternion()
    {
        qw = 1;
        qx = qy = qz = 0;
    }

    // setting constructor
    Quaternion(const double _q1, const double _q2, const double _q3, const double _q4) :
        qx(_q2), qy(_q3), qz(_q4), qw(_q1)
    {
    }

    // function call operator
    void operator()(const double _q1, const double _q2, const double _q3, const double _q4)
    {
        qx = _q2;
        qy = _q3;
        qz = _q4;
        qw = _q1;
    }

    // check if any elements are NAN
    bool        is_nan(void) const
    {
        return std::isnan(qw) || std::isnan(qx) || std::isnan(qy) || std::isnan(qz);
    }

    // return the rotation matrix equivalent for this quaternion
    Eigen::Matrix3d rotation_matrix(void) const;

    void		from_rotation_matrix(const Eigen::Matrix3d &R);

    // create a quaternion from Euler angles
    void        from_euler(double roll, double pitch, double yaw);

    void        from_vector312(double roll ,double pitch, double yaw);

    void to_axis_angle(Eigen::Vector3d &v);

    void from_axis_angle(Eigen::Vector3d v);

    void from_axis_angle(const Eigen::Vector3d &axis, double theta);

    void rotate(const Eigen::Vector3d &v);

    void from_axis_angle_fast(Eigen::Vector3d v);

    void from_axis_angle_fast(const Eigen::Vector3d &axis, double theta);

    void rotate_fast(const Eigen::Vector3d &dtheta);

    // get euler roll angle
    double       get_euler_roll() const;

    // get euler pitch angle
    double       get_euler_pitch() const;

    // get euler yaw angle
    double       get_euler_yaw() const;

    // create eulers from a quaternion
    void        to_euler(double &roll, double &pitch, double &yaw) const;

    // create eulers from a quaternion
    Eigen::Vector3d    to_vector312(void) const;
    Eigen::Vector3d    to_vector321(void) const;

    Quaternion get_rp_quaternion(void) const;

    double norm(void) const;
    void normalize();

    // initialise the quaternion to no rotation
    void initialise()
    {
        qw = 1.0f;
        qx = qy = qz = 0.0f;
    }

    Quaternion inverse(void) const;

    // allow a quaternion to be used as an array, 0 indexed
    double & operator()(uint8_t i)
    {
        double *_v = &qx;
#if MATH_CHECK_INDEXES
        assert(i < 4);
#endif
        return _v[i];
    }

    const double & operator()(uint8_t i) const
    {
        const double *_v = &qx;
#if MATH_CHECK_INDEXES
        assert(i < 4);
#endif
        return _v[i];
    }

    Quaternion operator*(const Quaternion &v) const;
    Quaternion &operator*=(const Quaternion &v);
    Quaternion operator/(const Quaternion &v) const;
};
}
