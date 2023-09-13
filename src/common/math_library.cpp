/**
 * @file frame_transform.cpp
 * @addtogroup lib
 * @author Nuno Marques <nuno.marques@dronesolutions.io>
 *
 * Adapted from MAVROS ftf_frame_conversions.cpp and ftf_quaternion_utils.cpp.
 */

#include <common/math_library.h>

#include <assert.h>

namespace math_lib
{

template <typename Arithmetic1, typename Arithmetic2>
typename std::enable_if<std::is_integral<typename std::common_type<Arithmetic1, Arithmetic2>::type>::value ,bool>::type
is_equal(const Arithmetic1 v_1, const Arithmetic2 v_2)
{
    typedef typename std::common_type<Arithmetic1, Arithmetic2>::type common_type;
    return static_cast<common_type>(v_1) == static_cast<common_type>(v_2);
}

/*
 * is_equal(): double/float implementation - takes into account
 * std::numeric_limits<T>::epsilon() to return if 2 values are equal.
 */
template <typename Arithmetic1, typename Arithmetic2>
typename std::enable_if<std::is_floating_point<typename std::common_type<Arithmetic1, Arithmetic2>::type>::value, bool>::type
is_equal(const Arithmetic1 v_1, const Arithmetic2 v_2)
{
    // typedef typename std::common_type<Arithmetic1, Arithmetic2>::type common_type;
    // typedef typename std::remove_cv<common_type>::type common_type_nonconst;

    return fabsf(v_1 - v_2) < std::numeric_limits<float>::epsilon();
}

template bool is_equal<int>(const int v_1, const int v_2);
template bool is_equal<short>(const short v_1, const short v_2);
template bool is_equal<long>(const long v_1, const long v_2);
template bool is_equal<float>(const float v_1, const float v_2);
template bool is_equal<double>(const double v_1, const double v_2);

template <typename T>
T constrain_value(const T amt, const T low, const T high)
{
    // the check for NaN as a float prevents propagation of floating point
    // errors through any function that uses constrain_value(). The normal
    // float semantics already handle -Inf and +Inf
    if (std::isnan(amt)) {
        return (low + high) / 2;
    }

    if (amt < low) {
        return low;
    }

    if (amt > high) {
        return high;
    }

    return amt;
}

template int constrain_value<int>(const int amt, const int low, const int high);
template long constrain_value<long>(const long amt, const long low, const long high);
template short constrain_value<short>(const short amt, const short low, const short high);
template float constrain_value<float>(const float amt, const float low, const float high);
template double constrain_value<double>(const double amt, const double low, const double high);

template <typename T>
float wrap_PI(const T radian)
{
    auto res = wrap_2PI(radian);
    if (res > M_PI) {
        res -= M_2PI;
    }
    return res;
}

template float wrap_PI<int>(const int radian);
template float wrap_PI<short>(const short radian);
template float wrap_PI<float>(const float radian);
template float wrap_PI<double>(const double radian);

template <typename T>
float wrap_2PI(const T radian)
{
    double res = fmodf(static_cast<float>(radian), M_2PI);
    if (res < 0) {
        res += M_2PI;
    }
    return res;
}

template float wrap_2PI<int>(const int radian);
template float wrap_2PI<short>(const short radian);
template float wrap_2PI<float>(const float radian);
template float wrap_2PI<double>(const double radian);

Eigen::Vector4d quaternion_division(Eigen::Vector4d q1, Eigen::Vector4d q2)
{
    Eigen::Vector4d ret;
    const double &quat0 = q1(3);
    const double &quat1 = q1(0);
    const double &quat2 = q1(1);
    const double &quat3 = q1(2);

    double rquat0 = q2(3);
    double rquat1 = q2(0);
    double rquat2 = q2(1);
    double rquat3 = q2(2);

    ret(3) = (rquat0*quat0 + rquat1*quat1 + rquat2*quat2 + rquat3*quat3);
    ret(0) = (rquat0*quat1 - rquat1*quat0 - rquat2*quat3 + rquat3*quat2);
    ret(1) = (rquat0*quat2 + rquat1*quat3 - rquat2*quat0 - rquat3*quat1);
    ret(2) = (rquat0*quat3 - rquat1*quat2 + rquat2*quat1 - rquat3*quat0);
    return ret;
}

Eigen::Matrix3d rotation_from_euler321(const Eigen::Vector3d &euler)
{
    Eigen::Matrix3d dcm;

    dcm = Eigen::AngleAxisd(euler.z(), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(euler.y(), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(euler.x(), Eigen::Vector3d::UnitX());

    return dcm;
}

Eigen::Matrix3d rotation_from_euler312(const Eigen::Vector3d &euler)
{
    Eigen::Matrix3d dcm;

    dcm = Eigen::AngleAxisd(euler.z(), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(euler.x(), Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(euler.y(), Eigen::Vector3d::UnitY());

    return dcm;
}

} // namespace math_lib
