#ifndef MATH_LIBRARY_H
#define MATH_LIBRARY_H

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <array>
#include <float.h>

#ifdef M_PI
# undef M_PI
#endif
#define M_PI          (3.141592653589793)
#ifdef M_PI_2
# undef M_PI_2
#endif

#define M_PI_2        (M_PI / 2)
#define M_2PI         (M_PI * 2)

#define DEG_TO_RAD    (M_PI / 180.0)
#define RAD_TO_DEG    (180.0 / M_PI)

// Centi-degrees to radians
#define DEGX100         5729.57795

#define GRAVITY_MSS     9.80665

// radius of earth in meters
#define RADIUS_OF_EARTH 6378100

// convert a longitude or latitude point to meters or centimeters.
// Note: this does not include the longitude scaling which is dependent upon location
#define LATLON_TO_M     0.01113195
#define LATLON_TO_CM    1.113195

// Semi-major axis of the Earth, in meters.
static const double WGS84_A = 6378137.0;

//Inverse flattening of the Earth
static const double WGS84_IF = 298.257223563;

// The flattening of the Earth
static const double WGS84_F = ((double)1.0 / WGS84_IF);

// Semi-minor axis of the Earth in meters
static const double WGS84_B = (WGS84_A * (1 - WGS84_F));

// Eccentricity of the Earth
static const double WGS84_E = (sqrt(2 * WGS84_F - WGS84_F * WGS84_F));

// air density at 15C at sea level in kg/m^3
#define AIR_DENSITY_SEA_LEVEL    1.225

#define C_TO_KELVIN 273.15

// Gas Constant is from Aerodynamics for Engineering Students, Third Edition, E.L.Houghton and N.B.Carruthers
#define ISA_GAS_CONSTANT 287.26
#define ISA_LAPSE_RATE 0.0065

namespace math_lib
{
template <typename Arithmetic1, typename Arithmetic2>
typename std::enable_if<std::is_integral<typename std::common_type<Arithmetic1, Arithmetic2>::type>::value ,bool>::type
is_equal(const Arithmetic1 v_1, const Arithmetic2 v_2);

template <typename Arithmetic1, typename Arithmetic2>
typename std::enable_if<std::is_floating_point<typename std::common_type<Arithmetic1, Arithmetic2>::type>::value, bool>::type
is_equal(const Arithmetic1 v_1, const Arithmetic2 v_2);

template <typename T>
inline bool is_zero(const T fVal1) {
    return (fabsf(static_cast<float>(fVal1)) < FLT_EPSILON);
}

template <typename T>
inline bool is_positive(const T fVal1) {
    return (static_cast<float>(fVal1) >= FLT_EPSILON);
}

template <typename T>
inline bool is_negative(const T fVal1) {
    return (static_cast<float>(fVal1) <= (-1.0 * FLT_EPSILON));
}

template <typename T>
T constrain_value(const T amt, const T low, const T high);

inline double constrain_double(const double amt, const double low, const double high)
{
    return constrain_value(amt, low, high);
}

inline float constrain_float(const float amt, const float low, const float high)
{
    return constrain_value(amt, low, high);
}

inline int16_t constrain_int16(const int16_t amt, const int16_t low, const int16_t high)
{
    return constrain_value(amt, low, high);
}

inline int32_t constrain_int32(const int32_t amt, const int32_t low, const int32_t high)
{
    return constrain_value(amt, low, high);
}

// degrees -> radians
static inline constexpr double radians(double deg)
{
    return deg * DEG_TO_RAD;
}

// radians -> degrees
static inline constexpr double degrees(double rad)
{
    return rad * RAD_TO_DEG;
}

template <typename T>
float wrap_PI(const T radian);

template <typename T>
float wrap_2PI(const T radian);

template<typename T>
double sq(const T val)
{
    return pow(static_cast<double>(val), 2);
}

template<typename T, typename... Params>
double sq(const T first, const Params... parameters)
{
    return sq(first) + sq(parameters...);
}

template<typename T, typename U, typename... Params>
double norm(const T first, const U second, const Params... parameters)
{
    return sqrt(sq(first, second, parameters...));
}

Eigen::Vector4d quaternion_division(Eigen::Vector4d q1, Eigen::Vector4d q2);

Eigen::Matrix3d rotation_from_euler321(const Eigen::Vector3d &euler);

Eigen::Matrix3d rotation_from_euler312(const Eigen::Vector3d &euler);

inline void quaternionNormalize(Eigen::Vector4d& q) 
{
  double norm = q.norm();
  q = q / norm;
  return;
}

/*
 * @brief Perform q1 * q2
 */
inline Eigen::Vector4d quaternionMultiplication(const Eigen::Vector4d q1, const Eigen::Vector4d q2) 
{
    Eigen::Matrix4d L;
    L(0, 0) =  q1(3); L(0, 1) =  q1(2); L(0, 2) = -q1(1); L(0, 3) =  q1(0);
    L(1, 0) = -q1(2); L(1, 1) =  q1(3); L(1, 2) =  q1(0); L(1, 3) =  q1(1);
    L(2, 0) =  q1(1); L(2, 1) = -q1(0); L(2, 2) =  q1(3); L(2, 3) =  q1(2);
    L(3, 0) = -q1(0); L(3, 1) = -q1(1); L(3, 2) = -q1(2); L(3, 3) =  q1(3);

    Eigen::Vector4d q = L * q2;
    quaternionNormalize(q);
    return q;
}

/*
 * @brief Convert the vector part of a quaternion to a
 *    full quaternion.
 * @note This function is useful to convert delta quaternion
 *    which is usually a 3x1 vector to a full quaternion.
 *    For more details, check Section 3.2 "Kalman Filter Update" in
 *    "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for quaternion Algebra".
 */
inline Eigen::Vector4d smallAngleQuaternion(const Eigen::Vector3d& dtheta) 
{
    Eigen::Vector3d dq = dtheta / 2.0;
    Eigen::Vector4d q;
    double dq_square_norm = dq.squaredNorm();

    if (dq_square_norm <= 1) {
        q.head<3>() = dq;
        q(3) = std::sqrt(1-dq_square_norm);
    } else {
        q.head<3>() = dq;
        q(3) = 1;
        q = q / std::sqrt(1+dq_square_norm);
    }

    return q;
}

inline Eigen::Vector4d InverseQuaternion(const Eigen::Vector4d& q) 
{
    Eigen::Vector4d q1;
    
    q1(0) = -q(0);
    q1(1) = -q(1);
    q1(2) = -q(2);
    q1(3) =  q(3);

    return q1;
}

/*
 *  @brief Create a skew-symmetric matrix from a 3-element vector.
 *  @note Performs the operation:
 *  w   ->  [  0 -w3  w2]
 *          [ w3   0 -w1]
 *          [-w2  w1   0]
 */
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& w) 
{
    Eigen::Matrix3d w_hat;
    w_hat(0, 0) = 0;
    w_hat(0, 1) = -w(2);
    w_hat(0, 2) = w(1);
    w_hat(1, 0) = w(2);
    w_hat(1, 1) = 0;
    w_hat(1, 2) = -w(0);
    w_hat(2, 0) = -w(1);
    w_hat(2, 1) = w(0);
    w_hat(2, 2) = 0;
    return w_hat;
}

inline Eigen::Vector3d QuaternionToAxisAngle(const Eigen::Vector4d& q) 
{
    double l = std::sqrt((q(0)*q(0))+(q(1)*q(1))+(q(2)*q(2)));
    Eigen::Vector3d out = Eigen::Vector3d(q(0), q(1), q(2));
    if(abs(l) > DBL_EPSILON) {
        out /= l;
        out *= wrap_PI(2.0 * std::atan2(l,q(3)));
    }
    return out;
}

inline Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& q) 
{
    const Eigen::Vector3d& q_vec = q.block(0, 0, 3, 1);
    const double& q4 = q(3);
    Eigen::Matrix3d R =
        (2*q4*q4-1)*Eigen::Matrix3d::Identity() -
        2*q4*skewSymmetric(q_vec) +
        2*q_vec*q_vec.transpose();
    //TODO: Is it necessary to use the approximation equation
    //    (Equation (87)) when the rotation angle is small?
    return R;
}

inline Eigen::Vector4d rotationToQuaternion(const Eigen::Matrix3d& R) 
{
    Eigen::Vector4d score;
    score(0) = R(0, 0);
    score(1) = R(1, 1);
    score(2) = R(2, 2);
    score(3) = R.trace();

    int max_row = 0, max_col = 0;
    score.maxCoeff(&max_row, &max_col);

    Eigen::Vector4d q = Eigen::Vector4d::Zero();
    if (max_row == 0) {
        q(0) = std::sqrt(1+2*R(0, 0)-R.trace()) / 2.0;
        q(1) = (R(0, 1)+R(1, 0)) / (4*q(0));
        q(2) = (R(0, 2)+R(2, 0)) / (4*q(0));
        q(3) = (R(1, 2)-R(2, 1)) / (4*q(0));
    } else if (max_row == 1) {
        q(1) = std::sqrt(1+2*R(1, 1)-R.trace()) / 2.0;
        q(0) = (R(0, 1)+R(1, 0)) / (4*q(1));
        q(2) = (R(1, 2)+R(2, 1)) / (4*q(1));
        q(3) = (R(2, 0)-R(0, 2)) / (4*q(1));
    } else if (max_row == 2) {
        q(2) = std::sqrt(1+2*R(2, 2)-R.trace()) / 2.0;
        q(0) = (R(0, 2)+R(2, 0)) / (4*q(2));
        q(1) = (R(1, 2)+R(2, 1)) / (4*q(2));
        q(3) = (R(0, 1)-R(1, 0)) / (4*q(2));
    } else {
        q(3) = std::sqrt(1+R.trace()) / 2.0;
        q(0) = (R(1, 2)-R(2, 1)) / (4*q(3));
        q(1) = (R(2, 0)-R(0, 2)) / (4*q(3));
        q(2) = (R(0, 1)-R(1, 0)) / (4*q(3));
    }

    if (q(3) < 0) q = -q;
    quaternionNormalize(q);
    return q;
}

} // namespace math_lib

#endif // FRAME_TRANSFORMS_H
