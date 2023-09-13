#include <common/quaternion.h>
#include <common/math_library.h>

namespace math_lib
{
// return the rotation matrix equivalent for this quaternion
Eigen::Matrix3d Quaternion::rotation_matrix(void) const
{
    const Eigen::Vector3d q_vec(qx,qy,qz);
    const double q4 = qw;
    Eigen::Matrix3d R =
        (2*q4*q4-1)*Eigen::Matrix3d::Identity() - 2*q4*skewSymmetric(q_vec) + 2*q_vec*q_vec.transpose();
    return R;
}

void Quaternion::from_rotation_matrix(const Eigen::Matrix3d &R)
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

    if (q(3) < 0) {
        q = -q;
    }
    this->qx = q(0);
    this->qy = q(1);
    this->qz = q(2);
    this->qw = q(3);
    normalize();
}

// create a quaternion from Euler angles
void Quaternion::from_euler(double roll, double pitch, double yaw)
{
    double cr2 = cos(roll*0.5);
    double cp2 = cos(pitch*0.5);
    double cy2 = cos(yaw*0.5);
    double sr2 = sin(roll*0.5);
    double sp2 = sin(pitch*0.5);
    double sy2 = sin(yaw*0.5);

    qw = cr2*cp2*cy2 + sr2*sp2*sy2;
    qx = sr2*cp2*cy2 - cr2*sp2*sy2;
    qy = cr2*sp2*cy2 + sr2*cp2*sy2;
    qz = cr2*cp2*sy2 - sr2*sp2*cy2;
}

// create a quaternion from Euler angles
void Quaternion::from_vector312(double roll ,double pitch, double yaw)
{
    Eigen::Matrix3d m;

    double c3 = cos(pitch);
    double s3 = sin(pitch);
    double s2 = sin(roll);
    double c2 = cos(roll);
    double s1 = sin(yaw);
    double c1 = cos(yaw);

    m(0,0) = c1 * c3 - s1 * s2 * s3;
    m(0,1) = -c2*s1;
    m(0,2) = s3*c1 + c3*s2*s1;
    m(1,0) = c3*s1 + s3*s2*c1;
    m(1,1) = c1 * c2;
    m(1,2) = s1*s3 - s2*c1*c3;
    m(2,0) = -s3*c2;
    m(2,1) = s2;
    m(2,2) = c3 * c2;

    from_rotation_matrix(m);
}

void Quaternion::from_axis_angle(Eigen::Vector3d v)
{
    double theta = v.norm();
    if (is_zero(theta)) {
        qw = 1.0;
        qx=qy=qz=0.0;
        return;
    }
    v /= theta;
    from_axis_angle(v,theta);
}

void Quaternion::from_axis_angle(const Eigen::Vector3d &axis, double theta)
{
    // axis must be a unit vector as there is no check for length
    if (is_zero(theta)) {
        qw = 1.0;
        qx=qy=qz=0.0;
        return;
    }
    double st2 = sinf(theta/2.0);

    qw = cosf(theta/2.0);
    qx = axis.x() * st2;
    qy = axis.y() * st2;
    qz = axis.z() * st2;
}

void Quaternion::rotate(const Eigen::Vector3d &v)
{
    Quaternion r;
    r.from_axis_angle(v);
    (*this) *= r;
}

void Quaternion::to_axis_angle(Eigen::Vector3d &v)
{
    double l = sqrt(sq(qx)+sq(qy)+sq(qz));
    v = Eigen::Vector3d(qx,qy,qz);
    if (!is_zero(l)) {
        v /= l;
        v *= wrap_PI(2.0 * atan2(l,qw));
    }
}

void Quaternion::from_axis_angle_fast(Eigen::Vector3d v)
{
    double theta = v.norm();
    if (is_zero(theta)) {
        qw = 1.0;
        qx=qy=qz=0.0;
        return;
    }
    v /= theta;
    from_axis_angle_fast(v,theta);
}

void Quaternion::from_axis_angle_fast(const Eigen::Vector3d &axis, double theta)
{
    double t2 = theta/2.0;
    double sqt2 = sq(t2);
    double st2 = t2-sqt2*t2/6.0;

    qw = 1.0f-(sqt2/2.0)+sq(sqt2)/24.0;
    qx = axis.x() * st2;
    qy = axis.y() * st2;
    qz = axis.z() * st2;
}

void Quaternion::rotate_fast(const Eigen::Vector3d &dtheta)
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
    this->qx = q(0);
    this->qy = q(1);
    this->qz = q(2);
    this->qw = q(3);
}

// get euler roll angle
double Quaternion::get_euler_roll() const
{
    return (atan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy)));
}

// get euler pitch angle
double Quaternion::get_euler_pitch() const
{
    return asin(2.0*(qw*qy - qz*qx));
}

// get euler yaw angle
double Quaternion::get_euler_yaw() const
{
    return atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz));
}

// create eulers from a quaternion
void Quaternion::to_euler(double &roll, double &pitch, double &yaw) const
{
    roll = get_euler_roll();
    pitch = get_euler_pitch();
    yaw = get_euler_yaw();
}

// create eulers from a quaternion
Eigen::Vector3d Quaternion::to_vector312(void) const
{
    Eigen::Matrix3d m = rotation_matrix();
    return Eigen::Vector3d(asin(m(2,1)), atan2(-m(2,0), m(2,2)), atan2(-m(0,1), m(1,1)));
}

Eigen::Vector3d Quaternion::to_vector321(void) const
{
    Eigen::Vector3d m;
    m.x() = get_euler_roll();
    m.y() = get_euler_pitch();
    m.z() = get_euler_yaw();
    return m;
}

Quaternion Quaternion::get_rp_quaternion(void) const
{
    double tqx;
    double tqy;
    double tqw;
    
    double qw2 = qw * qw;
    double qz2 = qz * qz;
    double qwx = qw * qx;
    double qwy = qw * qy;
    double qxz = qx * qz;
    double qyz = qy * qz;
    
    double qw2Pqz2 = (qw2 + qz2);
    if(!is_zero(qw2Pqz2)) {		
        tqw = sqrt(qw2Pqz2);
        double inv_tqw = 1.0 / tqw;
        tqx = (qwx + qyz) * inv_tqw;
        tqy = (qwy - qxz) * inv_tqw;
    } else {
        tqw = 0.0;
        tqx = qx;
        tqy = qy;
    }
    Quaternion result(tqw, tqx, tqy, 0);
    return result;
}

double Quaternion::norm(void) const
{
    return sqrt(sq(qw) + sq(qx) + sq(qy) + sq(qz));
}

Quaternion Quaternion::inverse(void) const
{
    return Quaternion(qw, -qx, -qy, -qz);
}

void Quaternion::normalize(void)
{
    double quatMag = norm();
    if (!is_zero(quatMag)) {
        double quatMagInv = 1.0/quatMag;
        qw *= quatMagInv;
        qx *= quatMagInv;
        qy *= quatMagInv;
        qz *= quatMagInv;
    }
}

Quaternion Quaternion::operator*(const Quaternion &v) const
{
    Quaternion ret;
    const double &w1 = qw;
    const double &x1 = qx;
    const double &y1 = qy;
    const double &z1 = qz;

    double w2 = v.qw;
    double x2 = v.qx;
    double y2 = v.qy;
    double z2 = v.qz;

    ret.qw = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    ret.qx = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    ret.qy = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    ret.qz = w1*z2 + x1*y2 - y1*x2 + z1*w2;

    return ret;
}

Quaternion &Quaternion::operator*=(const Quaternion &v)
{
    double w1 = qw;
    double x1 = qx;
    double y1 = qy;
    double z1 = qz;

    double w2 = v.qw;
    double x2 = v.qx;
    double y2 = v.qy;
    double z2 = v.qz;

    qw = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    qx = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    qy = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    qz = w1*z2 + x1*y2 - y1*x2 + z1*w2;

    return *this;
}

Quaternion Quaternion::operator/(const Quaternion &v) const
{
    Quaternion ret;
    const double &quat0 = qw;
    const double &quat1 = qx;
    const double &quat2 = qy;
    const double &quat3 = qz;

    double rquat0 = v.qw;
    double rquat1 = v.qx;
    double rquat2 = v.qy;
    double rquat3 = v.qz;

    ret.qw = (rquat0*quat0 + rquat1*quat1 + rquat2*quat2 + rquat3*quat3);
    ret.qx = (rquat0*quat1 - rquat1*quat0 - rquat2*quat3 + rquat3*quat2);
    ret.qy = (rquat0*quat2 + rquat1*quat3 - rquat2*quat0 - rquat3*quat1);
    ret.qz = (rquat0*quat3 - rquat1*quat2 + rquat2*quat1 - rquat3*quat0);
    return ret;
}
}