#include <ekf_core/ekf_Fusion.h>
#include <string>

using namespace math_lib;
using namespace Eigen;

void ekf_Fusion::FuseExtAtt(Vector4d ExtQuat, double AttNoise)
{
    MatrixXd P = state_server.state_cov;
    MatrixXd H = MatrixXd::Zero(3, P.cols());
    MatrixXd Q = MatrixXd::Zero(3, 3);

    // yaw measurement error variance (rad^2)
    Q(0, 0) = AttNoise;
    Q(1, 1) = AttNoise;
    Q(2, 2) = AttNoise;

    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;

    // Calculate the innovation
    Vector3d innovation = Vector3d::Zero();

    Vector4d curr_quat = state_server.imu_state.orientation;
    Vector4d q_temp = InverseQuaternion(curr_quat);
    Vector4d q_temp2 = quaternionMultiplication(ExtQuat, q_temp);
    // Compute the error of the state.
    innovation = QuaternionToAxisAngle(q_temp2);

    // printf("att err:%.4f, %.4f, %.4f \n", innovation(0)*RAD_TO_DEG, innovation(1)*RAD_TO_DEG, innovation(2)*RAD_TO_DEG);
        
    MatrixXd HP = MatrixXd::Zero(3, P.cols());

    HP.row(0) = P.row(0);
    HP.row(1) = P.row(1);
    HP.row(2) = P.row(2);

    MatrixXd S = HP * H.transpose() + Q;
    MatrixXd K_transpose = S.ldlt().solve(HP);
    MatrixXd K = K_transpose.transpose();

    VectorXd delta_x = K * innovation;

    // Update the IMU state.
    VectorXd delta_x_imu = delta_x.head<22>();

    updateState(delta_x_imu, false);

    updateStateCov(K, H, Q);
}


