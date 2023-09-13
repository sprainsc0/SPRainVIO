#include <ekf_core/ekf_Fusion.h>
#include <string>

using namespace math_lib;
using namespace Eigen;

// fuse selected position, velocity and height measurements
void ekf_Fusion::FuseVelNED(Vector3d VelObs, double VelNoise)
{
    MatrixXd P = state_server.state_cov;
    
    MatrixXd Q = MatrixXd::Zero(3, 3);

    // 获取测量更新数据
    Vector3d observation = VelObs;
    Vector3d innovNEDVel = Vector3d::Zero();
    Vector3d varInnovNEDVel = Vector3d::Zero();
    double velTestRatio = 0.0;

    innovNEDVel(0) = (observation(0) - state_server.imu_state.velocity(0));
    innovNEDVel(1) = (observation(1) - state_server.imu_state.velocity(1));
    innovNEDVel(2) = (observation(2) - state_server.imu_state.velocity(2));
    Q(0, 0) = VelNoise;
    Q(1, 1) = Q(0, 0);
    Q(2, 2) = Q(0, 0);
    varInnovNEDVel[0] = P(6,6) + Q(0, 0);
    varInnovNEDVel[1] = P(7,7) + Q(1, 1);
    varInnovNEDVel[2] = P(8,8) + Q(2, 2);

    double innovVelSumSq = 0; // sum of squares of velocity innovations
    double varVelSum = 0; // sum of velocity innovation variances
    for (uint8_t i = 0; i<3; i++) {
        // sum the innovation and innovation variances
        innovVelSumSq += sq(innovNEDVel[i]);
        varVelSum += innovNEDVel[i];
    }
    // apply an innovation consistency threshold test, but don't fail if bad IMU data
    // calculate the test ratio
    velTestRatio = innovVelSumSq / (varVelSum * _VelInnovGate);

    // set range for sequential fusion of velocity and position measurements depending on which data is available and its health
    if(velTestRatio < 1.0) {
        MatrixXd H = MatrixXd::Zero(3, P.cols());
        MatrixXd HP = MatrixXd::Zero(3, P.cols());

        H(0, 6) = 1.0;
        H(1, 7) = 1.0;
        H(2, 8) = 1.0;

        HP.row(0) = P.row(6);
        HP.row(1) = P.row(7);
        HP.row(2) = P.row(8);

        MatrixXd S = HP * H.transpose() + Q;
        MatrixXd K_transpose = S.ldlt().solve(HP);
        MatrixXd K = K_transpose.transpose();

        VectorXd delta_x = K * innovNEDVel;

        // Update the IMU state.
        VectorXd delta_x_imu = delta_x.head<22>();

        updateState(delta_x_imu);

        updateStateCov(K, H, Q);
    }
}
