#include <ekf_core/ekf_Fusion.h>
#include <string>

using namespace math_lib;
using namespace Eigen;

// fuse selected position, velocity and height measurements
void ekf_Fusion::FusePosNED(Vector3d PosObs, double PosNoise)
{
    MatrixXd P = state_server.state_cov;
    
    MatrixXd Q_pos = MatrixXd::Zero(3, 3);

    // 获取测量更新数据
    Vector3d observation = PosObs;
    Vector3d innovNEDPos = Vector3d::Zero();
    Vector3d varInnovNEDPos = Vector3d::Zero();
    double posTestRatio = 0.0;

    innovNEDPos(0) = observation(0) - state_server.imu_state.position(0);
    innovNEDPos(1) = observation(1) - state_server.imu_state.position(1);
    innovNEDPos(2) = observation(2) - state_server.imu_state.position(2);
    Q_pos(0, 0) = PosNoise;
    Q_pos(1, 1) = Q_pos(0, 0);
    Q_pos(2, 2) = Q_pos(0, 0);
    varInnovNEDPos[0] = P(12,12) + Q_pos(0, 0);
    varInnovNEDPos[1] = P(13,13) + Q_pos(1, 1);
    varInnovNEDPos[2] = P(14,14) + Q_pos(2, 2);
    
    // apply an innovation consistency threshold test, but don't fail if bad IMU data
    double maxPosInnov2 = _PosInnovGate*(varInnovNEDPos[0] + varInnovNEDPos[1] + varInnovNEDPos[2]);
    posTestRatio = (sq(innovNEDPos[0]) + sq(innovNEDPos[1]) + sq(innovNEDPos[2])) / maxPosInnov2;

    // set range for sequential fusion of velocity and position measurements depending on which data is available and its health
    if(posTestRatio < 1.0) {
        MatrixXd H_pos = MatrixXd::Zero(3, P.cols());
        MatrixXd HP_pos = MatrixXd::Zero(3, P.cols());

        H_pos(0, 12) = 1.0;
        H_pos(1, 13) = 1.0;
        H_pos(2, 14) = 1.0;

        HP_pos.row(0) = P.row(12);
        HP_pos.row(1) = P.row(13);
        HP_pos.row(2) = P.row(14);

        MatrixXd S_pos = HP_pos * H_pos.transpose() + Q_pos;
        MatrixXd K_transpose_pos = S_pos.ldlt().solve(HP_pos);
        MatrixXd K_pos = K_transpose_pos.transpose();

        VectorXd delta_x = K_pos * innovNEDPos;

        // Update the IMU state.
        VectorXd delta_x_imu = delta_x.head<22>();

        updateState(delta_x_imu);

        updateStateCov(K_pos, H_pos, Q_pos);
    }
}
