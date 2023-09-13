#include <ekf_core/ekf_Fusion.h>
#include <string>

using namespace math_lib;
using namespace Eigen;
using namespace msckf_core;

Vector3d ekf_Fusion::gravity = Vector3d(0, 0, -GRAVITY_MSS);
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();
FeatureIDType Feature::next_id = 0;

Feature::OptimizationConfig Feature::optimization_config;

ekf_Fusion::ekf_Fusion(void):
    dtEkfAvg(0.005)
{
}

ekf_Fusion::~ekf_Fusion(void)
{

}
void ekf_Fusion::InitialiseVariables()
{
    prevTnb = Matrix3d::Identity();
    // Initialize state server
    state_server.continuous_noise_cov = Matrix<double, 12, 12>::Zero();
    // Initialize state server
    state_server.continuous_noise_cov.block<3, 3>(0, 0) = Matrix3d::Identity()*sq(_gyrNoise);
    state_server.continuous_noise_cov.block<3, 3>(3, 3) = Matrix3d::Identity()*sq(_gyroBiasProcessNoise);
    state_server.continuous_noise_cov.block<3, 3>(6, 6) = Matrix3d::Identity()*sq(_accNoise);
    state_server.continuous_noise_cov.block<3, 3>(9, 9) = Matrix3d::Identity()*sq(_accelBiasProcessNoise);

    state_server.imu_state.orientation = Vector4d(0, 0, 0, 1);
    state_server.imu_state.angErr     = Vector3d::Zero(); // 0-2
    state_server.imu_state.gyro_bias  = Vector3d::Zero(); // 3-5
    state_server.imu_state.velocity   = Vector3d::Zero(); // 6-8
    state_server.imu_state.acc_bias   = Vector3d::Zero(); // 9-11
    state_server.imu_state.position   = Vector3d::Zero(); // 12-14
    state_server.imu_state.cam_imu_dt = 0.0; // 15

    state_server.imu_state.orientation_null = Vector4d(0, 0, 0, 1);
    state_server.imu_state.position_null = Vector3d::Zero();
    state_server.imu_state.velocity_null = Vector3d::Zero();

    state_server.cam_states.clear();

    delAngCorrected = Vector3d::Zero();
    delVelCorrected = Vector3d::Zero();

    imuSampleTime_ms = 0;

    accNavMag = 0.0;
    accNavMagHoriz = 0.0;
    velDotNED = Vector3d::Zero();
    velDotNEDfilt = Vector3d::Zero();
    tiltErrorVariance = 0.0;

    init_decl = false;
    mag_decl = 0.0;

    map_server.clear();
}

// initialise the covariance matrix
void ekf_Fusion::CovarianceInit()
{
    // zero the matrix
    state_server.state_cov = MatrixXd::Zero(22, 22);

    // attitude error
    // state_server.state_cov(0, 0) = 0.001;
    // state_server.state_cov(1, 1) = state_server.state_cov(0, 0);
    // state_server.state_cov(2, 2) = state_server.state_cov(0, 0);

    // gyro delta angle biases
    state_server.state_cov(3, 3) = 0.0001;
    state_server.state_cov(4, 4) = state_server.state_cov(3, 3);
    state_server.state_cov(5, 5) = state_server.state_cov(3, 3);

    // velocities
    state_server.state_cov(6, 6) = 0.05;
    state_server.state_cov(7, 7) = state_server.state_cov(6, 6);
    state_server.state_cov(8, 8) = state_server.state_cov(6, 6);

    // accl delta velocity bias
    state_server.state_cov(9, 9)   = 0.001;
    state_server.state_cov(10, 10) = state_server.state_cov(9, 9);
    state_server.state_cov(11, 11) = state_server.state_cov(9, 9);

    // positions
    // state_server.state_cov(12, 12) = 0.0;
    // state_server.state_cov(13, 13) = state_server.state_cov(12, 12);
    // state_server.state_cov(14, 14) = state_server.state_cov(12, 12);

    // extrinsic_rotation_cov
    state_server.state_cov(15, 15) = 1.0e-4;
    state_server.state_cov(16, 16) = state_server.state_cov(15, 15);
    state_server.state_cov(17, 17) = state_server.state_cov(15, 15);
    
    // extrinsic_translation_cov
    state_server.state_cov(18, 18) = 1.5e-5;
    state_server.state_cov(19, 19) = state_server.state_cov(18, 18);
    state_server.state_cov(20, 20) = state_server.state_cov(18, 18);

    // cam imu sync dt
    state_server.state_cov(21, 21) = 0.00001;
}

void ekf_Fusion::UpdateStrapdownRungeKuttaNED()
{
    Vector3d gyro_avg = delAngCorrected;
    Vector3d accl_avg = delVelCorrected;
    double gyro_norm = gyro_avg.norm();

    Vector4d& q = state_server.imu_state.orientation;
    Vector3d& v = state_server.imu_state.velocity;
    Vector3d& p = state_server.imu_state.position;

    prevTnb = quaternionToRotation(q); // world->body

    Vector3d delVelNav;
    delVelNav  = prevTnb.transpose() * delVelCorrected;
    delVelNav.z() += -GRAVITY_MSS;

    // 计算速度变化量
    velDotNED = delVelNav;

    // 对速度变化量进行低通滤波
    velDotNEDfilt = velDotNED * 0.05 + velDotNEDfilt * 0.95;

    // 计算速度变化量的量级
    accNavMag = velDotNEDfilt.norm();
    accNavMagHoriz = norm(velDotNEDfilt.x(), velDotNEDfilt.y());

    Matrix4d Omega = Matrix4d::Zero();
    Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro_avg);
    Omega.block<3, 1>(0, 3) = gyro_avg;
    Omega.block<1, 3>(3, 0) = -gyro_avg;

    Vector4d dq_dt, dq_dt2;

    // 四元数0阶积分
    if (gyro_norm > 1e-5) {
        dq_dt = (Matrix4d::Identity()*cos(gyro_norm*dtIMUavg*0.5) + Omega*(1.0/gyro_norm*sin(gyro_norm*dtIMUavg*0.5))) * q;
        dq_dt2 = (Matrix4d::Identity()*cos(gyro_norm*dtIMUavg*0.25) + Omega*(1.0/gyro_norm*sin(gyro_norm*dtIMUavg*0.25))) * q;
    } else {
        dq_dt = (Matrix4d::Identity()+Omega*0.5*dtIMUavg) * cos(gyro_norm*dtIMUavg*0.5) * q;
        dq_dt2 = (Matrix4d::Identity()+Omega*0.25*dtIMUavg) * cos(gyro_norm*dtIMUavg*0.25) * q;
    }

    Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
    Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

    // 4阶龙格-库塔积分
    // k1 = f(tn, yn)
    Vector3d k1_v_dot = prevTnb.transpose()*accl_avg + ekf_Fusion::gravity;
    Vector3d k1_p_dot = v;

    // k2 = f(tn+dt/2, yn+k1*dt/2)
    Vector3d k1_v = v + k1_v_dot*dtIMUavg/2;
    Vector3d k2_v_dot = dR_dt2_transpose*accl_avg + ekf_Fusion::gravity;
    Vector3d k2_p_dot = k1_v;

    // k3 = f(tn+dt/2, yn+k2*dt/2)
    Vector3d k2_v = v + k2_v_dot*dtIMUavg/2;
    Vector3d k3_v_dot = dR_dt2_transpose*accl_avg + ekf_Fusion::gravity;
    Vector3d k3_p_dot = k2_v;

    // k4 = f(tn+dt, yn+k3*dt)
    Vector3d k3_v = v + k3_v_dot*dtIMUavg;
    Vector3d k4_v_dot = dR_dt_transpose*accl_avg + ekf_Fusion::gravity;
    Vector3d k4_p_dot = k3_v;

    // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
    state_server.imu_state.orientation = dq_dt;
    quaternionNormalize(state_server.imu_state.orientation);
    state_server.imu_state.velocity = v + (k1_v_dot+k2_v_dot*2.0+k3_v_dot*2.0+k4_v_dot)*(dtIMUavg/6.0);
    state_server.imu_state.position = p + (k1_p_dot+k2_p_dot*2.0+k3_p_dot*2.0+k4_p_dot)*(dtIMUavg/6.0);

    // 状态量进行限制
    ConstrainStates();
}

void ekf_Fusion::CovariancePrediction(void)
{
    Vector3d gyro = delAngCorrected;
    Vector3d accl = delVelCorrected;
    double dtime = dtIMUavg;
    // Compute discrete transition and noise covariance matrix
    Matrix<double, 22, 22> F = Matrix<double, 22, 22>::Zero();
    Matrix<double, 22, 12> G = Matrix<double, 22, 12>::Zero();

    F.block<3, 3>(IMUState::rot_ids, IMUState::rot_ids) = -skewSymmetric(gyro);   // angErr
    F.block<3, 3>(IMUState::rot_ids, IMUState::bg_ids)  = -Matrix3d::Identity();  // gyro_bias
    F.block<3, 3>(IMUState::vel_ids, IMUState::rot_ids) = -quaternionToRotation(state_server.imu_state.orientation_null).transpose()*skewSymmetric(accl); // velocity
    F.block<3, 3>(IMUState::vel_ids, IMUState::ba_ids)  = -quaternionToRotation(state_server.imu_state.orientation_null).transpose(); // acc_bias
    F.block<3, 3>(IMUState::pos_ids, IMUState::vel_ids) = Matrix3d::Identity(); // position

    G.block<3, 3>(IMUState::rot_ids, IMUState::rot_ids) = -Matrix3d::Identity();
    G.block<3, 3>(IMUState::bg_ids,  IMUState::bg_ids)  = Matrix3d::Identity();
    G.block<3, 3>(IMUState::vel_ids, IMUState::vel_ids) = -quaternionToRotation(state_server.imu_state.orientation_null).transpose();
    G.block<3, 3>(IMUState::ba_ids,  IMUState::ba_ids)  = Matrix3d::Identity();

    // Approximate matrix exponential to the 3rd order,
    // which can be considered to be accurate enough assuming
    // dtime is within 0.01s.
    Matrix<double, 22, 22> Fdt = F * dtime;
    Matrix<double, 22, 22> Fdt_square = Fdt * Fdt;
    Matrix<double, 22, 22> Fdt_cube = Fdt_square * Fdt;
    Matrix<double, 22, 22> Phi = Matrix<double, 22, 22>::Identity() + Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

    // Modify the transition matrix
    Matrix3d R_kk_1 = quaternionToRotation(state_server.imu_state.orientation_null);
    Phi.block<3, 3>(IMUState::rot_ids, IMUState::rot_ids) = quaternionToRotation(state_server.imu_state.orientation) * R_kk_1.transpose();

    Vector3d u = R_kk_1 * ekf_Fusion::gravity;
    RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

    Matrix3d A1 = Phi.block<3, 3>(IMUState::vel_ids, IMUState::rot_ids);
    Vector3d w1 = skewSymmetric(state_server.imu_state.velocity_null-state_server.imu_state.velocity) * ekf_Fusion::gravity;
    Phi.block<3, 3>(IMUState::vel_ids, IMUState::rot_ids) = A1 - (A1*u-w1)*s;

    Matrix3d A2 = Phi.block<3, 3>(IMUState::pos_ids, IMUState::rot_ids);
    Vector3d w2 = skewSymmetric(dtime*state_server.imu_state.velocity_null+state_server.imu_state.position_null-state_server.imu_state.position) * ekf_Fusion::gravity;
    Phi.block<3, 3>(IMUState::pos_ids, IMUState::rot_ids) = A2 - (A2*u-w2)*s;

    // Propogate the state covariance matrix.
    Matrix<double, 22, 22> Q = Phi*G*state_server.continuous_noise_cov*G.transpose()*Phi.transpose()*dtime;
    // 更新协方差矩阵中IMU-IMU块部分
    state_server.state_cov.block<22, 22>(IMUState::rot_ids, IMUState::rot_ids) = Phi*state_server.state_cov.block<22, 22>(IMUState::rot_ids, IMUState::rot_ids)*Phi.transpose() + Q;

    // 更新协方差矩阵中IMU-Camera块部分
    if (state_server.cam_states.size() > 0) {
        state_server.state_cov.block(0, 22, 22, state_server.state_cov.cols()-22) = Phi * state_server.state_cov.block(0, 22, 22, state_server.state_cov.cols()-22);
        state_server.state_cov.block(22, 0, state_server.state_cov.rows()-22, 22) = state_server.state_cov.block(22, 0, state_server.state_cov.rows()-22, 22) * Phi.transpose();
    }

    // 强制协方差变为对称矩阵：主对角线元素取绝对值，非对角线元素对称元素取平均值
    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    // Update the state correspondes to null space.
    state_server.imu_state.orientation_null = state_server.imu_state.orientation;
    state_server.imu_state.position_null = state_server.imu_state.position;
    state_server.imu_state.velocity_null = state_server.imu_state.velocity;

    ConstrainVariances();
}

void ekf_Fusion::stateAugmentation(void)
{
    Vector3d gyro_avg = delAngCorrected;
    const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
    const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

    // Add a new camera state to the state server.
    // 由IMU以及IMU与camera的固连关系得到相机的位置和姿态
    Matrix3d R_w_i = quaternionToRotation(state_server.imu_state.orientation); // world -> body
    Matrix3d R_w_c = R_i_c * R_w_i;
    Vector3d t_c_w = state_server.imu_state.position + R_w_i.transpose()*t_c_i;

    // 创建新的camera位姿状态
    state_server.cam_states[state_server.imu_state.id] = CAMState(state_server.imu_state.id);
    CAMState& cam_state = state_server.cam_states[state_server.imu_state.id];

    cam_state.timestamp = camDataDelayed.time_us;
    cam_state.orientation = rotationToQuaternion(R_w_c);
    cam_state.position = t_c_w;

    cam_state.orientation_null = cam_state.orientation;
    cam_state.position_null = cam_state.position;

    // Update the covariance matrix of the state.
    // To simplify computation, the matrix J below is the nontrivial block
    // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
    // -aided Inertial Navigation".
    // 增广状态以后，需要得到增广状态（相机位置、相机四元数）对msckf状态（增广前的状态）的雅克比
    // 用于计算增广后的相机状态协方差
    Matrix<double, 6, 22> J = Matrix<double, 6, 22>::Zero();
    J.block<3, 3>(0, 0) = R_i_c;
    J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_c_i);
    //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
    J.block<3, 3>(3, 12) = Matrix3d::Identity();
    // 外参
    J.block<3, 3>(0, 15) = Matrix3d::Identity();
    J.block<3, 3>(3, 18) = R_w_i.transpose();
    // Jt
    J.block<3, 1>(0, 21) = R_i_c*gyro_avg;
    J.block<3, 1>(3, 21) = state_server.imu_state.velocity + R_w_i*skewSymmetric(gyro_avg)*t_c_i;

    /*
        构造增广前的协方差矩阵：
        |imu协方差      imu与相机协方差|
        |相机与imu协方差     相机协方差|
    */
    // 调整协方差矩阵大小
    size_t old_rows = state_server.state_cov.rows();
    size_t old_cols = state_server.state_cov.cols();
    state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

    // Rename some matrix blocks for convenience.
    const Matrix<double, 22, 22>& P11 = state_server.state_cov.block<22, 22>(0, 0);
    const MatrixXd& P12 = state_server.state_cov.block(0, 22, 22, old_cols-22);

    // Fill in the augmented state covariance.
    // MSCKF状态-增广状态协方差计算
    state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;  // 相机与imu协方差
    state_server.state_cov.block(0, old_cols, old_rows, 6) = state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose(); // imu与相机协方差
    state_server.state_cov.block<6, 6>(old_rows, old_cols) = J * P11 * J.transpose(); // 相机协方差

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    ConstrainVariances();
}

void ekf_Fusion::updateState(VectorXd &delta_x_imu, bool update_all)
{
    const Vector4d dq_imu = smallAngleQuaternion(delta_x_imu.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(dq_imu, state_server.imu_state.orientation);
    quaternionNormalize(state_server.imu_state.orientation);
    if(update_all) {
        state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
        state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
        state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
        state_server.imu_state.position += delta_x_imu.segment<3>(12);
    }

    ConstrainStates();
}

void ekf_Fusion::updateStateCov(MatrixXd K, MatrixXd H, MatrixXd Q)
{
    // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H.cols()) - K*H;
    //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //  K*K.transpose()*Feature::observation_noise;
    state_server.state_cov = I_KH * state_server.state_cov * I_KH.transpose() + K*Q*K.transpose();

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    ConstrainVariances();
}
// 对协方差矩阵进行限制
void ekf_Fusion::ConstrainVariances()
{
    // attitude error
    for (uint8_t i=0; i<=2; i++) state_server.state_cov(i,i) = constrain_double(state_server.state_cov(i,i),0.0,1.0); 

    // gyro bias
    for (uint8_t i=3; i<=5; i++) state_server.state_cov(i,i) = constrain_double(state_server.state_cov(i,i),0.0,10.0);

    // velocities
    for (uint8_t i=6; i<=8; i++) state_server.state_cov(i,i) = constrain_double(state_server.state_cov(i,i),0.0,1.0e3); 

    // accl bias
    for (uint8_t i=9; i<=11; i++) state_server.state_cov(i,i) = constrain_double(state_server.state_cov(i,i),0.0,20.0);

    // position
    for (uint8_t i=12; i<=14; i++) state_server.state_cov(i,i) = constrain_double(state_server.state_cov(i,i),0.0,1.0e6);
}

// constrain states to prevent ill-conditioning
void ekf_Fusion::ConstrainStates()
{
    // 四元数范围 +-1
    for (uint8_t i=0; i<=3; i++) state_server.imu_state.orientation[i] = constrain_double(state_server.imu_state.orientation[i],-1.0,1.0);
    // 速度限制范围 +-500 m/s (有空速计可根据 airspeed * EAS2TAS)
    for (uint8_t i=0; i<=2; i++) state_server.imu_state.velocity[i] = constrain_double(state_server.imu_state.velocity[i],-5.0e2,5.0e2);
    // 位置限制 +-1000 km
    for (uint8_t i=0; i<=2; i++) state_server.imu_state.position[i] = constrain_double(state_server.imu_state.position[i],-1.0e6,1.0e6);
    // 陀螺仪偏置限制 +-0.5 rad/s
    for (uint8_t i=0; i<=2; i++) state_server.imu_state.gyro_bias[i] = constrain_double(state_server.imu_state.gyro_bias[i],-1.5,1.5);
    // 加速计偏置限制 +-1 m/s
    for (uint8_t i=0; i<=2; i++) state_server.imu_state.acc_bias[i] = constrain_double(state_server.imu_state.acc_bias[i],-5.0,5.0);
    // cam imu时间误差
    state_server.imu_state.cam_imu_dt = constrain_double(state_server.imu_state.cam_imu_dt,-0.5,0.5);
}

