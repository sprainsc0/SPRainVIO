#ifndef __EKF_FUSION_H__
#define __EKF_FUSION_H__

/**
 * Includes
 */
#include <common/math_library.h>
#include <common/quaternion.h>
#include <common/imu_state.h>
#include <common/cam_state.h>
#include <common/feature.hpp>
#include <common/quaternion.h>
#include <nav_common.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>


class ekf_Fusion
{
public:
    // Constructor
    ekf_Fusion(void);
    ~ekf_Fusion(void);

    void InitialiseVariables();
    void CovarianceInit();

    // 状态更新
    void UpdateStrapdownRungeKuttaNED(void);
    // 协方差矩阵更新
    void CovariancePrediction(void);
    void stateAugmentation(void);
    // 测量更新
    void FuseEulerYaw(double YawNoise, Eigen::Vector3d MagFiled, double YawAngle, bool usePredictedYaw, bool useExternalYawSensor);
    void FuseVelNED(Eigen::Vector3d VelObs, double VelNoise);
    void FusePosNED(Eigen::Vector3d PosObs, double PosNoise);
    void FuseExtAtt(Eigen::Vector4d ExtQuat, double AttNoise);

    void FeatureJacobian(const msckf_core::FeatureIDType& feature_id,const std::vector<msckf_core::StateIDType>& cam_state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);
    bool MeasurementUpdate(const Eigen::MatrixXd& H, const Eigen::VectorXd& r);

    // Noise
    static constexpr double _gyrNoise              = 1.5000E-2; // 1.0E-3;    // gyro process noise : rad/s
    static constexpr double _accNoise              = 3.5000E-1; // 5.0E-2;    // accelerometer process noise : m/s^2
    static constexpr double _gyroBiasProcessNoise  = 1.0E-4;     // gyro bias state process noise : rad/s
    static constexpr double _accelBiasProcessNoise = 3.0E-3;     // accel bias state process noise : m/s^2
    static constexpr double _noaidHorizNoise       = 6.0;   // horizontal position zero position measurement noise : m
    static constexpr double _yawNoise              = 0.3;   // magnetic yaw measurement noise : rad
    static constexpr double _observation_noise     = 0.025;
    static constexpr bool _estimate_external       = true;
    static Eigen::Vector3d gravity;

protected:
    struct StateServer {
        msckf_core::IMUState imu_state;
        msckf_core::CamStateServer cam_states;

        // State covariance matrix
        Eigen::MatrixXd state_cov;
        Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };
    // State vector
    StateServer state_server;

    // Features used
    msckf_core::MapServer map_server;

    double dtEkfAvg;
    double dtIMUavg;
    uint32_t imuSampleTime_ms;
    Eigen::Vector3d delAngCorrected;
    Eigen::Vector3d delVelCorrected;
    nav::imu_elements imuDataDelayed{};
    nav::feature_elements camDataDelayed{};

    double tiltErrorVariance;

    bool init_decl;
    double mag_decl;

    double accNavMag;                // NED三轴速度变化量（加速度）
    double accNavMagHoriz;           // NED水平速度变化量（加速度）
    Eigen::Vector3d velDotNED;             // NED下的速度变化量
    Eigen::Vector3d velDotNEDfilt;         // 用于落地检测等功能
    Eigen::Matrix3d prevTnb;               // NED坐标->机体坐标

private:
    // 数据检测，数据噪声容许门限值 
    static constexpr double _VelInnovGate  = 9.0; // sq(3.0f);
    static constexpr double _PosInnovGate  = 9.0; // sq(3.0f);
    static constexpr double _yawInnovGate  = 400.0; // sq(20.0f);

    void updateState(Eigen::VectorXd &delta_x_imu, bool update_all = true);

    void updateStateCov(Eigen::MatrixXd K, Eigen::MatrixXd H, Eigen::MatrixXd Q);

    void MeasurementJacobian(const msckf_core::StateIDType& cam_state_id, const msckf_core::FeatureIDType& feature_id,
        Eigen::Matrix<double, 4, 6>& H_x, Eigen::Matrix<double, 4, 3>& H_f, Eigen::Vector4d& r);

    // constrain variances (diagonal terms) in the state covariance matrix
    void ConstrainVariances();
    void ConstrainStates();
};

#endif
