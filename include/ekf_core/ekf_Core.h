#ifndef __EKF_CORE_H__
#define __EKF_CORE_H__

/**
 * Includes
 */
#include <rclcpp/rclcpp.hpp>
#include <semaphore.h>

#include <common/math_library.h>
#include <common/perform.h>

#include <ekf_core/ekf_Fusion.h>
#include <ekf_core/ekf_Param.h>

#include <image_track/image_track.h>

#include <sprain_msgs/msg/vehicle_attitude.hpp>
#include <sprain_msgs/msg/sensor_bias.hpp>
#include <sprain_msgs/msg/local_position.hpp>
#include <sprain_msgs/msg/estimator_status.hpp>

#include <sprain_msgs/msg/sensor_delta.hpp>
#include <sprain_msgs/msg/camera_measurement.hpp>
#include <sprain_msgs/msg/sensor_vio.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#define earthRate                        0.000072921 // earth rotation rate (rad/sec)

#define DATA_VALID_HISTORY_THRESHOLD     6u

class ekf_Core : public ekf_Fusion
{
public:
    // Constructor
    ekf_Core(std::shared_ptr<rclcpp::Node> node);
    ~ekf_Core(void);

    bool setup_core(void);

    uint64_t timestamp(void) const { return RCL_NS_TO_US(_node->now().nanoseconds()); }

    bool InitialiseFilter(void);
    bool RestartFilter(void);
    void InitialiseVariables(void);
    
    void UpdateFilter(void);
    void updateFilterStatus(void);
    void StatePublished(void);
    void StatusPublished(void);

    void ControlAttFuse(void);
    void ControlCamFuse(void);
    void ControlConstFuse(void);

    // 滤波器状态输出
    void getRotationBodyToNED(Eigen::Matrix3d &mat) const;
    void getQuaternion(Eigen::Vector4d &quat, std::array<float, 3>& variances) const;
    void getGyroBias(Eigen::Vector3d &gyroBias) const;
    void getAccelBias(Eigen::Vector3d &accelBias) const;
    void getAccXYZ(Eigen::Vector3d &accXYZ) const;
    void getPosNED(Eigen::Vector3d &posNED, std::array<float, 3>& variances) const;
    void getVelNED(Eigen::Vector3d &vel, std::array<float, 3>& variances) const;

    // 滤波器状态判断
    bool healthy(void) const;

    void delta_callback(const sprain_msgs::msg::SensorDelta::ConstSharedPtr data);
    void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr data);
    void cam_callback(const sensor_msgs::msg::Image::ConstSharedPtr img0, 
        const sensor_msgs::msg::Image::ConstSharedPtr img1, int cam_id0, int cam_id1);

    void att_callback(const sprain_msgs::msg::VehicleAttitude::ConstSharedPtr data);
    void pos_callback(const sprain_msgs::msg::LocalPosition::ConstSharedPtr data);
    void gd_callback(const geometry_msgs::msg::PointStamped::ConstSharedPtr data);

    void shutdown(void);

private:
    std::shared_ptr<rclcpp::Node> _node;

    ekf_Parameter _param;
    ImageProcessor _image_track;

    // sem_t  update_sem;
    static bool   thread_run;
    bool   update_overrun;
    uint32_t cam_feature_size;
    bool   cam_update;

    std::mutex camera_queue_mtx;
    std::mutex imu_queue_mtx;
    std::mutex attitude_queue_mtx;

    message_filters::Subscriber<sensor_msgs::msg::Image> _cam0_img_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> _cam1_img_sub;

    // std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>> _image_sync;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> sync_pol;
    std::shared_ptr<message_filters::Synchronizer<sync_pol>> _image_sync;

    rclcpp::Subscription<sprain_msgs::msg::SensorDelta>::SharedPtr       delta_subscription;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr               imu_subscription;
    rclcpp::Subscription<sprain_msgs::msg::VehicleAttitude>::SharedPtr   ext_att_subscription;
    rclcpp::Subscription<sprain_msgs::msg::LocalPosition>::SharedPtr     ext_pos_subscription;

    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr    gd_pos_subscription;

    nav::imu_elements     imuDataNew;
    nav::att_elements     attDataNew;     // 地磁数据
    nav::att_elements     attDataDelayed;
    nav::feature_elements camDataNew;

    bool is_first_img;
    uint32_t filter_error_count;
    uint64_t last_imu_sample{0};
    
    std::vector<nav::imu_elements>     imu_msg_buffer;
    std::vector<nav::att_elements>     att_msg_buffer;
    std::vector<nav::feature_elements> cam_msg_buffer;

    bool     statesInitialised;
    bool     tiltAlignComplete;
    bool     delAngBiasLearned;
    bool     yawAlignComplete;
    bool     statesInitDynamics;
    Eigen::Vector3d lastKnownPositionNE;

    uint32_t firstInitTime_ms;   // 滤波器初始化等待时间
    uint32_t ekfStartTime_ms;    // 滤波器初始化时间

    // ------传感器数据-------------------------------------
    Eigen::Vector3d               accelmeter;
    Eigen::Vector3d               gyroscope;
    // end-------------------------------------------------

    // ------滤波器逻辑控制---------------------------------
    void checkAttitudeAlignmentStatus();
    void checkGyroCalStatus(void);
    // end-------------------------------------------------

    // ------地磁数据融合控制变量&函数-----------------------
    bool     magYawResetRequest;          // 地磁航向对准标志位，初始化，第一次打开地磁状态估计，地磁状态估计异常
    Eigen::Vector4d prevQuatMagReset;       // 记录上一次航向角对准时的姿态

    void controlMagYawReset(void);
    void alignYawAngle();
    void recordYawReset();
    // end--------------------------------------------------

    // ------Camera fuse-----------------------
    double tracking_rate;
    uint64_t last_cam_fusion_ms;
    // Chi squared test table.
    static std::map<int, double> chi_squared_test_table;

    void addFeatureObservations(void);
    bool gatingTest(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, const int& dof);
    void removeLostFeatures(void);
    void findRedundantCamStates(std::vector<msckf_core::StateIDType>& rm_cam_state_ids);
    void pruneCamStateBuffer(void);
    // end-------------------------------------------------

    rclcpp::Publisher<sprain_msgs::msg::SensorVio>::SharedPtr       vio_publisher;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr           odometry_publisher;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr     feature_publisher;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr               path_publisher;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr               gd_path_publisher;

    std::unique_ptr<tf2_ros::TransformBroadcaster>                  tf_broadcaster;
    std::unique_ptr<tf2_ros::TransformBroadcaster>                  tf_aircraft;

    nav_msgs::msg::Path                                             path_msg;
    nav_msgs::msg::Path                                             gd_path_msg;

    bool is_gd_position_init{false};
    Eigen::Vector3d gd_position_orign;
    Eigen::Vector3d gd_position;

    perf_counter_t  _perf_Update;
    perf_counter_t  _perf_CovariancePrediction;
    perf_counter_t  _perf_FuseAttitude;
    perf_counter_t  _perf_FuseCAM;

    perf_counter_t  _perf_Interval;
    perf_counter_t  _perf_Att_Interval;
    perf_counter_t  _perf_CAM_Interval;
    perf_counter_t  _perf_Remove_Interval;
    perf_counter_t  _perf_Prune_Interval;

    perf_counter_t  _perf_CheckMove_Error;
    perf_counter_t  _perf_InitPos_Error;
    perf_counter_t  _perf_Gating1_Error;
    perf_counter_t  _perf_Gating2_Error;
    perf_counter_t  _perf_Const_pos;
    perf_counter_t  _perf_Cam_overdrop;
    perf_counter_t  _perf_Update_overrun;
    perf_counter_t  _perf_Update_failed;
    perf_counter_t  _perf_Reset_count;
};

#endif
