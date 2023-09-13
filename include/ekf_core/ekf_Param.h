#ifndef __EKF_PARAM_H__
#define __EKF_PARAM_H__

#include <stdint.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>


class ekf_Parameter
{
public:
    ekf_Parameter(std::shared_ptr<rclcpp::Node> node);
    ~ekf_Parameter(void);

    void getParameter(void);

    static constexpr bool usecompass                 = true;

    // Threshold for determine keyframes
    static constexpr double translation_threshold = 0.4;
    static constexpr double rotation_threshold    = 0.2618;
    static constexpr double tracking_rate_threshold = 0.5;
    static constexpr uint32_t max_cam_state_size = 20; // Maximum number of camera states

    bool external_att;
    // topics
    std::string imu_topics;
    std::string vio_topics;
    std::string cam0_topics;
    std::string cam1_topics;

    // Transformation between the IMU and the
    // left camera (cam0)
    Eigen::Matrix3d R_imu_cam0;
    Eigen::Vector3d t_cam0_imu;

    Eigen::Isometry3d T_cam0_cam1;

private:
    std::shared_ptr<rclcpp::Node> _node;

    template<class T>
    void setNgetNodeParameter(T& param, 
        const std::string& param_name, 
        const T& default_value, 
        const rcl_interfaces::msg::ParameterDescriptor &parameter_descriptor=rcl_interfaces::msg::ParameterDescriptor());

};

#endif
