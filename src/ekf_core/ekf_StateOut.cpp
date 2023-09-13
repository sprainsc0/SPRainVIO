#include <ekf_core/ekf_Core.h>
#include <rclcpp/logger.hpp>
#include <string>
#include <iostream>
#include <tf2_eigen/tf2_eigen.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace math_lib;

static uint8_t debug_count = 0;
static uint8_t path_count = 0;

void ekf_Core::getRotationBodyToNED(Eigen::Matrix3d &mat) const
{
    mat = quaternionToRotation(state_server.imu_state.orientation).transpose();
}

void ekf_Core::getQuaternion(Eigen::Vector4d& ret, std::array<float, 3>& variances) const
{
    ret = InverseQuaternion(state_server.imu_state.orientation);
    variances[0] = state_server.state_cov(0, 0);
    variances[1] = state_server.state_cov(1, 1);
    variances[2] = state_server.state_cov(2, 2);
}

void ekf_Core::getGyroBias(Eigen::Vector3d &gyroBias) const
{
    if (!statesInitialised) {
        gyroBias = Eigen::Vector3d::Zero();
        return;
    }
    gyroBias = state_server.imu_state.gyro_bias;
}

void ekf_Core::getAccelBias(Eigen::Vector3d &accelBias) const
{
    if (!statesInitialised) {
        accelBias = Eigen::Vector3d::Zero();
        return;
    }
    accelBias = state_server.imu_state.acc_bias;
}

void ekf_Core::getAccXYZ(Eigen::Vector3d &accXYZ) const
{
    accXYZ = accelmeter;
}

void ekf_Core::getPosNED(Eigen::Vector3d &posNED, std::array<float, 3>& variances) const
{
    posNED = state_server.imu_state.position;

    variances[0] = state_server.state_cov(12, 12);
    variances[1] = state_server.state_cov(13, 13);
    variances[2] = state_server.state_cov(14, 14);
}


void ekf_Core::getVelNED(Eigen::Vector3d &vel, std::array<float, 3>& variances) const
{
    vel = state_server.imu_state.velocity;
    variances[0] = state_server.state_cov(6, 6);
    variances[1] = state_server.state_cov(7, 7);
    variances[2] = state_server.state_cov(8, 8);
}

bool ekf_Core::healthy(void) const
{
    // Give the filter a second to settle before use
    if ((imuSampleTime_ms - ekfStartTime_ms) < 1000 ) {
        return false;
    }

    double position_x_std = std::sqrt(state_server.state_cov(12, 12));
    double position_y_std = std::sqrt(state_server.state_cov(13, 13));
    double position_z_std = std::sqrt(state_server.state_cov(14, 14));

    if (position_x_std > 6.0 ||
        position_y_std > 6.0 ||
        position_z_std > 6.0) {
        return false;
    }

    return true;
}

void ekf_Core::updateFilterStatus(void)
{
    bool filterHealthy = healthy();
    if(tiltAlignComplete && (yawAlignComplete || !_param.external_att)) {
        if(!filterHealthy) {
            filter_error_count++;
        } else {
            filter_error_count = 0;
        }
    }

    cam_update = (imuSampleTime_ms-last_cam_fusion_ms) < 1000;

    if(filter_error_count > 88) {
        filter_error_count = 0;
        RestartFilter();
    }
}

void ekf_Core::StatePublished(void)
{
    if(!statesInitialised || !cam_update) {
        return;
    }

    sprain_msgs::msg::SensorVio sensor_vio;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector4d qut;
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;

    getQuaternion(qut, sensor_vio.attvariance);
    getPosNED(pos, sensor_vio.posvariance);
    getVelNED(vel, sensor_vio.velvariance);

    getGyroBias(gyro_bias);

    sensor_vio.timestamp = camDataNew.time_us; // camDataDelayed.time_us;

    sensor_vio.quat[0] = qut[0];
    sensor_vio.quat[1] = qut[1];
    sensor_vio.quat[2] = qut[2];
    sensor_vio.quat[3] = qut[3];

    sensor_vio.position[0] = pos[0];
    sensor_vio.position[1] = pos[1];
    sensor_vio.position[2] = pos[2];

    sensor_vio.velocity[0] = vel[0];
    sensor_vio.velocity[1] = vel[1];
    sensor_vio.velocity[2] = vel[2];

    sensor_vio.delang[0] = 0.0;
    sensor_vio.delang[1] = 0.0;
    sensor_vio.delang[2] = 0.0;

    sensor_vio.angvariance[0] = 0.0;
    sensor_vio.angvariance[1] = 0.0;
    sensor_vio.angvariance[2] = 0.0;

    sensor_vio.gyrobias[0] = gyro_bias.x();
    sensor_vio.gyrobias[1] = gyro_bias.y();
    sensor_vio.gyrobias[2] = gyro_bias.z();

    sensor_vio.posoffset[0] = 0.0;
    sensor_vio.posoffset[1] = 0.0;
    sensor_vio.posoffset[2] = 0.0;

    sensor_vio.cam_imu_dt = state_server.imu_state.cam_imu_dt;

    sensor_vio.lag_ms = 0;
    if((imuSampleTime_ms-last_cam_fusion_ms) < 2000) {
        vio_publisher->publish(sensor_vio);
    }
}

void ekf_Core::StatusPublished(void)
{
    if(!statesInitialised) {
        return;
    }

    // Convert the IMU frame to the body frame.
    Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
    T_i_w.linear() = quaternionToRotation(state_server.imu_state.orientation).transpose();
    T_i_w.translation() = state_server.imu_state.position;

    geometry_msgs::msg::TransformStamped tf_msg;

    tf_msg = tf2::eigenToTransform(T_i_w);
    tf_msg.header.stamp = _node->now();
    tf_msg.header.frame_id = "world";
    tf_msg.child_frame_id = "odom";

    tf_broadcaster->sendTransform(tf_msg);

    // nav_msgs::msg::Odometry odom_msg;
    // odom_msg.header.stamp = _node->now();
    // odom_msg.header.frame_id = "world";
    // odom_msg.child_frame_id = "odom";

    // odom_msg.pose.pose = tf2::toMsg(T_i_w); 
    // odom_msg.twist.twist.linear = tf2::toMsg2(state_server.imu_state.velocity);

    // // Convert the covariance.
    // Eigen::Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
    // Eigen::Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
    // Eigen::Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
    // Eigen::Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
    // Eigen::Matrix<double, 6, 6> P_imu_pose = Eigen::Matrix<double, 6, 6>::Zero();
    // P_imu_pose << P_pp, P_po, P_op, P_oo;

    // for (int i = 0; i < 6; ++i) {
    //     for (int j = 0; j < 6; ++j) {
    //         odom_msg.pose.covariance[6*i+j] = P_imu_pose(i, j);
    //     }
    // }

    // // Construct the covariance for the velocity.
    // Eigen::Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         odom_msg.twist.covariance[i*6+j] = P_imu_vel(i, j);
    //     }
    // }

    // odometry_publisher->publish(odom_msg);

    if(path_count > 2) {
        path_count = 0;
        path_msg.header.stamp = _node->now();
        path_msg.header.frame_id = "world";

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header = path_msg.header;

        pose_msg.pose.position.x = state_server.imu_state.position(0);
        pose_msg.pose.position.y = state_server.imu_state.position(1);
        pose_msg.pose.position.z = state_server.imu_state.position(2);

        path_msg.poses.push_back(pose_msg);
        if(path_msg.poses.size() > 300) {
            path_msg.poses.erase(path_msg.poses.begin(), path_msg.poses.begin()+(path_msg.poses.size() - 300));
        }

        gd_path_msg.header = path_msg.header;

        geometry_msgs::msg::PoseStamped gd_pose_msg;

        gd_pose_msg.pose.position.x = gd_position(0);
        gd_pose_msg.pose.position.y = gd_position(1);
        gd_pose_msg.pose.position.z = gd_position(2);

        gd_path_msg.poses.push_back(gd_pose_msg);
        if(gd_path_msg.poses.size() > 300) {
            gd_path_msg.poses.erase(gd_path_msg.poses.begin(), gd_path_msg.poses.begin()+(gd_path_msg.poses.size() - 300));
        }

        path_publisher->publish(path_msg);
        gd_path_publisher->publish(gd_path_msg);
    }
    path_count++;

    pcl::PointCloud<pcl::PointXYZ> feature_msg_ptr;
    feature_msg_ptr.header.frame_id = "world";
    feature_msg_ptr.height = 1;
    for (const auto& item : map_server) {
        const auto& feature = item.second;
        if (feature.is_initialized) {
            Eigen::Vector3d feature_position = feature.position;
            feature_msg_ptr.points.push_back(pcl::PointXYZ(feature_position(0), feature_position(1), feature_position(2)));
        }
    }
    feature_msg_ptr.width = feature_msg_ptr.points.size();

    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(feature_msg_ptr, pc2_msg);
    pc2_msg.header.stamp = _node->now();
    pc2_msg.header.frame_id = "world";

    feature_publisher->publish(pc2_msg);

    if (tiltAlignComplete && (yawAlignComplete || !_param.external_att)) {
        if(debug_count > 60) {
            debug_count = 0;
            printf("Performance PC_INTERVAL\n");
            perf_print_all(PC_INTERVAL);
            printf("Performance PC_ELAPSED\n");
            perf_print_all(PC_ELAPSED);
            printf("Performance PC_COUNT\n");
            perf_print_all(PC_COUNT);
            printf("IMUavg:%.6fs, EKFavg:%.6fs, Cam-IMU dt:%.6fs\n", dtIMUavg, dtEkfAvg, state_server.imu_state.cam_imu_dt);
            printf("Track Rate:%.4f\n", tracking_rate);
            printf("Buffer size:CamB-%d CamA-%d IMU-%d ATT-%d Server-%d CamState-%d\n", 
                cam_feature_size, 
                (uint32_t)cam_msg_buffer.size(), 
                (uint32_t)imu_msg_buffer.size(), 
                (uint32_t)att_msg_buffer.size(), 
                (uint32_t)map_server.size(), 
                (uint32_t)state_server.cam_states.size());
            _image_track.printf_track_info();
        }
        debug_count++;
    }
}