#include <ekf_core/ekf_Core.h>
#include <boost/math/distributions/chi_squared.hpp>
#include <tf2_eigen/tf2_eigen.h>
// #include <pcl_conversions/pcl_conversions.h>
#include <string>
#include <cstdio>
#include <iostream>

using namespace math_lib;
using namespace msckf_core;
using namespace Eigen;

StateIDType IMUState::next_id = 0;
bool ekf_Core::thread_run = false;
std::map<int, double> ekf_Core::chi_squared_test_table;

ekf_Core::ekf_Core(std::shared_ptr<rclcpp::Node> node) :
    _node(node),
    _param(node),
    _image_track(node),
    update_overrun(false),
    cam_update(false),
    is_first_img(true),
    filter_error_count(0),
    statesInitialised(false),
    statesInitDynamics(false),
    firstInitTime_ms(0)
{
    accelmeter = Eigen::Vector3d::Zero();
    gyroscope = Eigen::Vector3d::Zero();

    tracking_rate = 0.0;
    mag_decl = 0.0;

    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(node);
    tf_aircraft    = std::make_unique<tf2_ros::TransformBroadcaster>(node);
}

ekf_Core::~ekf_Core(void)
{
    perf_free(_perf_Update);
    perf_free(_perf_CovariancePrediction);
    perf_free(_perf_FuseAttitude);
    perf_free(_perf_FuseCAM);
    perf_free(_perf_Interval);
    perf_free(_perf_Att_Interval);
    perf_free(_perf_CAM_Interval);
    perf_free(_perf_Remove_Interval);
    perf_free(_perf_Prune_Interval);
    perf_free(_perf_CheckMove_Error);
    perf_free(_perf_InitPos_Error);
    perf_free(_perf_Gating1_Error);
    perf_free(_perf_Gating2_Error);
    perf_free(_perf_Const_pos);
    perf_free(_perf_Cam_overdrop);
    perf_free(_perf_Update_overrun);
    perf_free(_perf_Reset_count);
}

void ekf_Core::shutdown(void)
{
    printf("Filter States: \n");
    printf("Gyro bias:[%.6f,%.6f,%.6f] \n", state_server.imu_state.gyro_bias(0), state_server.imu_state.gyro_bias(1), state_server.imu_state.gyro_bias(2));
    printf("Accl bias:[%.6f,%.6f,%.6f] \n", state_server.imu_state.acc_bias(0), state_server.imu_state.acc_bias(1), state_server.imu_state.acc_bias(2));
    printf("Cam-IMU sync:%.6f \n", state_server.imu_state.cam_imu_dt);
    printf("Run time:%.2fSec \n", (imuSampleTime_ms-ekfStartTime_ms)/1000.0);

    Eigen::Isometry3d t_cam0_imu_res = Eigen::Isometry3d::Identity();
    t_cam0_imu_res.rotate(state_server.imu_state.R_imu_cam0.transpose());
    t_cam0_imu_res.pretranslate(state_server.imu_state.t_cam0_imu);
    Eigen::Isometry3d t_imu_cam0_res = t_cam0_imu_res.inverse();
    
    printf("T_imu_cam0 : \n");
	std::cout << t_imu_cam0_res.linear() << std::endl;
  	std::cout << t_imu_cam0_res.translation().transpose() << std::endl;

    printf("Covariance : \n");
    printf("att   [%.5f, %.5f, %.5f]\n", state_server.state_cov(0, 0), state_server.state_cov(1, 1), state_server.state_cov(2, 2));
    printf("gbias [%.5f, %.5f, %.5f]\n", state_server.state_cov(3, 3), state_server.state_cov(4, 4), state_server.state_cov(5, 5));
    printf("vel   [%.5f, %.5f, %.5f]\n", state_server.state_cov(6, 6), state_server.state_cov(7, 7), state_server.state_cov(8, 8));
    printf("abias [%.5f, %.5f, %.5f]\n", state_server.state_cov(9, 9), state_server.state_cov(10, 10), state_server.state_cov(11, 11));
    printf("pos   [%.5f, %.5f, %.5f]\n", state_server.state_cov(12, 12), state_server.state_cov(13, 13), state_server.state_cov(14, 14));
    printf("erot  [%.5f, %.5f, %.5f]\n", state_server.state_cov(15, 15), state_server.state_cov(16, 16), state_server.state_cov(17, 17));
    printf("epos  [%.5f, %.5f, %.5f]\n", state_server.state_cov(18, 18), state_server.state_cov(19, 19), state_server.state_cov(20, 20));
    printf("dt    [%.8f]\n", state_server.state_cov(21, 21));
}

bool ekf_Core::setup_core(void)
{
    // Initialize the chi squared test table with confidence
    // level 0.95.
    for (int i = 1; i < 100; ++i) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_test_table[i] = boost::math::quantile(chi_squared_dist, 0.05);
    }

    map_server.clear();
    imu_msg_buffer.clear();
    att_msg_buffer.clear();
    cam_msg_buffer.clear();

    // sem_init(&update_sem, 0, 0);

    _image_track.init();
    _param.getParameter();

    vio_publisher      = _node->create_publisher<sprain_msgs::msg::SensorVio>(_param.vio_topics, 10);
    odometry_publisher = _node->create_publisher<nav_msgs::msg::Odometry>("/msckf_vio/odometry", 10);
    feature_publisher  = _node->create_publisher<sensor_msgs::msg::PointCloud2>("/msckf_vio/feature_point_cloud", 10);
    path_publisher     = _node->create_publisher<nav_msgs::msg::Path>("/msckf_vio/path", 10);
    gd_path_publisher  = _node->create_publisher<nav_msgs::msg::Path>("/msckf_vio/gd_path", 10);

    delta_subscription = _node->create_subscription<sprain_msgs::msg::SensorDelta>(
		"/fmu/sensor_delta/out",
		10,
		std::bind(&ekf_Core::delta_callback,
		this, std::placeholders::_1));

    imu_subscription = _node->create_subscription<sensor_msgs::msg::Imu>(
		_param.imu_topics,
		10,
		std::bind(&ekf_Core::imu_callback,
		this, std::placeholders::_1));
    
    ext_att_subscription = _node->create_subscription<sprain_msgs::msg::VehicleAttitude>(
		"/fmu/vehicle_attitude/out",
		10,
		std::bind(&ekf_Core::att_callback,
		this, std::placeholders::_1));

    ext_pos_subscription = _node->create_subscription<sprain_msgs::msg::LocalPosition>(
		"/fmu/local_position/out",
		10,
		std::bind(&ekf_Core::pos_callback,
		this, std::placeholders::_1));

    gd_pos_subscription = _node->create_subscription<geometry_msgs::msg::PointStamped>(
		"/leica/position",
		10,
		std::bind(&ekf_Core::gd_callback,
		this, std::placeholders::_1));

    // rclcpp::QoS qos_img(10);
  	// auto rmw_qos_profile = qos_img.get_rmw_qos_profile();

    // 订阅D435I双目图像
	// _cam0_img_sub.subscribe(_node, "/camera/infra1/image_rect_raw", rmw_qos_profile);
  	// _cam1_img_sub.subscribe(_node, "/camera/infra2/image_rect_raw", rmw_qos_profile);
    _cam0_img_sub.subscribe(_node, _param.cam0_topics);
  	_cam1_img_sub.subscribe(_node, _param.cam1_topics);

	// _image_sync = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(_cam0_img_sub, _cam1_img_sub, 10);
    _image_sync = std::make_shared<message_filters::Synchronizer<sync_pol>>(sync_pol(10), _cam0_img_sub, _cam1_img_sub);
  	_image_sync->registerCallback(std::bind(&ekf_Core::cam_callback, this,
                                         std::placeholders::_1,
                                         std::placeholders::_2,
                                         0, 1)); 

    _perf_CovariancePrediction = perf_alloc(PC_ELAPSED, "cov_ela");
    _perf_FuseAttitude         = perf_alloc(PC_ELAPSED, "att_ela");
    _perf_FuseCAM              = perf_alloc(PC_ELAPSED, "cam_ela");
    _perf_Update               = perf_alloc(PC_ELAPSED, "update");

    _perf_Interval        = perf_alloc(PC_INTERVAL, "update_int");
    _perf_Att_Interval    = perf_alloc(PC_INTERVAL, "att_int");
    _perf_CAM_Interval    = perf_alloc(PC_INTERVAL, "cam_int");
    _perf_Remove_Interval = perf_alloc(PC_INTERVAL, "remove_int");
    _perf_Prune_Interval  = perf_alloc(PC_INTERVAL, "prune_int");

    _perf_CheckMove_Error = perf_alloc(PC_COUNT, "checkmove");
    _perf_InitPos_Error   = perf_alloc(PC_COUNT, "initpos");
    _perf_Gating1_Error   = perf_alloc(PC_COUNT, "gating1");
    _perf_Gating2_Error   = perf_alloc(PC_COUNT, "gating2");
    _perf_Const_pos       = perf_alloc(PC_COUNT, "const");
    _perf_Cam_overdrop    = perf_alloc(PC_COUNT, "camdrop");
    _perf_Update_overrun  = perf_alloc(PC_COUNT, "overrun");
    _perf_Update_failed   = perf_alloc(PC_COUNT, "ekf_failed");
    _perf_Reset_count     = perf_alloc(PC_COUNT, "reset");

    return true;
}

void ekf_Core::delta_callback(const sprain_msgs::msg::SensorDelta::ConstSharedPtr data)
{
	if(!data->healthy) {
		return;
	}
	if(!data->calibrated) {
		return;
	}
	// default delta dtis 0.005f
	if(data->delta_ang_dt < 0.001 || data->delta_ang_dt > 0.03) {
		return;
	}

    perf_count(_perf_Interval, timestamp());

    imuDataNew.time_us = data->timestamp;
    imuDataNew.delAng = Vector3d(data->delta_angle[0]/(double)data->delta_ang_dt,
						    data->delta_angle[1]/(double)data->delta_ang_dt,
						    data->delta_angle[2]/(double)data->delta_ang_dt);
    imuDataNew.delVel = Vector3d(data->delta_velocity[0]/(double)data->delta_vel_dt,
							data->delta_velocity[1]/(double)data->delta_vel_dt,
							data->delta_velocity[2]/(double)data->delta_vel_dt);
    imuDataNew.delAngDT = std::max((double)data->delta_ang_dt, 1.0e-4);
    imuDataNew.delVelDT = std::max((double)data->delta_vel_dt, 1.0e-4);

    _image_track.feed_imu(imuDataNew, state_server.imu_state.gyro_bias);

    // calculate the achieved average time step rate for the EKF using a combination spike and LPF
    double dtNow = constrain_double(0.5*(imuDataNew.delAngDT+imuDataNew.delVelDT),0.5 * dtEkfAvg, 2.0 * dtEkfAvg);
    dtEkfAvg = 0.98 * dtEkfAvg + 0.02 * dtNow;
    imu_queue_mtx.lock();
    imu_msg_buffer.push_back(imuDataNew);
    imu_queue_mtx.unlock();
}

void ekf_Core::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr data)
{
    perf_count(_perf_Interval, timestamp());

    imuDataNew.time_us = RCL_NS_TO_US(rclcpp::Time(data->header.stamp).nanoseconds());
    imuDataNew.delAng = Vector3d(data->angular_velocity.x,
						    data->angular_velocity.y,
						    data->angular_velocity.z);
    imuDataNew.delVel = Vector3d(data->linear_acceleration.x,
							data->linear_acceleration.y,
							data->linear_acceleration.z);
    
    if(last_imu_sample == 0) {
        imuDataNew.delAngDT = 0.005;
        imuDataNew.delVelDT = 0.005;
        last_imu_sample = imuDataNew.time_us;
    } else {
        imuDataNew.delAngDT = (imuDataNew.time_us - last_imu_sample) * 1e-6;
        imuDataNew.delVelDT = (imuDataNew.time_us - last_imu_sample) * 1e-6;
        last_imu_sample = imuDataNew.time_us;
    }

    _image_track.feed_imu(imuDataNew, state_server.imu_state.gyro_bias);

    // calculate the achieved average time step rate for the EKF using a combination spike and LPF
    double dtNow = constrain_double(0.5*(imuDataNew.delAngDT+imuDataNew.delVelDT),0.5 * dtEkfAvg, 2.0 * dtEkfAvg);
    dtEkfAvg = 0.98 * dtEkfAvg + 0.02 * dtNow;
    imu_queue_mtx.lock();
    imu_msg_buffer.push_back(imuDataNew);
    imu_queue_mtx.unlock();
}

void ekf_Core::cam_callback(const sensor_msgs::msg::Image::ConstSharedPtr img0, const sensor_msgs::msg::Image::ConstSharedPtr img1, int cam_id0, int cam_id1)
{
    _image_track.feed_stereo(img0, img1);

    const sprain_msgs::msg::CameraMeasurement data = _image_track.get_feature();

    camDataNew.time_us = data.timestamp + state_server.imu_state.cam_imu_dt*1000000;
    camDataNew.seq++;

    camDataNew.cam_feature.clear();
    for(uint32_t i=0; i<data.features.size(); i++) {
        camDataNew.cam_feature.push_back(sprain_msgs::msg::FeatureMeasurement());
        camDataNew.cam_feature[i] = data.features[i];
    }

    if(is_first_img) {
        is_first_img = false;
        state_server.imu_state.timestamp = camDataNew.time_us;
    }

    StatePublished();

    camera_queue_mtx.lock();
    cam_msg_buffer.push_back(camDataNew);
    camera_queue_mtx.unlock();

    if(thread_run) {
        return;
    }

    std::thread thread([&] {
        UpdateFilter();
    });

    thread.detach();
}

void ekf_Core::att_callback(const sprain_msgs::msg::VehicleAttitude::ConstSharedPtr data)
{
    if(!statesInitialised) {
        return;
    }
    attDataNew.time_us = data->timestamp;

    attDataNew.quat = Vector4d(data->q[0],
							data->q[1],
							data->q[2],
                            data->q[3]);
    attitude_queue_mtx.lock();
    att_msg_buffer.push_back(attDataNew);
    attitude_queue_mtx.unlock();
}

void ekf_Core::pos_callback(const sprain_msgs::msg::LocalPosition::ConstSharedPtr data)
{
    // Convert the IMU frame to the body frame.
    Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
    T_i_w.linear() = quaternionToRotation(attDataNew.quat).transpose();
    T_i_w.translation() = Eigen::Vector3d(data->pos[0],data->pos[1],data->pos[2]);

    geometry_msgs::msg::TransformStamped tf_msg;

    tf_msg = tf2::eigenToTransform(T_i_w);
    tf_msg.header.stamp = _node->now();
    tf_msg.header.frame_id = "world";
    tf_msg.child_frame_id = "aircraft";

    tf_aircraft->sendTransform(tf_msg);
}

void ekf_Core::gd_callback(const geometry_msgs::msg::PointStamped::ConstSharedPtr data)
{
    geometry_msgs::msg::PoseStamped gd_pose_msg;

    if(!is_gd_position_init) {
        gd_position_orign(0) = data->point.x;
        gd_position_orign(1) = data->point.y;
        gd_position_orign(2) = data->point.z;
        is_gd_position_init = true;
    }

    gd_position(0) = data->point.x-gd_position_orign(0);
    gd_position(1) = data->point.y-gd_position_orign(1);
    gd_position(2) = data->point.z-gd_position_orign(2);
}

bool ekf_Core::RestartFilter(void)
{
    statesInitialised = false;
    statesInitDynamics = true;
    perf_count(_perf_Reset_count, timestamp());
    printf("Filter Error Online Reset... \n");
    return true;
}

void ekf_Core::InitialiseVariables()
{
    ekf_Fusion::InitialiseVariables();

    state_server.imu_state.R_imu_cam0 = _param.R_imu_cam0;
    state_server.imu_state.t_cam0_imu = _param.t_cam0_imu;

    CAMState::T_cam0_cam1 = _param.T_cam0_cam1;

    lastKnownPositionNE = Vector3d::Zero();

    is_first_img = true;
    magYawResetRequest = false;

    tiltAlignComplete = false;
    yawAlignComplete = false;
    delAngBiasLearned = false;

    statesInitialised = false;
    imuSampleTime_ms = timestamp()/1000;
    ekfStartTime_ms = imuSampleTime_ms;
    last_cam_fusion_ms = 0;

    tracking_rate = 0.0;
}

bool ekf_Core::InitialiseFilter(void)
{
    if (imu_msg_buffer.size() < 100) {
        return false;
    }

    // 等待1秒，等待传感器数据缓存填充
    if (firstInitTime_ms == 0) {
        firstInitTime_ms = imuSampleTime_ms;
        return true;
    } else if (imuSampleTime_ms - firstInitTime_ms < 1000) {
        return true;
    }

    InitialiseVariables();
    CovarianceInit();

    Vector3d sum_angular_vel = Vector3d::Zero();
    Vector3d sum_linear_acc = Vector3d::Zero();

    for (const auto& imu_msg : imu_msg_buffer) {
        sum_angular_vel += imu_msg.delAng;
        sum_linear_acc += imu_msg.delVel;
    }

    state_server.imu_state.gyro_bias = Vector3d::Zero(); // sum_angular_vel / imu_msg_buffer.size();
    state_server.imu_state.acc_bias = Vector3d::Zero();

    accelmeter = sum_linear_acc / imu_msg_buffer.size();

    // 初始化倾斜角旋转
    if(statesInitDynamics) {
        state_server.imu_state.orientation = attDataDelayed.quat;
    } else {
        Eigen::Quaterniond q0_i_w = Eigen::Quaterniond::FromTwoVectors(accelmeter, -ekf_Fusion::gravity);
        // JPL四元数， world->body
        state_server.imu_state.orientation = rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());
    }

    statesInitialised = true;
    statesInitDynamics = false;

    imu_msg_buffer.clear();
    cam_msg_buffer.clear();

    // 等待惯导数据缓存填满
    return true;
}

void ekf_Core::UpdateFilter(void)
{
    // sem_wait(&update_sem);

    thread_run = true;

    // 获取滤波器运行时间参数
    imuSampleTime_ms = timestamp()/1000;
    
    if (!statesInitialised) {
        InitialiseFilter();
        thread_run = false;
        return;
    }

    int used_cam_msg_cntr = 0;
    const uint64_t start_ts = timestamp();
    perf_begin(_perf_Update, timestamp());
    cam_feature_size = cam_msg_buffer.size();
    for (const auto& cam_msg : cam_msg_buffer) {
        if((cam_msg.seq - camDataDelayed.seq) != 1) {
            perf_count(_perf_Cam_overdrop, timestamp());
        }

        camDataDelayed.time_us = cam_msg.time_us;
        camDataDelayed.seq = cam_msg.seq;
        camDataDelayed.cam_feature.clear();
        for(uint32_t i=0; i<cam_msg.cam_feature.size(); i++) {
            camDataDelayed.cam_feature.push_back(sprain_msgs::msg::FeatureMeasurement());
            camDataDelayed.cam_feature[i] = cam_msg.cam_feature[i];
        }

        // 读取陀螺仪和加速计数据
        accelmeter       = imuDataNew.delVel;
        gyroscope        = imuDataNew.delAng;
        
        // 检测初始化状态
        checkAttitudeAlignmentStatus();
        // 检测陀螺仪状态
        checkGyroCalStatus();

        int used_imu_msg_cntr = 0;
        for (const auto& imu_msg : imu_msg_buffer) {
            uint64_t imu_time = imu_msg.time_us;
            if (imu_time <= state_server.imu_state.timestamp) {
                ++used_imu_msg_cntr;
                continue;
            }
            if (imu_time > camDataDelayed.time_us) {
                break;
            }

            imuDataDelayed = imu_msg;

            // 修正惯导数据偏置
            delAngCorrected = imuDataDelayed.delAng - state_server.imu_state.gyro_bias;
            delVelCorrected = imuDataDelayed.delVel - state_server.imu_state.acc_bias;
            dtIMUavg = (imuDataDelayed.time_us-state_server.imu_state.timestamp) * 1e-6;
            dtIMUavg = constrain_double(dtIMUavg, 0.001, 0.02);
            // 判断是否有绝对位置及速度融合
            UpdateStrapdownRungeKuttaNED();

            perf_begin(_perf_CovariancePrediction, timestamp());
            CovariancePrediction();
            perf_end(_perf_CovariancePrediction, timestamp());

            // Update the state info
            state_server.imu_state.timestamp = imuDataDelayed.time_us;
            ++used_imu_msg_cntr;
        }
        
        // Set the state ID for the new IMU state.
        state_server.imu_state.id = IMUState::next_id++;
        // Remove all used IMU msgs.
        imu_queue_mtx.lock();
        imu_msg_buffer.erase(imu_msg_buffer.begin(), imu_msg_buffer.begin()+used_imu_msg_cntr);
        imu_queue_mtx.unlock();

        ControlConstFuse();

        ControlCamFuse();

        ControlAttFuse();

        _image_track.cam_imu_sync(state_server.imu_state.cam_imu_dt);

        ++used_cam_msg_cntr;
    }
    perf_end(_perf_Update, timestamp());

    if((timestamp()-start_ts) > 33000) {
        update_overrun = true;
        perf_count(_perf_Update_overrun, timestamp());
    }
    
    camera_queue_mtx.lock();
    cam_msg_buffer.erase(cam_msg_buffer.begin(), cam_msg_buffer.begin()+used_cam_msg_cntr);
    camera_queue_mtx.unlock();

    updateFilterStatus();

    StatusPublished();

    thread_run = false;
    update_overrun = false;
}

void ekf_Core::checkAttitudeAlignmentStatus()
{
    if (!tiltAlignComplete) {
        tiltErrorVariance += (state_server.state_cov(0,0)+state_server.state_cov(1,1)+state_server.state_cov(2,2));
        if (tiltErrorVariance < sq(radians(5.0))) {
            tiltAlignComplete = true;
            printf("Nav tilt OK\n");
        }
    }

    if (!yawAlignComplete && tiltAlignComplete && _param.usecompass) {
        magYawResetRequest = true;
    }
}

void ekf_Core::checkGyroCalStatus(void)
{
    // check delta angle bias variances
    const double delAngBiasVarMax = sq((double)radians(0.15 * dtEkfAvg));
    delAngBiasLearned = (state_server.state_cov(3, 3) <= delAngBiasVarMax) &&
                        (state_server.state_cov(4, 4) <= delAngBiasVarMax) &&
                        (state_server.state_cov(5, 5) <= delAngBiasVarMax);
}