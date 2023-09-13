#include <ekf_core/ekf_Param.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

ekf_Parameter::ekf_Parameter(std::shared_ptr<rclcpp::Node> node):
    _node(node)
{
	T_cam0_cam1 = Eigen::Isometry3d::Identity();
}

ekf_Parameter::~ekf_Parameter(void)
{
    
}

void ekf_Parameter::getParameter(void)
{   
	vector<double> t_imu_cam_array(16);
	setNgetNodeParameter(t_imu_cam_array, "T_imu_cam", vector<double>(16));
	cv::Mat T_imu_cam0_mat = cv::Mat(t_imu_cam_array).clone().reshape(1, 4);

	Eigen::Isometry3d T_imu_cam0;
	T_imu_cam0.linear()(0, 0)   = T_imu_cam0_mat.at<double>(0, 0);
	T_imu_cam0.linear()(0, 1)   = T_imu_cam0_mat.at<double>(0, 1);
	T_imu_cam0.linear()(0, 2)   = T_imu_cam0_mat.at<double>(0, 2);
	T_imu_cam0.linear()(1, 0)   = T_imu_cam0_mat.at<double>(1, 0);
	T_imu_cam0.linear()(1, 1)   = T_imu_cam0_mat.at<double>(1, 1);
	T_imu_cam0.linear()(1, 2)   = T_imu_cam0_mat.at<double>(1, 2);
	T_imu_cam0.linear()(2, 0)   = T_imu_cam0_mat.at<double>(2, 0);
	T_imu_cam0.linear()(2, 1)   = T_imu_cam0_mat.at<double>(2, 1);
	T_imu_cam0.linear()(2, 2)   = T_imu_cam0_mat.at<double>(2, 2);
	T_imu_cam0.translation()(0) = T_imu_cam0_mat.at<double>(0, 3);
	T_imu_cam0.translation()(1) = T_imu_cam0_mat.at<double>(1, 3);
	T_imu_cam0.translation()(2) = T_imu_cam0_mat.at<double>(2, 3);
	Eigen::Isometry3d T_cam0_imu = T_imu_cam0.inverse();

	R_imu_cam0 = T_cam0_imu.linear().transpose();
	t_cam0_imu = T_cam0_imu.translation();

	vector<double> t_cn_cnm1_array(16);
	setNgetNodeParameter(t_cn_cnm1_array, "T_cn_cnm1", vector<double>(16));
	cv::Mat cam0_cam1_mat = cv::Mat(t_cn_cnm1_array).clone().reshape(1, 4);
	T_cam0_cam1.linear()(0, 0)   = cam0_cam1_mat.at<double>(0, 0);
	T_cam0_cam1.linear()(0, 1)   = cam0_cam1_mat.at<double>(0, 1);
	T_cam0_cam1.linear()(0, 2)   = cam0_cam1_mat.at<double>(0, 2);
	T_cam0_cam1.linear()(1, 0)   = cam0_cam1_mat.at<double>(1, 0);
	T_cam0_cam1.linear()(1, 1)   = cam0_cam1_mat.at<double>(1, 1);
	T_cam0_cam1.linear()(1, 2)   = cam0_cam1_mat.at<double>(1, 2);
	T_cam0_cam1.linear()(2, 0)   = cam0_cam1_mat.at<double>(2, 0);
	T_cam0_cam1.linear()(2, 1)   = cam0_cam1_mat.at<double>(2, 1);
	T_cam0_cam1.linear()(2, 2)   = cam0_cam1_mat.at<double>(2, 2);
	T_cam0_cam1.translation()(0) = cam0_cam1_mat.at<double>(0, 3);
	T_cam0_cam1.translation()(1) = cam0_cam1_mat.at<double>(1, 3);
	T_cam0_cam1.translation()(2) = cam0_cam1_mat.at<double>(2, 3);

	setNgetNodeParameter(external_att, "ext_att", false);

	setNgetNodeParameter(imu_topics,  "imu_topic",  string("/fmu/imu/out"));
	setNgetNodeParameter(vio_topics,  "vio_topic",  string("/fmu/sensor_vio/in"));
	setNgetNodeParameter(cam0_topics, "cam0_topic", string("/camera/infra1/image_rect_raw"));
	setNgetNodeParameter(cam1_topics, "cam1_topic", string("/camera/infra2/image_rect_raw"));

    printf("===================SPRain Navigation System Parameter=================\n");
	printf("T_imu_cam0 :\n");
	cout << T_imu_cam0.linear() << endl;
  	cout << T_imu_cam0.translation().transpose() << endl;

	printf("T_cam0_cam1 :\n");
	cout << T_cam0_cam1.linear() << endl;
  	cout << T_cam0_cam1.translation().transpose() << endl;

	printf("Ext_Att :%d\n", external_att);
	printf("IMU  Topic: %s \n", imu_topics.c_str());
	printf("VIO  Topic: %s \n", vio_topics.c_str());
	printf("CAM0 Topic: %s \n", cam0_topics.c_str());
	printf("CAM1 Topic: %s \n", cam1_topics.c_str());
}

template<class T>
void ekf_Parameter::setNgetNodeParameter(T& param, const std::string& param_name, const T& default_value, const rcl_interfaces::msg::ParameterDescriptor &parameter_descriptor)
{
	rclcpp::ParameterValue result_value = rclcpp::ParameterValue(default_value);

	try
	{
		if (!_node->has_parameter(param_name))
		{
			result_value = _node->declare_parameter(param_name, result_value, parameter_descriptor);
		}
		else
		{
			result_value = _node->get_parameter(param_name).get_parameter_value();
		}
		param = result_value.get<T>();
	}
	catch(const std::exception& e)
	{
		RCLCPP_WARN_STREAM(_node->get_logger(), "Could not set param: " << param_name << " with " << 
							rclcpp::Parameter(param_name, rclcpp::ParameterValue(default_value)).value_to_string() << e.what());
		param = default_value;
	}
}
