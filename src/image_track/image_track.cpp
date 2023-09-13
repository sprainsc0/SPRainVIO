#include <cstdio>
#include <atomic>
#include <getopt.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <csignal>
#include <termios.h>
#include <memory>
#include <Eigen/Dense>

#include <image_track.h>

#include <sensor_msgs/image_encodings.hpp>

using namespace std::chrono_literals;
using namespace std;
using namespace cv;
using namespace Eigen;

ImageProcessor::ImageProcessor(std::shared_ptr<rclcpp::Node> node) : 
  _node(node),
  is_first_img(true),
  cam_imu_sync_dt(-0.009),
  dtime(0.033),
  gyro_bias(0.0,0.0,0.0),
  prev_features_ptr(new GridFeatures()),
  curr_features_ptr(new GridFeatures())
{
	
}

ImageProcessor::~ImageProcessor(void)
{
	perf_free(_perf_feed_cam);
	perf_free(_perf_track);
}

void ImageProcessor::init(void)
{
	// 获取参数
	getParameter();

	// 创建特征点检测
	if(!processor_config.use_cuda) {
		detector_ptr = FastFeatureDetector::create(processor_config.fast_threshold);
	} 
#if defined(USE_CUDA)
	else {
		cuda_detector_ptr = cv::cuda::FastFeatureDetector::create(processor_config.fast_threshold);
	}
#endif

	// Create image transport
  	image_transport::ImageTransport it(_node);

  	// debug_stereo_pub = image_transport::create_publisher(_node, "img/debug_stereo_image");
	debug_stereo_pub = it.advertise("img/debug_stereo_image", 2);

	imu_msg_buffer.clear();

	_perf_feed_cam = perf_alloc(PC_INTERVAL, "feed_cam");
	_perf_track = perf_alloc(PC_ELAPSED, "track_ela");
	_perf_imu_error = perf_alloc(PC_COUNT, "imu_error");
	_perf_image_drop = perf_alloc(PC_COUNT, "image_drop");
}

void ImageProcessor::feed_imu(const nav::imu_elements data, Vector3d bias)
{
    if (is_first_img){
		return;
	}
	
	nav::imu_elements imu_data = data;
	gyro_bias = Vec3f(bias(0), bias(1), bias(2));
	// 对IMU数据时间戳进行同步
	imu_data.time_us -= cam_imu_sync_dt * 1e6;
    imu_msg_buffer.push_back(imu_data);
}

void ImageProcessor::feed_stereo(const sensor_msgs::msg::Image::ConstSharedPtr& img0, 
        const sensor_msgs::msg::Image::ConstSharedPtr& img1)
{
	cam0_curr_img_ptr = cv_bridge::toCvShare(img0, sensor_msgs::image_encodings::MONO8);
  	cam1_curr_img_ptr = cv_bridge::toCvShare(img1, sensor_msgs::image_encodings::MONO8);

	perf_count(_perf_feed_cam, timestamp());

	perf_begin(_perf_track, timestamp());
	// 建立光流金字塔
	createImagePyramids();

	if(is_first_img) {
		initializeFirstFrame();
		is_first_img = false;
		drawFeaturesStereo();
	} else {
		trackFeatures();

		addNewFeatures();

		pruneGridFeatures();

		drawFeaturesStereo();
	}

	calculate_features();

	// Update the previous image and previous features.
	cam0_prev_img_ptr = cam0_curr_img_ptr;
	prev_features_ptr = curr_features_ptr;
	if(!processor_config.use_cuda) {
		std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);
	}

	// Initialize the current features to empty vectors.
	curr_features_ptr.reset(new GridFeatures());
	for (int code = 0; code < processor_config.grid_row*processor_config.grid_col; ++code) {
		(*curr_features_ptr)[code] = vector<FeatureMetaData>(0);
	}

	perf_end(_perf_track, timestamp());
}

void ImageProcessor::createImagePyramids()
{
#if defined(USE_CUDA)
	if(processor_config.use_cuda) {
		return;
	}
#endif

	const Mat& curr_cam0_img = cam0_curr_img_ptr->image;
	buildOpticalFlowPyramid(
		curr_cam0_img, curr_cam0_pyramid_,
		Size(processor_config.patch_size, processor_config.patch_size),
		processor_config.pyramid_levels, true, BORDER_REFLECT_101,
		BORDER_CONSTANT, false);

	const Mat& curr_cam1_img = cam1_curr_img_ptr->image;
	buildOpticalFlowPyramid(
		curr_cam1_img, curr_cam1_pyramid_,
		Size(processor_config.patch_size, processor_config.patch_size),
		processor_config.pyramid_levels, true, BORDER_REFLECT_101,
		BORDER_CONSTANT, false);
}

// 相机参数和观测到点坐标位置计算cam1坐标位置
void ImageProcessor::undistortPoints(const vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    vector<cv::Point2f>& pts_out,
    const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics) 
{

  	if(pts_in.size() == 0){
	  return;
  	} 

	const cv::Matx33d K(
		intrinsics[0], 0.0, intrinsics[2],
		0.0, intrinsics[1], intrinsics[3],
		0.0, 0.0, 1.0);

	const cv::Matx33d K_new(
		new_intrinsics[0], 0.0, new_intrinsics[2],
		0.0, new_intrinsics[1], new_intrinsics[3],
		0.0, 0.0, 1.0);

	if (distortion_model == "radtan") {
		cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
							rectification_matrix, K_new);
	} else if (distortion_model == "equidistant") {
		cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
									rectification_matrix, K_new);
	} else {
		// ROS_WARN_ONCE("The model %s is unrecognized, use radtan instead...", distortion_model.c_str());
		cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
							rectification_matrix, K_new);
	}
}

vector<cv::Point2f> ImageProcessor::distortPoints(const vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const string& distortion_model,
    const cv::Vec4d& distortion_coeffs) 
{

	const cv::Matx33d K(intrinsics[0], 0.0, intrinsics[2],
						0.0, intrinsics[1], intrinsics[3],
						0.0, 0.0, 1.0);

  	vector<cv::Point2f> pts_out;
	if (distortion_model == "radtan") {
		vector<cv::Point3f> homogenous_pts;
		cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
		cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
						distortion_coeffs, pts_out);
	} else if (distortion_model == "equidistant") {
		cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
	} else {
		// ROS_WARN_ONCE("The model %s is unrecognized, using radtan instead...", distortion_model.c_str());
		vector<cv::Point3f> homogenous_pts;
		cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
		cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K, distortion_coeffs, pts_out);
	}

	return pts_out;
}

void ImageProcessor::stereoMatch(const vector<cv::Point2f>& cam0_points,
    vector<cv::Point2f>& cam1_points,
    vector<unsigned char>& inlier_markers) 
{

	if (cam0_points.size() == 0){
		return;
	} 

	if(cam1_points.size() == 0) {
		// 将cam0_points特征点通过外参旋转初始化cam1_points特征点
		const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
		vector<cv::Point2f> cam0_points_undistorted;
		// 相机参数和观测到点坐标位置计算实际坐标位置
		undistortPoints(cam0_points, cam0_intrinsics, cam0_distortion_model,
						cam0_distortion_coeffs, cam0_points_undistorted,
						R_cam0_cam1);
		// 相机参数和实际坐标位置计算观测到点坐标位置
		cam1_points = distortPoints(cam0_points_undistorted, cam1_intrinsics,
									cam1_distortion_model, cam1_distortion_coeffs);
	}

	// LK-光流算法跟踪特征点
	if(!processor_config.use_cuda) {
		calcOpticalFlowPyrLK(curr_cam0_pyramid_, curr_cam1_pyramid_,
			cam0_points, cam1_points,
			inlier_markers, noArray(),
			Size(processor_config.patch_size, processor_config.patch_size),
			processor_config.pyramid_levels,
			TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
						processor_config.max_iteration,
						processor_config.track_precision),
			cv::OPTFLOW_USE_INITIAL_FLOW);

	} 
#if defined(USE_CUDA)
	else {
		const Mat& curr_cam0_img = cam0_curr_img_ptr->image;
		const Mat& curr_cam1_img = cam1_curr_img_ptr->image;
		
		cv::cuda::GpuMat cur_gpu_img0(curr_cam0_img);
		cv::cuda::GpuMat cur_gpu_img1(curr_cam1_img);
		cv::cuda::GpuMat cur_gpu_pts0(cam0_points);
		cv::cuda::GpuMat cur_gpu_pts1(cam1_points);
		cv::cuda::GpuMat gpu_status;

		cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
		cv::Size(processor_config.patch_size, processor_config.patch_size), 
		processor_config.pyramid_levels,
		processor_config.max_iteration);
		d_pyrLK_sparse->calc(cur_gpu_img0, cur_gpu_img1, cur_gpu_pts0, cur_gpu_pts1, gpu_status);

		vector<cv::Point2f> tmp1_cur_pts(cur_gpu_pts1.cols);
		cur_gpu_pts1.download(tmp1_cur_pts);
		cam1_points = tmp1_cur_pts;

		vector<uchar> tmp_status(gpu_status.cols);
		gpu_status.download(tmp_status);
		inlier_markers = tmp_status;
	}
#endif

	// 检测剔除超出图像范围的特征点
	for(uint32_t i = 0; i < cam1_points.size(); ++i) {
		if(inlier_markers[i] == 0){
			continue;
		} 

		if(cam1_points[i].y < 0 ||
			cam1_points[i].y > cam1_curr_img_ptr->image.rows-1 ||
			cam1_points[i].x < 0 ||
			cam1_points[i].x > cam1_curr_img_ptr->image.cols-1) {
			inlier_markers[i] = 0;
		}
		
	}

	// 计算cam0和cam1的旋转平移关系
	const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
	const cv::Vec3d t_cam0_cam1 = R_cam1_imu.t() * (t_cam0_imu-t_cam1_imu);
	// 计算本质矩阵E
	const cv::Matx33d t_cam0_cam1_hat(
		0.0, -t_cam0_cam1[2], t_cam0_cam1[1],
		t_cam0_cam1[2], 0.0, -t_cam0_cam1[0],
		-t_cam0_cam1[1], t_cam0_cam1[0], 0.0);
	const cv::Matx33d E = t_cam0_cam1_hat * R_cam0_cam1;

	// F根据本质矩阵进一步剔除异常值
	vector<cv::Point2f> cam0_points_undistorted(0);
	vector<cv::Point2f> cam1_points_undistorted(0);
	undistortPoints(
		cam0_points, cam0_intrinsics, cam0_distortion_model,
		cam0_distortion_coeffs, cam0_points_undistorted);
	undistortPoints(
		cam1_points, cam1_intrinsics, cam1_distortion_model,
		cam1_distortion_coeffs, cam1_points_undistorted);
	
	double norm_pixel_unit = 4.0 / (
		cam0_intrinsics[0]+cam0_intrinsics[1]+
		cam1_intrinsics[0]+cam1_intrinsics[1]);

	for (uint32_t i = 0; i < cam0_points_undistorted.size(); ++i) {
		if (inlier_markers[i] == 0) {
			continue;
		}
		// 转化为齐次坐标
		cv::Vec3d pt0(cam0_points_undistorted[i].x,
			cam0_points_undistorted[i].y,
			1.0);
		cv::Vec3d pt1(cam1_points_undistorted[i].x,
			cam1_points_undistorted[i].y,
			1.0);
		// 重投影误差剔除不符合的特征点对
		cv::Vec3d epipolar_line = E * pt0;
		double error = fabs((pt1.t() * epipolar_line)[0]) / sqrt(
			epipolar_line[0]*epipolar_line[0]+
			epipolar_line[1]*epipolar_line[1]);
		if (error > processor_config.stereo_threshold*norm_pixel_unit){
			inlier_markers[i] = 0;
		}
	}
}

void ImageProcessor::initializeFirstFrame() 
{
	// 计算每个网格尺寸
	const Mat& img = cam0_curr_img_ptr->image;
	static int grid_height = img.rows / processor_config.grid_row;
	static int grid_width = img.cols / processor_config.grid_col;

	// step1: 第一帧cam0图像提取特征点
	vector<KeyPoint> new_features(0);
	if(!processor_config.use_cuda) {
		detector_ptr->detect(img, new_features);
	} 
#if defined(USE_CUDA)
	else {
		cv::cuda::GpuMat gFrame(img);
		cuda_detector_ptr->detect(gFrame, new_features);
	}
#endif

	// step2: 针对新提取的特征点匹配双目特征点
	vector<cv::Point2f> cam0_points(new_features.size());
	for (uint32_t i = 0; i < new_features.size(); ++i) {
		cam0_points[i] = new_features[i].pt;
	}

	vector<cv::Point2f> cam1_points(0);
	vector<unsigned char> inlier_markers(0);
	
	// 匹配特征点并进行标记
	stereoMatch(cam0_points, cam1_points, inlier_markers);

	vector<cv::Point2f> cam0_inliers(0);
	vector<cv::Point2f> cam1_inliers(0);
	vector<float> response_inliers(0);

	for (uint32_t i = 0; i < inlier_markers.size(); ++i) {
		if (inlier_markers[i] == 0) continue;
		cam0_inliers.push_back(cam0_points[i]); 
		cam1_inliers.push_back(cam1_points[i]);
		response_inliers.push_back(new_features[i].response); // 特征点强度
	}

	/*step3：将左右目特征点划分在图像网格中，用curr_features_ptr存储，
	 *       如果提取的特征点数量大于grid_min_feature_num，则按照response筛选
	 */
	GridFeatures grid_new_features;
	// 画面分为20个网格
	for (int code = 0; code < processor_config.grid_row*processor_config.grid_col; ++code) {
		grid_new_features[code] = vector<FeatureMetaData>(0);
	}

	for (uint32_t i = 0; i < cam0_inliers.size(); ++i) {
		const cv::Point2f& cam0_point = cam0_inliers[i];
		const cv::Point2f& cam1_point = cam1_inliers[i];
		const float& response = response_inliers[i];

		// 计算特征点位于哪个网格内
		int row = static_cast<int>(cam0_point.y / grid_height);
		int col = static_cast<int>(cam0_point.x / grid_width);
		int code = row*processor_config.grid_col + col;

		FeatureMetaData new_feature;
		new_feature.response = response;
		new_feature.cam0_point = cam0_point;
		new_feature.cam1_point = cam1_point;
		grid_new_features[code].push_back(new_feature);
	}

	// 根据response进行特征点排序
	for (auto& item : grid_new_features) {
		std::sort(item.second.begin(), item.second.end(),
			&ImageProcessor::featureCompareByResponse);
	}

	// Collect new features within each grid with high response.
	for (int code = 0; code < processor_config.grid_row*processor_config.grid_col; ++code) {
		vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
		vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

		//1.当新提的点>=每个格子要求的最小特征数量，则按照响应值选择新提的特征点，直到满足要求
		//2.当新提的点<每个格子要求的最小特征数量，则选择所有新提的点

		for (uint32_t k = 0; k < (uint32_t)processor_config.grid_min_feature_num &&
			k < new_features_this_grid.size(); ++k) {
			// 将新的特征点对加入curr_features_ptr，id++
			features_this_grid.push_back(new_features_this_grid[k]);
			features_this_grid.back().id = next_feature_id++;
			features_this_grid.back().lifetime = 1;
		}
	}
}

void ImageProcessor::predictFeatureTracking(const vector<cv::Point2f>& input_pts,
    const cv::Matx33f& R_p_c,
    const cv::Vec4d& intrinsics,
    vector<cv::Point2f>& compensated_pts) 
{

	// Return directly if there are no input features.
	if (input_pts.size() == 0) {
		compensated_pts.clear();
		return;
	}
	compensated_pts.resize(input_pts.size());

	// Intrinsic matrix.
	cv::Matx33f K(
		intrinsics[0], 0.0, intrinsics[2],
		0.0, intrinsics[1], intrinsics[3],
		0.0, 0.0, 1.0);
	cv::Matx33f H = K * R_p_c * K.inv();

	for (uint32_t i = 0; i < input_pts.size(); ++i) {
		cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
		cv::Vec3f p2 = H * p1;
		compensated_pts[i].x = p2[0] / p2[2];
		compensated_pts[i].y = p2[1] / p2[2];
	}
}

void ImageProcessor::integrateImuData(Matx33f& cam0_R_p_c, Matx33f& cam1_R_p_c) 
{
	// 找到图像时间戳附近的IMU数据
	auto begin_iter = imu_msg_buffer.begin();
	// auto end_iter = imu_msg_buffer.end();

	// if((end_iter-begin_iter) > 8) {
	// 	begin_iter = end_iter - 7;
	// }

	const double cam0_prev_ts = RCL_NS_TO_US(rclcpp::Time(cam0_prev_img_ptr->header.stamp).nanoseconds())*1e-6;
	const double cam0_curr_ts = RCL_NS_TO_US(rclcpp::Time(cam0_curr_img_ptr->header.stamp).nanoseconds())*1e-6;
	while (begin_iter != imu_msg_buffer.end()) {
		double imu_ts = begin_iter->time_us*1e-6;
		if ((imu_ts - cam0_prev_ts) < -0.01) {
			++begin_iter;
		} else {
			break;
		}
	}
	
	auto end_iter = begin_iter;
	while (end_iter != imu_msg_buffer.end()) {
		double imu_ts = end_iter->time_us*1e-6;
		if ((imu_ts - cam0_curr_ts) < 0.01) {
			++end_iter;
		} else {
			break;
		}
	}

	// Compute the mean angular velocity in the IMU frame.
	Vec3f mean_ang_vel(0.0, 0.0, 0.0);
	for (auto iter = begin_iter; iter < end_iter; ++iter) {
		mean_ang_vel += (Vec3f(iter->delAng(0), iter->delAng(1), iter->delAng(2)) - gyro_bias);
	}

	if (end_iter-begin_iter > 0) {
		mean_ang_vel *= 1.0 / (end_iter-begin_iter);
	} else {
		perf_count(_perf_imu_error, timestamp());
	}

	// Delete the useless and used imu messages.
	imu_msg_buffer.erase(imu_msg_buffer.begin(), end_iter);

	// Transform the mean angular velocity from the IMU
	// frame to the cam0 and cam1 frames.
	Vec3f cam0_mean_ang_vel = R_cam0_imu.t() * mean_ang_vel;
	Vec3f cam1_mean_ang_vel = R_cam1_imu.t() * mean_ang_vel;

	const double cam_curr_ts = RCL_NS_TO_US(rclcpp::Time(cam0_curr_img_ptr->header.stamp).nanoseconds())*1e-6;
	const double cam_prev_ts = RCL_NS_TO_US(rclcpp::Time(cam0_prev_img_ptr->header.stamp).nanoseconds())*1e-6;
	// Compute the relative rotation.
	dtime = (cam_curr_ts - cam_prev_ts);
	if(dtime > 0.05) {
		perf_count(_perf_image_drop, timestamp());
	}

	Rodrigues(cam0_mean_ang_vel*dtime, cam0_R_p_c);
	Rodrigues(cam1_mean_ang_vel*dtime, cam1_R_p_c);
	cam0_R_p_c = cam0_R_p_c.t();
	cam1_R_p_c = cam1_R_p_c.t();
}


void ImageProcessor::rescalePoints(vector<Point2f>& pts1, 
	vector<Point2f>& pts2,
    float& scaling_factor) 
{

	scaling_factor = 0.0f;

	for (uint32_t i = 0; i < pts1.size(); ++i) {
		scaling_factor += sqrt(pts1[i].dot(pts1[i]));
		scaling_factor += sqrt(pts2[i].dot(pts2[i]));
	}

	scaling_factor = (pts1.size()+pts2.size()) / scaling_factor * sqrt(2.0f);

	for (uint32_t i = 0; i < pts1.size(); ++i) {
		pts1[i] *= scaling_factor;
		pts2[i] *= scaling_factor;
	}
}

void ImageProcessor::twoPointRansac(const vector<Point2f>& pts1, const vector<Point2f>& pts2,
    const cv::Matx33f& R_p_c, const cv::Vec4d& intrinsics,
    const std::string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    const double& inlier_error,
    const double& success_probability,
    vector<int>& inlier_markers) 
{

	// Check the size of input point size.
	if (pts1.size() != pts2.size()) {
		printf("Sets of different size (%lu and %lu) are used...\n", pts1.size(), pts2.size());
	}

	double norm_pixel_unit = 2.0 / (intrinsics[0]+intrinsics[1]);
	int iter_num = static_cast<int>(ceil(log(1-success_probability) / log(1-0.7*0.7)));

	// Initially, mark all points as inliers.
	inlier_markers.clear();
	inlier_markers.resize(pts1.size(), 1);

	// Undistort all the points.
	vector<Point2f> pts1_undistorted(pts1.size());
	vector<Point2f> pts2_undistorted(pts2.size());
	undistortPoints(
		pts1, intrinsics, distortion_model,
		distortion_coeffs, pts1_undistorted);
	undistortPoints(
		pts2, intrinsics, distortion_model,
		distortion_coeffs, pts2_undistorted);

	// Compenstate the points in the previous image with
	// the relative rotation.
	for (auto& pt : pts1_undistorted) {
		Vec3f pt_h(pt.x, pt.y, 1.0f);
		//Vec3f pt_hc = dR * pt_h;
		Vec3f pt_hc = R_p_c * pt_h;
		pt.x = pt_hc[0];
		pt.y = pt_hc[1];
	}

	// Normalize the points to gain numerical stability.
	float scaling_factor = 0.0f;
	rescalePoints(pts1_undistorted, pts2_undistorted, scaling_factor);
	norm_pixel_unit *= scaling_factor;

	// Compute the difference between previous and current points,
	// which will be used frequently later.
	vector<Point2d> pts_diff(pts1_undistorted.size());
	for (uint32_t i = 0; i < pts1_undistorted.size(); ++i) {
		pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];
	}

	// Mark the point pairs with large difference directly.
	// BTW, the mean distance of the rest of the point pairs
	// are computed.
	double mean_pt_distance = 0.0;
	int raw_inlier_cntr = 0;
	for (uint32_t i = 0; i < pts_diff.size(); ++i) {
		double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
		// 25 pixel distance is a pretty large tolerance for normal motion.
		// However, to be used with aggressive motion, this tolerance should
		// be increased significantly to match the usage.
		if (distance > 50.0*norm_pixel_unit) {
			inlier_markers[i] = 0;
		} else {
			mean_pt_distance += distance;
			++raw_inlier_cntr;
		}
	}
	mean_pt_distance /= raw_inlier_cntr;

	// If the current number of inliers is less than 3, just mark
	// all input as outliers. This case can happen with fast
	// rotation where very few features are tracked.
	if (raw_inlier_cntr < 3) {
		for (auto& marker : inlier_markers) {
			marker = 0;
		}
		return;
	}

	// Before doing 2-point RANSAC, we have to check if the motion
	// is degenerated, meaning that there is no translation between
	// the frames, in which case, the model of the RANSAC does not
	// work. If so, the distance between the matched points will
	// be almost 0.
	//if (mean_pt_distance < inlier_error*norm_pixel_unit) {
	if (mean_pt_distance < norm_pixel_unit) {
		//ROS_WARN_THROTTLE(1.0, "Degenerated motion...");
		for (uint32_t i = 0; i < pts_diff.size(); ++i) {
			if (inlier_markers[i] == 0) {
				continue;
			}
			if (sqrt(pts_diff[i].dot(pts_diff[i])) > inlier_error*norm_pixel_unit) {
				inlier_markers[i] = 0;
			}
		}
		return;
	}

	// In the case of general motion, the RANSAC model can be applied.
	// The three column corresponds to tx, ty, and tz respectively.
	MatrixXd coeff_t(pts_diff.size(), 3);
	for (uint32_t i = 0; i < pts_diff.size(); ++i) {
		coeff_t(i, 0) = pts_diff[i].y;
		coeff_t(i, 1) = -pts_diff[i].x;
		coeff_t(i, 2) = pts1_undistorted[i].x*pts2_undistorted[i].y - pts1_undistorted[i].y*pts2_undistorted[i].x;
	}

	vector<int> raw_inlier_idx;
	for (uint32_t i = 0; i < inlier_markers.size(); ++i) {
		if (inlier_markers[i] != 0) {
			raw_inlier_idx.push_back(i);
		}
	}

	vector<int> best_inlier_set;
	double best_error = 1e10;
	random_numbers::RandomNumberGenerator random_gen;

	for (int iter_idx = 0; iter_idx < iter_num; ++iter_idx) {
		// Randomly select two point pairs.
		// Although this is a weird way of selecting two pairs, but it
		// is able to efficiently avoid selecting repetitive pairs.
		int select_idx1 = random_gen.uniformInteger(
			0, raw_inlier_idx.size()-1);
		int select_idx_diff = random_gen.uniformInteger(
			1, raw_inlier_idx.size()-1);
		int select_idx2 = ((select_idx1 + select_idx_diff) < (int)raw_inlier_idx.size()) ? select_idx1+select_idx_diff : select_idx1+select_idx_diff-raw_inlier_idx.size();

		int pair_idx1 = raw_inlier_idx[select_idx1];
		int pair_idx2 = raw_inlier_idx[select_idx2];

		// Construct the model;
		Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
		Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
		Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
		vector<double> coeff_l1_norm(3);
		coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
		coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
		coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
		int base_indicator = min_element(coeff_l1_norm.begin(),coeff_l1_norm.end())-coeff_l1_norm.begin();

		Vector3d model(0.0, 0.0, 0.0);
		if (base_indicator == 0) {
			Matrix2d A;
			A << coeff_ty, coeff_tz;
			Vector2d solution = A.inverse() * (-coeff_tx);
			model(0) = 1.0;
			model(1) = solution(0);
			model(2) = solution(1);
		} else if (base_indicator ==1) {
			Matrix2d A;
			A << coeff_tx, coeff_tz;
			Vector2d solution = A.inverse() * (-coeff_ty);
			model(0) = solution(0);
			model(1) = 1.0;
			model(2) = solution(1);
		} else {
			Matrix2d A;
			A << coeff_tx, coeff_ty;
			Vector2d solution = A.inverse() * (-coeff_tz);
			model(0) = solution(0);
			model(1) = solution(1);
			model(2) = 1.0;
		}

		// Find all the inliers among point pairs.
		VectorXd error = coeff_t * model;

		vector<int> inlier_set;
		for (int i = 0; i < error.rows(); ++i) {
			if (inlier_markers[i] == 0) {
				continue;
			}
			if (std::abs(error(i)) < inlier_error*norm_pixel_unit) {
				inlier_set.push_back(i);
			}
		}

		// If the number of inliers is small, the current
		// model is probably wrong.
		if (inlier_set.size() < 0.2*pts1_undistorted.size()) {
			continue;
		}

		// Refit the model using all of the possible inliers.
		VectorXd coeff_tx_better(inlier_set.size());
		VectorXd coeff_ty_better(inlier_set.size());
		VectorXd coeff_tz_better(inlier_set.size());
		for (uint32_t i = 0; i < inlier_set.size(); ++i) {
			coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
			coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
			coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
		}

		Vector3d model_better(0.0, 0.0, 0.0);
		if (base_indicator == 0) {
			MatrixXd A(inlier_set.size(), 2);
			A << coeff_ty_better, coeff_tz_better;
			Vector2d solution =
				(A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
			model_better(0) = 1.0;
			model_better(1) = solution(0);
			model_better(2) = solution(1);
		} else if (base_indicator ==1) {
			MatrixXd A(inlier_set.size(), 2);
			A << coeff_tx_better, coeff_tz_better;
			Vector2d solution =
				(A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
			model_better(0) = solution(0);
			model_better(1) = 1.0;
			model_better(2) = solution(1);
		} else {
			MatrixXd A(inlier_set.size(), 2);
			A << coeff_tx_better, coeff_ty_better;
			Vector2d solution =
				(A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
			model_better(0) = solution(0);
			model_better(1) = solution(1);
			model_better(2) = 1.0;
		}

		// Compute the error and upate the best model if possible.
		VectorXd new_error = coeff_t * model_better;

		double this_error = 0.0;
		for (const auto& inlier_idx : inlier_set) {
			this_error += std::abs(new_error(inlier_idx));
		}
		this_error /= inlier_set.size();

		if (inlier_set.size() > best_inlier_set.size()) {
			best_error = this_error;
			best_inlier_set = inlier_set;
		}
	}

	// Fill in the markers.
	inlier_markers.clear();
	inlier_markers.resize(pts1.size(), 0);
	for (const auto& inlier_idx : best_inlier_set) {
		inlier_markers[inlier_idx] = 1;
	}

	//printf("inlier ratio: %lu/%lu\n",
	//    best_inlier_set.size(), inlier_markers.size());
}

// prev frames cam0 ----------> cam1
//              |光流             |
//              |ransac          |ransac
//              |   stereo match |
// curr frames cam0 ----------> cam1
/*step1：用pre和cur之间IMU的平均角速度粗略计算pre和cur image之间的旋转，
 *       然后粗略计算pre中的特征点在cur image中的位置，为光流提供初值
 *step2：cam0 pre和cur帧间光流跟踪
 *step3: curr cam0和cam1双目匹配（光流跟踪，包括外点筛选）
 *step4：剔除外点，
 *       1.cur左右图光流跟踪上，且cam0前后图光流跟踪上的特征点才得以保留
 *       2.cam0的prev和curr匹配点进行RANSAC剔除外点
 *       3.cam1的prev和curr匹配点进行RANSAC剔除外点（RANSAC模型为归一化相机坐标的对极约束）
 */
void ImageProcessor::trackFeatures() 
{
	// 计算网格尺寸
	static int grid_height =
		cam0_curr_img_ptr->image.rows / processor_config.grid_row;
	static int grid_width =
		cam0_curr_img_ptr->image.cols / processor_config.grid_col;

	/*step1：用IMU的平均角速度粗略计算pre和cur image之间的旋转，
	 * 然后粗略计算pre中的特征点在cur image中的位置，为光流提供初值
	 */
	Matx33f cam0_R_p_c;
	Matx33f cam1_R_p_c;
	integrateImuData(cam0_R_p_c, cam1_R_p_c);

	// 管理上一帧的特征点            
	vector<FeatureIDType> prev_ids(0);
	vector<int> prev_lifetime(0);
	vector<Point2f> prev_cam0_points(0);
	vector<Point2f> prev_cam1_points(0);

	for (const auto& item : *prev_features_ptr) {
		for (const auto& prev_feature : item.second) {
			prev_ids.push_back(prev_feature.id);
			prev_lifetime.push_back(prev_feature.lifetime);
			prev_cam0_points.push_back(prev_feature.cam0_point);
			prev_cam1_points.push_back(prev_feature.cam1_point);
		}
	}

	// 统计上一帧特征点对数
	before_tracking = prev_cam0_points.size();

	if (prev_ids.size() == 0) return;

	// LK-光流跟踪特征点.
	vector<Point2f> curr_cam0_points(0);
	vector<unsigned char> track_inliers(0);

	// 结合IMU旋转估根据上一帧特征点计当前帧的位置
	predictFeatureTracking(prev_cam0_points, cam0_R_p_c, cam0_intrinsics, curr_cam0_points);
	if(!processor_config.use_cuda) {
		calcOpticalFlowPyrLK(
			prev_cam0_pyramid_, curr_cam0_pyramid_,
			prev_cam0_points, curr_cam0_points,
			track_inliers, noArray(),
			Size(processor_config.patch_size, processor_config.patch_size),
			processor_config.pyramid_levels,
			TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
			processor_config.max_iteration,
			processor_config.track_precision),
			cv::OPTFLOW_USE_INITIAL_FLOW);
	} 
#if defined(USE_CUDA)
	else {
		const Mat& prev_cam0_img = cam0_prev_img_ptr->image;
		const Mat& curr_cam0_img = cam0_curr_img_ptr->image;
		
		cv::cuda::GpuMat prev_gpu_img0(prev_cam0_img);
		cv::cuda::GpuMat curr_gpu_img0(curr_cam0_img);
		cv::cuda::GpuMat prev_gpu_pts0(prev_cam0_points);
		cv::cuda::GpuMat curr_gpu_pts0(curr_cam0_points);
		cv::cuda::GpuMat gpu_status;

		cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cv::cuda::SparsePyrLKOpticalFlow::create(
		cv::Size(processor_config.patch_size, processor_config.patch_size), 
		processor_config.pyramid_levels, 
		processor_config.max_iteration);
		d_pyrLK_sparse->calc(prev_gpu_img0, curr_gpu_img0, prev_gpu_pts0, curr_gpu_pts0, gpu_status);

		vector<cv::Point2f> tmp0_prve_pts(prev_gpu_pts0.cols);
		prev_gpu_pts0.download(tmp0_prve_pts);
		prev_cam0_points = tmp0_prve_pts;

		vector<cv::Point2f> tmp1_cur_pts(curr_gpu_pts0.cols);
		curr_gpu_pts0.download(tmp1_cur_pts);
		curr_cam0_points = tmp1_cur_pts;

		vector<uchar> tmp_status(gpu_status.cols);
		gpu_status.download(tmp_status);
		track_inliers = tmp_status;
	}
#endif

	// 剔除跟踪失效的特征点
	for (uint32_t i = 0; i < curr_cam0_points.size(); ++i) {
		if (track_inliers[i] == 0) {
			continue;
		}
		if (curr_cam0_points[i].y < 0 ||
			curr_cam0_points[i].y > cam0_curr_img_ptr->image.rows-1 ||
			curr_cam0_points[i].x < 0 ||
			curr_cam0_points[i].x > cam0_curr_img_ptr->image.cols-1) {
			track_inliers[i] = 0;
		}
	}

	// Collect the tracked points.
	vector<FeatureIDType> prev_tracked_ids(0);
	vector<int> prev_tracked_lifetime(0);
	vector<Point2f> prev_tracked_cam0_points(0);
	vector<Point2f> prev_tracked_cam1_points(0);
	vector<Point2f> curr_tracked_cam0_points(0);

	removeUnmarkedElements(
		prev_ids, track_inliers, prev_tracked_ids);
	removeUnmarkedElements(
		prev_lifetime, track_inliers, prev_tracked_lifetime);
	removeUnmarkedElements(
		prev_cam0_points, track_inliers, prev_tracked_cam0_points);
	removeUnmarkedElements(
		prev_cam1_points, track_inliers, prev_tracked_cam1_points);
	removeUnmarkedElements(
		curr_cam0_points, track_inliers, curr_tracked_cam0_points);

	// Number of features left after tracking.
	after_tracking = curr_tracked_cam0_points.size();

	// Outlier removal involves three steps, which forms a close
	// loop between the previous and current frames of cam0 (left)
	// and cam1 (right). Assuming the stereo matching between the
	// previous cam0 and cam1 images are correct, the three steps are:
	//
	// prev frames cam0 ----------> cam1
	//              |                |
	//              |ransac          |ransac
	//              |   stereo match |
	// curr frames cam0 ----------> cam1
	//
	// 1) Stereo matching between current images of cam0 and cam1.
	// 2) RANSAC between previous and current images of cam0.
	// 3) RANSAC between previous and current images of cam1.
	//
	// For Step 3, tracking between the images is no longer needed.
	// The stereo matching results are directly used in the RANSAC.

	// Step 1: stereo matching.
	vector<Point2f> curr_cam1_points(0);
	vector<unsigned char> match_inliers(0);
	stereoMatch(curr_tracked_cam0_points, curr_cam1_points, match_inliers);

	vector<FeatureIDType> prev_matched_ids(0);
	vector<int> prev_matched_lifetime(0);
	vector<Point2f> prev_matched_cam0_points(0);
	vector<Point2f> prev_matched_cam1_points(0);
	vector<Point2f> curr_matched_cam0_points(0);
	vector<Point2f> curr_matched_cam1_points(0);

	removeUnmarkedElements(
		prev_tracked_ids, match_inliers, prev_matched_ids);
	removeUnmarkedElements(
		prev_tracked_lifetime, match_inliers, prev_matched_lifetime);
	removeUnmarkedElements(
		prev_tracked_cam0_points, match_inliers, prev_matched_cam0_points);
	removeUnmarkedElements(
		prev_tracked_cam1_points, match_inliers, prev_matched_cam1_points);
	removeUnmarkedElements(
		curr_tracked_cam0_points, match_inliers, curr_matched_cam0_points);
	removeUnmarkedElements(
		curr_cam1_points, match_inliers, curr_matched_cam1_points);

	// Number of features left after stereo matching.
	after_matching = curr_matched_cam0_points.size();

	// Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
	vector<int> cam0_ransac_inliers(0);
	twoPointRansac(prev_matched_cam0_points, curr_matched_cam0_points,
		cam0_R_p_c, cam0_intrinsics, cam0_distortion_model,
		cam0_distortion_coeffs, processor_config.ransac_threshold,
		0.99, cam0_ransac_inliers);

	vector<int> cam1_ransac_inliers(0);
	twoPointRansac(prev_matched_cam1_points, curr_matched_cam1_points,
		cam1_R_p_c, cam1_intrinsics, cam1_distortion_model,
		cam1_distortion_coeffs, processor_config.ransac_threshold,
		0.99, cam1_ransac_inliers);

	// Number of features after ransac.
	after_ransac = 0;

	for (uint32_t i = 0; i < cam0_ransac_inliers.size(); ++i) {
		if (cam0_ransac_inliers[i] == 0 ||
			cam1_ransac_inliers[i] == 0) {
				continue;
		}
		int row = static_cast<int>(
			curr_matched_cam0_points[i].y / grid_height);
		int col = static_cast<int>(
			curr_matched_cam0_points[i].x / grid_width);
		int code = row*processor_config.grid_col + col;
		(*curr_features_ptr)[code].push_back(FeatureMetaData());

		FeatureMetaData& grid_new_feature = (*curr_features_ptr)[code].back();
		grid_new_feature.id = prev_matched_ids[i];
		grid_new_feature.lifetime = ++prev_matched_lifetime[i];
		grid_new_feature.cam0_point = curr_matched_cam0_points[i];
		grid_new_feature.cam1_point = curr_matched_cam1_points[i];

		++after_ransac;
	}

	// Compute the tracking rate.
	prev_feature_num = 0;
	for (const auto& item : *prev_features_ptr) {
		prev_feature_num += item.second.size();
	}

	curr_feature_num = 0;
	for (const auto& item : *curr_features_ptr) {
		curr_feature_num += item.second.size();
	}
}

void ImageProcessor::printf_track_info()
{
	printf("cam_dt:%.6f imu:%d candidates:%d track:%d match:%d ransac:%d/%d=%f\n",
		dtime, (uint32_t)imu_msg_buffer.size(), before_tracking, after_tracking, after_matching,
		curr_feature_num, prev_feature_num,
		static_cast<double>(curr_feature_num)/
		(static_cast<double>(prev_feature_num)+1e-5));
}

void ImageProcessor::addNewFeatures() 
{
  	const Mat& curr_img = cam0_curr_img_ptr->image;

	// Size of each grid.
	static int grid_height =
		cam0_curr_img_ptr->image.rows / processor_config.grid_row;
	static int grid_width =
		cam0_curr_img_ptr->image.cols / processor_config.grid_col;

	// 创建mask避免重复检测存在的特征点.
	Mat mask(curr_img.rows, curr_img.cols, CV_8U, Scalar(1));

	for (const auto& features : *curr_features_ptr) {
		for (const auto& feature : features.second) {
			const int y = static_cast<int>(feature.cam0_point.y);
			const int x = static_cast<int>(feature.cam0_point.x);

			int up_lim = y-2, bottom_lim = y+3,
				left_lim = x-2, right_lim = x+3;
			if (up_lim < 0) {
				up_lim = 0;
			}
			if (bottom_lim > curr_img.rows) {
				bottom_lim = curr_img.rows;
			}
			if (left_lim < 0) {
				left_lim = 0;
			}
			if (right_lim > curr_img.cols) {
				right_lim = curr_img.cols;
			}

			Range row_range(up_lim, bottom_lim);
			Range col_range(left_lim, right_lim);
			mask(row_range, col_range) = 0;
		}
	}

	// Detect new features.
	vector<KeyPoint> new_features(0);
	if(!processor_config.use_cuda) {
		detector_ptr->detect(curr_img, new_features, mask);
	} 
#if defined(USE_CUDA)
	else {
		cv::cuda::GpuMat gFrame(curr_img);
		cv::cuda::GpuMat gpu_mask(mask);
		cuda_detector_ptr->detect(gFrame, new_features, gpu_mask);
	}
#endif

	// Collect the new detected features based on the grid.
	// Select the ones with top response within each grid afterwards.
	vector<vector<KeyPoint>> new_feature_sieve(
		processor_config.grid_row*processor_config.grid_col);
	for (const auto& feature : new_features) {
		int row = static_cast<int>(feature.pt.y / grid_height);
		int col = static_cast<int>(feature.pt.x / grid_width);
		new_feature_sieve[row*processor_config.grid_col+col].push_back(feature);
	}

	new_features.clear();
	// 去除最大数量以外的特征点
	for (auto& item : new_feature_sieve) {
		if (item.size() > (uint32_t)processor_config.grid_max_feature_num) {
			std::sort(item.begin(), item.end(),
				&ImageProcessor::keyPointCompareByResponse);
			item.erase(item.begin()+processor_config.grid_max_feature_num, item.end());
		}
		new_features.insert(new_features.end(), item.begin(), item.end());
	}

	int detected_new_features = new_features.size();

	// Find the stereo matched points for the newly
	// detected features.
	vector<cv::Point2f> cam0_points(new_features.size());
	for (uint32_t i = 0; i < new_features.size(); ++i) {
		cam0_points[i] = new_features[i].pt;
	}

	vector<cv::Point2f> cam1_points(0);
	vector<unsigned char> inlier_markers(0);
	stereoMatch(cam0_points, cam1_points, inlier_markers);

	vector<cv::Point2f> cam0_inliers(0);
	vector<cv::Point2f> cam1_inliers(0);
	vector<float> response_inliers(0);
	for (uint32_t i = 0; i < inlier_markers.size(); ++i) {
		if (inlier_markers[i] == 0) {
			continue;
		}
		cam0_inliers.push_back(cam0_points[i]);
		cam1_inliers.push_back(cam1_points[i]);
		response_inliers.push_back(new_features[i].response);
	}

  	int matched_new_features = cam0_inliers.size();

	// 检测画面是否同步
	if (matched_new_features < 5 && static_cast<double>(matched_new_features)/static_cast<double>(detected_new_features) < 0.1) {
		RCLCPP_INFO_THROTTLE(_node->get_logger(), *(_node->get_clock()), 1000.0, "Images at [%f] seems unsynced...",(float)cam0_curr_img_ptr->header.stamp.sec);
	}

	// Group the features into grids
	GridFeatures grid_new_features;
	for (int code = 0; code < processor_config.grid_row*processor_config.grid_col; ++code) {
		grid_new_features[code] = vector<FeatureMetaData>(0);
	}

	for (uint32_t i = 0; i < cam0_inliers.size(); ++i) {
		const cv::Point2f& cam0_point = cam0_inliers[i];
		const cv::Point2f& cam1_point = cam1_inliers[i];
		const float& response = response_inliers[i];

		int row = static_cast<int>(cam0_point.y / grid_height);
		int col = static_cast<int>(cam0_point.x / grid_width);
		int code = row*processor_config.grid_col + col;

		FeatureMetaData new_feature;
		new_feature.response = response;
		new_feature.cam0_point = cam0_point;
		new_feature.cam1_point = cam1_point;
		grid_new_features[code].push_back(new_feature);
	}

	// 排序.
	for (auto& item : grid_new_features) {
		std::sort(item.second.begin(), item.second.end(), &ImageProcessor::featureCompareByResponse);
	}

	int new_added_feature_num = 0;
	// Collect new features within each grid with high response.
	for (int code = 0; code < processor_config.grid_row*processor_config.grid_col; ++code) {
		vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
		vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

		if (features_this_grid.size() >= (uint32_t)processor_config.grid_min_feature_num) {
			continue;
		}

		uint32_t vacancy_num = processor_config.grid_min_feature_num - features_this_grid.size();
		for (uint32_t k = 0;k < vacancy_num && k < new_features_this_grid.size(); ++k) {
			features_this_grid.push_back(new_features_this_grid[k]);
			features_this_grid.back().id = next_feature_id++;
			features_this_grid.back().lifetime = 1;

			++new_added_feature_num;
		}
	}
}

void ImageProcessor::pruneGridFeatures() 
{
	for (auto& item : *curr_features_ptr) {
		auto& grid_features = item.second;
		// Continue if the number of features in this grid does
		// not exceed the upper bound.
		if (grid_features.size() <= (uint32_t)processor_config.grid_max_feature_num) {
			continue;
		}
		std::sort(grid_features.begin(), grid_features.end(), &ImageProcessor::featureCompareByLifetime);
		grid_features.erase(grid_features.begin()+ processor_config.grid_max_feature_num, grid_features.end());
	}
}

void ImageProcessor::drawFeaturesStereo() 
{
	// 定义不同特征点颜色
	Scalar tracked(0, 255, 0);
	Scalar new_feature(0, 255, 255);

	// static int grid_height =
	// 	cam0_curr_img_ptr->image.rows / processor_config.grid_row;
	// static int grid_width =
	// 	cam0_curr_img_ptr->image.cols / processor_config.grid_col;

	// Create an output image.
	int img_height = cam0_curr_img_ptr->image.rows;
	int img_width = cam0_curr_img_ptr->image.cols;
	Mat out_img(img_height, img_width*2, CV_8UC3);
	cvtColor(cam0_curr_img_ptr->image,
			out_img.colRange(0, img_width), CV_GRAY2RGB);
	cvtColor(cam1_curr_img_ptr->image,
			out_img.colRange(img_width, img_width*2), CV_GRAY2RGB);

	// 绘制网格
	// for (int i = 1; i < processor_config.grid_row; ++i) {
	// 	Point pt1(0, i*grid_height);
	// 	Point pt2(img_width*2, i*grid_height);
	// 	line(out_img, pt1, pt2, Scalar(255, 0, 0));
	// }
	// for (int i = 1; i < processor_config.grid_col; ++i) {
	// 	Point pt1(i*grid_width, 0);
	// 	Point pt2(i*grid_width, img_height);
	// 	line(out_img, pt1, pt2, Scalar(255, 0, 0));
	// }
	// for (int i = 1; i < processor_config.grid_col; ++i) {
	// 	Point pt1(i*grid_width+img_width, 0);
	// 	Point pt2(i*grid_width+img_width, img_height);
	// 	line(out_img, pt1, pt2, Scalar(255, 0, 0));
	// }

	// 收集上一帧的特征点ID
	vector<FeatureIDType> prev_ids(0);
	for (const auto& grid_features : *prev_features_ptr) {
		for (const auto& feature : grid_features.second) {
			prev_ids.push_back(feature.id);
		}
	}
	
	// 收集上一帧的特征点
	map<FeatureIDType, Point2f> prev_cam0_points;
	map<FeatureIDType, Point2f> prev_cam1_points;
	for (const auto& grid_features : *prev_features_ptr) {
		for (const auto& feature : grid_features.second) {
			prev_cam0_points[feature.id] = feature.cam0_point;
			prev_cam1_points[feature.id] = feature.cam1_point;
		}
	}

	// 收集当前帧的特征点
	map<FeatureIDType, Point2f> curr_cam0_points;
	map<FeatureIDType, Point2f> curr_cam1_points;
	for (const auto& grid_features : *curr_features_ptr) {
		for (const auto& feature : grid_features.second) {
			curr_cam0_points[feature.id] = feature.cam0_point;
			curr_cam1_points[feature.id] = feature.cam1_point;
		}
	}

	// 绘制特征点
	for (const auto& id : prev_ids) {
		if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
			curr_cam0_points.find(id) != curr_cam0_points.end()) {
			cv::Point2f prev_pt0 = prev_cam0_points[id];
			cv::Point2f prev_pt1 = prev_cam1_points[id] + Point2f(img_width, 0.0);
			cv::Point2f curr_pt0 = curr_cam0_points[id];
			cv::Point2f curr_pt1 = curr_cam1_points[id] + Point2f(img_width, 0.0);

			circle(out_img, curr_pt0, 3, tracked, -1);
			circle(out_img, curr_pt1, 3, tracked, -1);
			line(out_img, prev_pt0, curr_pt0, tracked, 1);
			line(out_img, prev_pt1, curr_pt1, tracked, 1);

			prev_cam0_points.erase(id);
			prev_cam1_points.erase(id);
			curr_cam0_points.erase(id);
			curr_cam1_points.erase(id);
		}
	}

	// 绘制新特征点
	for (const auto& new_cam0_point : curr_cam0_points) {
		cv::Point2f pt0 = new_cam0_point.second;
		cv::Point2f pt1 = curr_cam1_points[new_cam0_point.first] +
			Point2f(img_width, 0.0);

		circle(out_img, pt0, 3, new_feature, -1);
		circle(out_img, pt1, 3, new_feature, -1);
	}

	cv_bridge::CvImage debug_image(cam0_curr_img_ptr->header, "bgr8", out_img);

	sensor_msgs::msg::Image::SharedPtr debug_image_msg = debug_image.toImageMsg();
	debug_image_msg->header.frame_id = "world";
	debug_stereo_pub.publish(debug_image_msg);
	
	// imshow("Feature", out_img);
	// waitKey(5);
}

void ImageProcessor::calculate_features() 
{
	// Publish features.
	feature_msg_ptr.timestamp = RCL_NS_TO_US(rclcpp::Time(cam0_curr_img_ptr->header.stamp).nanoseconds());

	vector<FeatureIDType> curr_ids(0);
	vector<Point2f> curr_cam0_points(0);
	vector<Point2f> curr_cam1_points(0);

	for (const auto& grid_features : (*curr_features_ptr)) {
		for (const auto& feature : grid_features.second) {
			curr_ids.push_back(feature.id);
			curr_cam0_points.push_back(feature.cam0_point);
			curr_cam1_points.push_back(feature.cam1_point);
		}
	}

	vector<Point2f> curr_cam0_points_undistorted(0);
	vector<Point2f> curr_cam1_points_undistorted(0);

	undistortPoints(
		curr_cam0_points, cam0_intrinsics, cam0_distortion_model,
		cam0_distortion_coeffs, curr_cam0_points_undistorted);
	undistortPoints(
		curr_cam1_points, cam1_intrinsics, cam1_distortion_model,
		cam1_distortion_coeffs, curr_cam1_points_undistorted);

	feature_msg_ptr.features.clear();
	for (uint32_t i = 0; i < curr_ids.size(); ++i) {
		feature_msg_ptr.features.push_back(sprain_msgs::msg::FeatureMeasurement());
		feature_msg_ptr.features[i].id = curr_ids[i];
		feature_msg_ptr.features[i].u0 = curr_cam0_points_undistorted[i].x;
		feature_msg_ptr.features[i].v0 = curr_cam0_points_undistorted[i].y;
		feature_msg_ptr.features[i].u1 = curr_cam1_points_undistorted[i].x;
		feature_msg_ptr.features[i].v1 = curr_cam1_points_undistorted[i].y;
	}
}

template<class T>
void ImageProcessor::setNgetNodeParameter(T& param, const std::string& param_name, const T& default_value, const rcl_interfaces::msg::ParameterDescriptor &parameter_descriptor)
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

void ImageProcessor::cam_ext_param(cv::Matx33d R_imu_cam0, cv::Vec3d t_imu_cam0)
{
	R_cam0_imu = R_imu_cam0.t();
	t_cam0_imu = -R_imu_cam0.t() * t_imu_cam0;
}

void ImageProcessor::getParameter()
{
	setNgetNodeParameter(processor_config.use_cuda, "cuda", false);
	setNgetNodeParameter(processor_config.grid_row, "grid_row", (int64_t)4);
	setNgetNodeParameter(processor_config.grid_col, "grid_col", (int64_t)5);
	setNgetNodeParameter(processor_config.grid_min_feature_num, "grid_min_feature_num", (int64_t)2);
	setNgetNodeParameter(processor_config.grid_max_feature_num, "grid_max_feature_num", (int64_t)3);
	setNgetNodeParameter(processor_config.pyramid_levels, "pyramid_levels", (int64_t)3);
	setNgetNodeParameter(processor_config.patch_size, "patch_size", (int64_t)15);
	setNgetNodeParameter(processor_config.fast_threshold, "fast_threshold", (int64_t)10);
	setNgetNodeParameter(processor_config.max_iteration, "max_iteration", (int64_t)30);
	setNgetNodeParameter(processor_config.track_precision, "track_precision", 0.01);
	setNgetNodeParameter(processor_config.ransac_threshold, "ransac_threshold", 3.0);
	setNgetNodeParameter(processor_config.stereo_threshold, "stereo_threshold", 5.0);

	setNgetNodeParameter(cam0_distortion_model, "cam0_distortion_model", string("radtan"));
	setNgetNodeParameter(cam1_distortion_model, "cam1_distortion_model", string("radtan"));

	vector<int64_t> cam0_resolution_temp(2);
	setNgetNodeParameter(cam0_resolution_temp, "cam0_resolution", vector<int64_t>(2));
	cam0_resolution[0] = cam0_resolution_temp[0];
  	cam0_resolution[1] = cam0_resolution_temp[1];

	vector<int64_t> cam1_resolution_temp(2);
	setNgetNodeParameter(cam1_resolution_temp, "cam1_resolution", vector<int64_t>(2));
	cam1_resolution[0] = cam1_resolution_temp[0];
  	cam1_resolution[1] = cam1_resolution_temp[1];

	vector<double> cam0_intrinsics_temp(4);
	setNgetNodeParameter(cam0_intrinsics_temp, "cam0_intrinsics", vector<double>(4));
	cam0_intrinsics[0] = cam0_intrinsics_temp[0];
	cam0_intrinsics[1] = cam0_intrinsics_temp[1];
	cam0_intrinsics[2] = cam0_intrinsics_temp[2];
	cam0_intrinsics[3] = cam0_intrinsics_temp[3];

	vector<double> cam1_intrinsics_temp(4);
	setNgetNodeParameter(cam1_intrinsics_temp, "cam1_intrinsics", vector<double>(4));
	cam1_intrinsics[0] = cam1_intrinsics_temp[0];
	cam1_intrinsics[1] = cam1_intrinsics_temp[1];
	cam1_intrinsics[2] = cam1_intrinsics_temp[2];
	cam1_intrinsics[3] = cam1_intrinsics_temp[3];

	vector<double> cam0_distortion_coeffs_temp(4);
	setNgetNodeParameter(cam0_distortion_coeffs_temp, "cam0_coeffs", vector<double>(4));
	cam0_distortion_coeffs[0] = cam0_distortion_coeffs_temp[0];
	cam0_distortion_coeffs[1] = cam0_distortion_coeffs_temp[1];
	cam0_distortion_coeffs[2] = cam0_distortion_coeffs_temp[2];
	cam0_distortion_coeffs[3] = cam0_distortion_coeffs_temp[3];

	vector<double> cam1_distortion_coeffs_temp(4);
	setNgetNodeParameter(cam1_distortion_coeffs_temp, "cam1_coeffs", vector<double>(4));
	cam1_distortion_coeffs[0] = cam1_distortion_coeffs_temp[0];
	cam1_distortion_coeffs[1] = cam1_distortion_coeffs_temp[1];
	cam1_distortion_coeffs[2] = cam1_distortion_coeffs_temp[2];
	cam1_distortion_coeffs[3] = cam1_distortion_coeffs_temp[3];

	vector<double> t_imu_cam(16);
	setNgetNodeParameter(t_imu_cam, "T_imu_cam", vector<double>(16));
	cv::Mat     T_imu_cam0 = cv::Mat(t_imu_cam).clone().reshape(1, 4);
	cv::Matx33d R_imu_cam0(T_imu_cam0(cv::Rect(0,0,3,3)));
	cv::Vec3d   t_imu_cam0 = T_imu_cam0(cv::Rect(3,0,1,3));
	R_cam0_imu = R_imu_cam0.t();
	t_cam0_imu = -R_imu_cam0.t() * t_imu_cam0;

	vector<double> t_cn_cnm1(16);
	setNgetNodeParameter(t_cn_cnm1, "T_cn_cnm1", vector<double>(16));
	cv::Mat T_cam0_cam1 = cv::Mat(t_cn_cnm1).clone().reshape(1, 4);
	cv::Mat T_imu_cam1 = T_cam0_cam1 * T_imu_cam0;
	cv::Matx33d R_imu_cam1(T_imu_cam1(cv::Rect(0,0,3,3)));
	cv::Vec3d   t_imu_cam1 = T_imu_cam1(cv::Rect(3,0,1,3));
	R_cam1_imu = R_imu_cam1.t();
	t_cam1_imu = -R_imu_cam1.t() * t_imu_cam1;

	// printf parameter info
	printf("=======================Image Processor Parameter==================\n");
	printf("use_cuda :%d\n", processor_config.use_cuda);
	printf("grid_row :%ld\n", processor_config.grid_row);
	printf("grid_col :%ld\n", processor_config.grid_col);
	printf("grid_min_feature_num :%ld\n", processor_config.grid_min_feature_num);
	printf("grid_max_feature_num :%ld\n", processor_config.grid_max_feature_num);
	printf("pyramid_levels :%ld\n", processor_config.pyramid_levels);
	printf("patch_size :%ld\n", processor_config.patch_size);
	printf("fast_threshold :%ld\n", processor_config.fast_threshold);
	printf("max_iteration :%ld\n", processor_config.max_iteration);
	printf("track_precision :%.2f\n", processor_config.track_precision);
	printf("ransac_threshold :%.2f\n", processor_config.ransac_threshold);
	printf("stereo_threshold :%.2f\n", processor_config.stereo_threshold);

	printf("========================Camera Information========================\n");
	printf("cam0:\n");
	printf("cam0_model :%s\n", cam0_distortion_model.c_str());
	printf("cam0_resolution :[%d, %d]\n", cam0_resolution[0], cam0_resolution[1]);
	printf("cam0_intrinsics :[%.4f, %.4f, %.4f, %.4f]\n", cam0_intrinsics[0], cam0_intrinsics[1], cam0_intrinsics[2], cam0_intrinsics[3]);
	printf("cam0_coeffs :[%.4f, %.4f, %.4f, %.4f]\n", cam0_distortion_coeffs[0], cam0_distortion_coeffs[1], cam0_distortion_coeffs[2], cam0_distortion_coeffs[3]);

	printf("cam1:\n");
	printf("cam1_model :%s\n", cam1_distortion_model.c_str());
	printf("cam1_resolution :[%d, %d]\n", cam1_resolution[0], cam1_resolution[1]);
	printf("cam1_intrinsics :[%.4f, %.4f, %.4f, %.4f]\n", cam1_intrinsics[0], cam1_intrinsics[1], cam1_intrinsics[2], cam1_intrinsics[3]);
	printf("cam1_coeffs :[%.4f, %.4f, %.4f, %.4f]\n", cam1_distortion_coeffs[0], cam1_distortion_coeffs[1], cam1_distortion_coeffs[2], cam1_distortion_coeffs[3]);

	printf("T_imu_cam :\n");
	cout << T_imu_cam0 << endl;

	printf("T_cn_cnm1 :\n");
	cout << T_cam0_cam1 << endl;

	printf("T_imu_cam0 :\n[ %.5f, %.5f, %.5f, %.5f \n  %.5f, %.5f, %.5f, %.5f \n  %.5f, %.5f, %.5f, %.5f \n  %.5f, %.5f, %.5f, %.5f ]\n", 
		R_imu_cam0(0,0), R_imu_cam0(0,1), R_imu_cam0(0,2), t_imu_cam0(0),
		R_imu_cam0(1,0), R_imu_cam0(1,1), R_imu_cam0(1,2), t_imu_cam0(1),
		R_imu_cam0(2,0), R_imu_cam0(2,1), R_imu_cam0(2,2), t_imu_cam0(2),
		0.0,0.0, 0.0, 1.0);

	printf("T_imu_cam1 :\n[ %.5f, %.5f, %.5f, %.5f \n  %.5f, %.5f, %.5f, %.5f \n  %.5f, %.5f, %.5f, %.5f \n  %.5f, %.5f, %.5f, %.5f ]\n", 
		R_imu_cam1(0,0), R_imu_cam1(0,1), R_imu_cam1(0,2), t_imu_cam1(0),
		R_imu_cam1(1,0), R_imu_cam1(1,1), R_imu_cam1(1,2), t_imu_cam1(1),
		R_imu_cam1(2,0), R_imu_cam1(2,1), R_imu_cam1(2,2), t_imu_cam1(2),
		0.0,0.0, 0.0, 1.0);
}
