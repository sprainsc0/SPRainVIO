#ifndef __IMAGE_PROCESSOR_H__
#define __IMAGE_PROCESSOR_H__

#include <stdint.h>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

#if defined(USE_CUDA)
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif

#include <common/random_numbers.h>
#include <common/perform.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <sprain_msgs/msg/camera_measurement.hpp>

#include <nav_common.h>

class ImageProcessor
{
public:
	ImageProcessor(std::shared_ptr<rclcpp::Node> node);
    ~ImageProcessor(void);

    struct ProcessorConfig {
        bool use_cuda;
        int64_t grid_row;
        int64_t grid_col;
        int64_t grid_min_feature_num;
        int64_t grid_max_feature_num;

        int64_t pyramid_levels;
        int64_t patch_size;
        int64_t fast_threshold;
        int64_t max_iteration;
        double track_precision;
        double ransac_threshold;
        double stereo_threshold;
    };

    /*
    * @brief FeatureIDType An alias for unsigned long long int.
    */
    typedef unsigned long long int FeatureIDType;

    /*
    * @brief FeatureMetaData Contains necessary information
    *    of a feature for easy access.
    */
    struct FeatureMetaData {
        FeatureIDType id;
        float response;
        int lifetime;
        cv::Point2f cam0_point;
        cv::Point2f cam1_point;
    };

    void init(void);

    uint64_t timestamp(void) const { return RCL_NS_TO_US(_node->now().nanoseconds()); }

    void cam_imu_sync(double td) { cam_imu_sync_dt = td; }

    void cam_ext_param(cv::Matx33d R_imu_cam0, cv::Vec3d t_imu_cam0);

    void feed_imu(const nav::imu_elements data, Eigen::Vector3d bias);

    void feed_stereo( const sensor_msgs::msg::Image::ConstSharedPtr& img0, 
        const sensor_msgs::msg::Image::ConstSharedPtr& img1);

    sprain_msgs::msg::CameraMeasurement get_feature(void) const { return feature_msg_ptr; }

    void printf_track_info(void);

private:
    std::shared_ptr<rclcpp::Node> _node;

    sprain_msgs::msg::CameraMeasurement feature_msg_ptr;

    image_transport::Publisher debug_stereo_pub;

    // Feature detector
    ProcessorConfig processor_config;
    cv::Ptr<cv::Feature2D> detector_ptr;
#if defined(USE_CUDA)
    cv::Ptr<cv::cuda::FastFeatureDetector> cuda_detector_ptr;
#endif
    // IMU data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between IMU and Image messages.
    std::vector<nav::imu_elements> imu_msg_buffer;

    /*
    * @brief GridFeatures Organize features based on the grid
    *    they belong to. Note that the key is encoded by the
    *    grid index.
    */
    typedef std::map<int, std::vector<FeatureMetaData>> GridFeatures;

    bool is_first_img;

    double cam_imu_sync_dt;
    double dtime;

    // ID for the next new feature.
    FeatureIDType next_feature_id;

    cv::Vec3f gyro_bias;

    // Camera calibration parameters
    std::string cam0_distortion_model;
    cv::Vec2i cam0_resolution;
    cv::Vec4d cam0_intrinsics;
    cv::Vec4d cam0_distortion_coeffs;

    std::string cam1_distortion_model;
    cv::Vec2i cam1_resolution;
    cv::Vec4d cam1_intrinsics;
    cv::Vec4d cam1_distortion_coeffs;

    // Take a vector from cam0 frame to the IMU frame.
    cv::Matx33d R_cam0_imu;
    cv::Vec3d t_cam0_imu;
    // Take a vector from cam1 frame to the IMU frame.
    cv::Matx33d R_cam1_imu;
    cv::Vec3d t_cam1_imu;

    // Previous and current images
    cv_bridge::CvImageConstPtr cam0_prev_img_ptr;
    cv_bridge::CvImageConstPtr cam0_curr_img_ptr;
    cv_bridge::CvImageConstPtr cam1_curr_img_ptr;

    // Pyramids for previous and current image
    std::vector<cv::Mat> prev_cam0_pyramid_;
    std::vector<cv::Mat> curr_cam0_pyramid_;
    std::vector<cv::Mat> curr_cam1_pyramid_;

    // Features in the previous and current image.
    std::shared_ptr<GridFeatures> prev_features_ptr;
    std::shared_ptr<GridFeatures> curr_features_ptr;

    int before_tracking;
    int after_tracking;
    int after_matching;
    int after_ransac;
    int prev_feature_num;
    int curr_feature_num;

    perf_counter_t  _perf_feed_cam;
    perf_counter_t  _perf_track;
    perf_counter_t  _perf_imu_error;
    perf_counter_t  _perf_image_drop;

    void createImagePyramids();

    void twoPointRansac(const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2,
        const cv::Matx33f& R_p_c,
        const cv::Vec4d& intrinsics,
        const std::string& distortion_model,
        const cv::Vec4d& distortion_coeffs,
        const double& inlier_error,
        const double& success_probability,
        std::vector<int>& inlier_markers);

    void undistortPoints(const std::vector<cv::Point2f>& pts_in,
        const cv::Vec4d& intrinsics,
        const std::string& distortion_model,
        const cv::Vec4d& distortion_coeffs,
        std::vector<cv::Point2f>& pts_out,
        const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
        const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));
    
    std::vector<cv::Point2f> distortPoints(const std::vector<cv::Point2f>& pts_in,
        const cv::Vec4d& intrinsics,
        const std::string& distortion_model,
        const cv::Vec4d& distortion_coeffs);
    
    void rescalePoints(std::vector<cv::Point2f>& pts1,
        std::vector<cv::Point2f>& pts2,
        float& scaling_factor);

    void integrateImuData(cv::Matx33f& cam0_R_p_c, cv::Matx33f& cam1_R_p_c);

    void predictFeatureTracking(const std::vector<cv::Point2f>& input_pts,
        const cv::Matx33f& R_p_c,
        const cv::Vec4d& intrinsics,
        std::vector<cv::Point2f>& compenstated_pts);

    void stereoMatch(const std::vector<cv::Point2f>& cam0_points,
        std::vector<cv::Point2f>& cam1_points,
        std::vector<unsigned char>& inlier_markers);

    void initializeFirstFrame();

    void trackFeatures();

    void addNewFeatures();

    void pruneGridFeatures();

    void drawFeaturesStereo();

    void calculate_features();

    void getParameter();

    template<class T>
    void setNgetNodeParameter(T& param, 
        const std::string& param_name, 
        const T& default_value, 
        const rcl_interfaces::msg::ParameterDescriptor &parameter_descriptor=rcl_interfaces::msg::ParameterDescriptor());

    /*
    * @brief keyPointCompareByResponse
    *    Compare two keypoints based on the response.
    */
    static bool keyPointCompareByResponse(
        const cv::KeyPoint& pt1,
        const cv::KeyPoint& pt2) 
    {
        // Keypoint with higher response will be at the
        // beginning of the vector.
        return pt1.response > pt2.response;
    }
    /*
    * @brief featureCompareByResponse
    *    Compare two features based on the response.
    */
    static bool featureCompareByResponse(
        const FeatureMetaData& f1,
        const FeatureMetaData& f2) 
    {
        // Features with higher response will be at the
        // beginning of the vector.
        return f1.response > f2.response;
    }
    /*
    * @brief featureCompareByLifetime
    *    Compare two features based on the lifetime.
    */
    static bool featureCompareByLifetime(
        const FeatureMetaData& f1,
        const FeatureMetaData& f2) 
    {
        // Features with longer lifetime will be at the
        // beginning of the vector.
        return f1.lifetime > f2.lifetime;
    }

    /*
    * @brief removeUnmarkedElements Remove the unmarked elements
    *    within a vector.
    * @param raw_vec: vector with outliers.
    * @param markers: 0 will represent a outlier, 1 will be an inlier.
    * @return refined_vec: a vector without outliers.
    *
    * Note that the order of the inliers in the raw_vec is perserved
    * in the refined_vec.
    */
    template <typename T>
    void removeUnmarkedElements(const std::vector<T>& raw_vec,
        const std::vector<unsigned char>& markers,
        std::vector<T>& refined_vec) 
    {
        if (raw_vec.size() != markers.size()) {
            printf("The input size of raw_vec(%lu) and markers(%lu) does not match...", raw_vec.size(), markers.size());
        }
        for (uint32_t i = 0; i < markers.size(); ++i) {
            if (markers[i] == 0) {
                continue;
            }
            refined_vec.push_back(raw_vec[i]);
        }
    }
};

#endif
