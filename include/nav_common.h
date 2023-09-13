#ifndef NAV_COMMON_H_
#define NAV_COMMON_H_

#include <stdint.h>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <common/quaternion.h>
#include <sprain_msgs/msg/feature_measurement.hpp>

namespace nav 
{
struct imu_elements {
	Eigen::Vector3d    delAng;         // IMU delta angle measurements in body frame (rad)
	Eigen::Vector3d    delVel;         // IMU delta velocity measurements in body frame (m/sec)
	double       delAngDT;       // time interval over which delAng has been measured (sec)
	double       delVelDT;       // time interval over which delVelDT has been measured (sec)
	uint64_t    time_us;        // measurement timestamp (msec)
};

struct mag_elements {
	Eigen::Vector3d    mag;            // body frame magnetic field measurements (Gauss)
	uint64_t    time_us;        // measurement timestamp (msec)
};

struct att_elements {
	Eigen::Vector4d    quat;            // body frame magnetic field measurements (Gauss)
	uint64_t    time_us;        // measurement timestamp (msec)
};

struct feature_elements {
    uint64_t    seq;
	std::vector<sprain_msgs::msg::FeatureMeasurement> cam_feature;  // distance measured by the range sensor (m)
	uint64_t    time_us;        // measurement timestamp (msec)
};

// Fusion status
struct faultStatus {
    uint64_t last_beta_ts;
    uint64_t last_decl_ts;
    uint64_t last_imu_ts;

    bool bad_sideslip;
    bool bad_decl;
    bool bad_imu;

    bool useGpsVel;
    bool useGpsPos;
    bool fuseVelData;
    bool fusePosData;
    bool fuseHgtData;
};

union filter_status {
    struct {
        bool attitude           : 1;
        bool altitude           : 1;
        bool velocity           : 1;
        bool position           : 1;
        bool constant           : 1;
        bool terrain_alt        : 1; 
        bool const_pos_mode     : 1;
        bool using_gps          : 1;
        bool using_vio          : 1;
        bool gps_glitching      : 1; 
        bool gps_quality_good   : 1; 
    } flags;
    uint16_t value;
};

union fusion_status {
    struct {
        bool gps_pos            : 1;
        bool gps_vel            : 1;
        bool gps_hgt            : 1;
        bool baro_hgt           : 1;
        bool rng_hgt            : 1; 
        bool vio_vel            : 1;
        bool vio_pos            : 1;
        bool vio_hgt            : 1;
        bool of_vel             : 1;
        bool mag_earth          : 1; 
        bool yaw_mag            : 1; 
        bool yaw_ext            : 1;
        bool yaw_gsf            : 1;
        bool terrain            : 1;
    } flags;
    uint16_t value;
};
}
#endif
