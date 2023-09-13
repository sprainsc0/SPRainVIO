/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_IMU_STATE_H
#define MSCKF_VIO_IMU_STATE_H

#include <map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <common/quaternion.h>

namespace msckf_core {

/*
 * @brief IMUState State for IMU
 */
struct IMUState {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	typedef long long int StateIDType;

	// An unique identifier for the IMU state.
	StateIDType id;

	// id for next IMU state
	static StateIDType next_id;

	// Time when the state is recorded
	uint64_t timestamp;
	
	Eigen::Vector4d orientation;

	Eigen::Vector3d angErr;

	Eigen::Vector3d gyro_bias;

	Eigen::Vector3d velocity;

	Eigen::Vector3d acc_bias;

	Eigen::Vector3d position;

	Eigen::Matrix3d R_imu_cam0;
	Eigen::Vector3d t_cam0_imu;

	double cam_imu_dt;

	// These three variables should have the same physical
	// interpretation with `orientation`, `position`, and
	// `velocity`. There three variables are used to modify
	// the transition matrices to make the observability matrix
	// have proper null space.
	Eigen::Vector4d orientation_null;
	Eigen::Vector3d position_null;
	Eigen::Vector3d velocity_null;

	static constexpr int rot_ids = 0;
	static constexpr int bg_ids  = 3;
	static constexpr int vel_ids = 6;
	static constexpr int ba_ids  = 9;
	static constexpr int pos_ids = 12;
	static constexpr int qci_ids = 15;
	static constexpr int pci_ids = 18;
	static constexpr int dt_ids  = 21;

	IMUState(): id(0), timestamp(0), 
		orientation(Eigen::Vector4d(0, 0, 0, 1)),
		angErr(Eigen::Vector3d::Zero()),
		gyro_bias(Eigen::Vector3d::Zero()),
		velocity(Eigen::Vector3d::Zero()),
		acc_bias(Eigen::Vector3d::Zero()),
		position(Eigen::Vector3d::Zero()),
		cam_imu_dt(0.0),
		orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
		position_null(Eigen::Vector3d::Zero()),
		velocity_null(Eigen::Vector3d::Zero()) 
	{
	}

	IMUState(const StateIDType& new_id): id(new_id), timestamp(0), 
		orientation(Eigen::Vector4d(0, 0, 0, 1)),
		angErr(Eigen::Vector3d::Zero()),
		gyro_bias(Eigen::Vector3d::Zero()),
		velocity(Eigen::Vector3d::Zero()),
		acc_bias(Eigen::Vector3d::Zero()),
		position(Eigen::Vector3d::Zero()),
		cam_imu_dt(0.0),
		orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
		position_null(Eigen::Vector3d::Zero()),
		velocity_null(Eigen::Vector3d::Zero()) 
	{
	}
};

typedef IMUState::StateIDType StateIDType;

} // namespace msckf_core

#endif // MSCKF_VIO_IMU_STATE_H
