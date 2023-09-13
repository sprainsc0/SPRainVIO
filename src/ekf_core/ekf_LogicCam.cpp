#include <ekf_core/ekf_Core.h>
#include <string>
#include <cstdio>
#include <iostream>

using namespace math_lib;
using namespace msckf_core;
using namespace Eigen;

void ekf_Core::ControlCamFuse(void)
{
    if(tiltAlignComplete && (yawAlignComplete || !_param.external_att)) {
        perf_count(_perf_CAM_Interval, timestamp());

        if(camDataDelayed.cam_feature.size() < 6) {
            RCLCPP_INFO_THROTTLE(_node->get_logger(), *(_node->get_clock()), 1000.0,"feature point lost, size:%d", camDataDelayed.cam_feature.size());
            return;
        }

        perf_begin(_perf_FuseCAM, timestamp());

        stateAugmentation();

        addFeatureObservations();

        removeLostFeatures();

        pruneCamStateBuffer();

        perf_end(_perf_FuseCAM, timestamp());
    }
}

void ekf_Core::addFeatureObservations(void) 
{
    StateIDType state_id = state_server.imu_state.id;
    int curr_feature_num = map_server.size();
    int tracked_feature_num = 0;

    // Add new observations for existing features or new
    // features in the map server.
    // 遍历所有特征点
    for (const auto& feature : camDataDelayed.cam_feature) {
        if (map_server.find(feature.id) == map_server.end()) {
            // 新的特征点
            map_server[feature.id] = Feature(feature.id);
            map_server[feature.id].observations[state_id] = Vector4d(feature.u0, feature.v0, feature.u1, feature.v1);
        } else {
            // 旧的特征点
            map_server[feature.id].observations[state_id] = Vector4d(feature.u0, feature.v0, feature.u1, feature.v1);
            ++tracked_feature_num;
        }
    }
    // 计算还在跟踪德特征点比率
    tracking_rate = static_cast<double>(tracked_feature_num) / static_cast<double>(curr_feature_num);
}

bool ekf_Core::gatingTest(const MatrixXd& H, const VectorXd& r, const int& dof) 
{
    MatrixXd P1 = H * state_server.state_cov * H.transpose();
    MatrixXd P2 = sq(_observation_noise) * MatrixXd::Identity(H.rows(), H.rows());
    double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

    //cout << dof << " " << gamma << " " <<
    //  chi_squared_test_table[dof] << " ";

    if (gamma < chi_squared_test_table[dof]) {
        return true;
    } else {
        return false;
    }
}

//step1：从跟丢的特征中筛选可以三角化，并进行三角化
/*筛选条件：step1.1：该特征的被观测次数>=3次
 *        step1.2：如果点没有被初始化，
 *                 1.该特征第一个和最后一个观测帧之间的运动足够大（视差大）
 *                 2.三角化后所有点都在相机前面
 *（已经初始化且满足step1.1的点可用于测量更新）
 */
//step2：进行EKF测量更新
//step3：删除所有跟丢的特征
void ekf_Core::removeLostFeatures()
{
    // Remove the features that lost track.
    // BTW, find the size the final Jacobian matrix and residual vector.
    int jacobian_row_size = 0;
    std::vector<FeatureIDType> invalid_feature_ids(0);
    std::vector<FeatureIDType> processed_feature_ids(0);

    for (auto iter = map_server.begin(); iter != map_server.end(); ++iter) {
        // Rename the feature to be checked.
        auto& feature = iter->second;

        // 最新IMU_ID存在，表示还在追踪或新加入德特征点,不参与筛选
        if (feature.observations.find(state_server.imu_state.id) != feature.observations.end()) {
            continue;
        }
        // 特征点观测次数小于3次，抛弃
        if (feature.observations.size() < 3) {
            invalid_feature_ids.push_back(feature.id);
            continue;
        }

        // Check if the feature can be initialized if it
        // has not been.
        if (!feature.is_initialized) {
            //如果没有足够大的运动，判定该特征为invalid_feature
            if (!feature.checkMotion(state_server.cam_states)) {
                invalid_feature_ids.push_back(feature.id);
                perf_count(_perf_CheckMove_Error, timestamp());
                continue;
            } else {
                //如果不满足三角化条件，则判断为invalid_feature
                if(!feature.initializePosition(state_server.cam_states)) {
                    invalid_feature_ids.push_back(feature.id);
                    perf_count(_perf_InitPos_Error, timestamp());
                    continue;
                }
            }
        }

        jacobian_row_size += 4*feature.observations.size() - 3;
        // 三角化后的特征
        processed_feature_ids.push_back(feature.id);
    }

    //cout << "invalid/processed feature #: " <<
    //  invalid_feature_ids.size() << "/" <<
    //  processed_feature_ids.size() << endl;
    //cout << "jacobian row #: " << jacobian_row_size << endl;

    // Remove the features that do not have enough measurements.
    for (const auto& feature_id : invalid_feature_ids) {
        map_server.erase(feature_id);
    }

    // Return if there is no lost feature to be processed.
    if (processed_feature_ids.size() == 0) {
        return;
    }

    //step2：进行EKF测量更新
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 22+6*state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // Process the features which lose track.
    for (const auto& feature_id : processed_feature_ids) {
        auto& feature = map_server[feature_id];

        std::vector<StateIDType> cam_state_ids(0);
        for (const auto& measurement : feature.observations) {
            cam_state_ids.push_back(measurement.first);
        }

        MatrixXd H_xj;
        VectorXd r_j;
        FeatureJacobian(feature.id, cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        } else {
            perf_count(_perf_Gating1_Error, timestamp());
        }

        // Put an upper bound on the row size of measurement Jacobian,
        // which helps guarantee the executation time.
        if (stack_cntr > 1500) {
            break;
        }
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    if(MeasurementUpdate(H_x, r)) {
        perf_count(_perf_Remove_Interval, timestamp());
        lastKnownPositionNE.x() = state_server.imu_state.position.x();
        lastKnownPositionNE.y() = state_server.imu_state.position.y();
        lastKnownPositionNE.z() = state_server.imu_state.position.z();
        last_cam_fusion_ms = imuSampleTime_ms;
    } else {
        perf_count(_perf_Update_failed, timestamp());
    }

    //step3：删除所有跟丢的特征
    for (const auto& feature_id : processed_feature_ids) {
        map_server.erase(feature_id);
    }
}

//当cam state数达到阈值时，进行cam状态剔除,要剔除两帧
/*剔除策略：根据平移，旋转角度，特征跟踪率，判断次次新帧和次新帧距次次次新帧的运动大小
 *        如果运动小于阈值，就剔除对应的帧
 *        否则，剔除最老的帧
 *输出：相机状态id rm_cam_state_ids
 */
void ekf_Core::findRedundantCamStates(std::vector<StateIDType>& rm_cam_state_ids) 
{
    // Move the iterator to the key position.
    auto key_cam_state_iter = state_server.cam_states.end();
    for (int i = 0; i < 4; ++i) {
        --key_cam_state_iter;
    }
    auto cam_state_iter = key_cam_state_iter;
    ++cam_state_iter;
    auto first_cam_state_iter = state_server.cam_states.begin();

    // Pose of the key camera state.
    const Vector3d key_position = key_cam_state_iter->second.position;
    const Matrix3d key_rotation = quaternionToRotation(key_cam_state_iter->second.orientation);

    // Mark the camera states to be removed based on the
    // motion between states.
    for (int i = 0; i < 2; ++i) {
        const Vector3d position = cam_state_iter->second.position;
        const Matrix3d rotation = quaternionToRotation(cam_state_iter->second.orientation);

        double distance = (position-key_position).norm();
        double angle = AngleAxisd(rotation*key_rotation.transpose()).angle();

        if (angle < _param.rotation_threshold &&
            distance < _param.translation_threshold &&
            tracking_rate > _param.tracking_rate_threshold) {
            rm_cam_state_ids.push_back(cam_state_iter->first);
            ++cam_state_iter;
        } else {
            rm_cam_state_ids.push_back(first_cam_state_iter->first);
            ++first_cam_state_iter;
        }
    }

    // Sort the elements in the output vector.
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());
}

//step1：找到应该剔除的cam状态
/*step2：遍历被待删帧观测到的所有特征点，
 *       如果该特征没初始化且不能初始化（三角化），则删除该特征点中待删相机的观测
 *       如果该特征没初始化且能初始化（三角化），则进行三角化，然后进行EKF测量更新
 *       最后删除该特征点中待删相机的观测，删除协方差矩阵的对应块，及对应的cam状态
 */
void ekf_Core::pruneCamStateBuffer() 
{
    if (state_server.cam_states.size() < _param.max_cam_state_size) {
        return;
    }

    // Find two camera states to be removed.
    std::vector<StateIDType> rm_cam_state_ids(0);
    findRedundantCamStates(rm_cam_state_ids);

    /*step2：遍历被待删帧观测到的所有特征点，
     *       如果该特征不能初始化（三角化），则删除该特征点中待删相机的观测
     *       如果该特征能初始化（三角化），则进行三角化，然后进行EKF测量更新
     *       最后删除该特征点中待删相机的观测，删除协方差矩阵的对应块，及对应的cam状态
     */
    int jacobian_row_size = 0;
    for (auto& item : map_server) {
        auto& feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        std::vector<StateIDType> involved_cam_state_ids(0);
        for (const auto& cam_id : rm_cam_state_ids) {
            if (feature.observations.find(cam_id) != feature.observations.end()){
                involved_cam_state_ids.push_back(cam_id);
            }
        }

        if (involved_cam_state_ids.size() == 0) {
            continue;
        }
        if (involved_cam_state_ids.size() == 1) {
            feature.observations.erase(involved_cam_state_ids[0]);
            continue;
        }

        if (!feature.is_initialized) {
            // Check if the feature can be initialize.
            if (!feature.checkMotion(state_server.cam_states)) {
                // If the feature cannot be initialized, just remove
                // the observations associated with the camera states
                // to be removed.
                for (const auto& cam_id : involved_cam_state_ids) {
                    feature.observations.erase(cam_id);
                }
                perf_count(_perf_CheckMove_Error, timestamp());
                continue;
            } else {
                if(!feature.initializePosition(state_server.cam_states)) {
                    for (const auto& cam_id : involved_cam_state_ids) {
                        feature.observations.erase(cam_id);
                    }
                    perf_count(_perf_InitPos_Error, timestamp());
                    continue;
                }
            }
        }

        jacobian_row_size += 4*involved_cam_state_ids.size() - 3;
    }

    //cout << "jacobian row #: " << jacobian_row_size << endl;

    // Compute the Jacobian and residual.
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 22+6*state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    for (auto& item : map_server) {
        auto& feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        std::vector<StateIDType> involved_cam_state_ids(0);
        for (const auto& cam_id : rm_cam_state_ids) {
            if (feature.observations.find(cam_id) != feature.observations.end()) {
                involved_cam_state_ids.push_back(cam_id);
            }
        }

        if (involved_cam_state_ids.size() == 0) {
            continue;
        }

        MatrixXd H_xj;
        VectorXd r_j;
        FeatureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        } else {
            perf_count(_perf_Gating2_Error, timestamp());
        }

        for (const auto& cam_id : involved_cam_state_ids) {
            feature.observations.erase(cam_id);
        }
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform measurement update.
    if(MeasurementUpdate(H_x, r)) {
        perf_count(_perf_Prune_Interval, timestamp());
        lastKnownPositionNE.x() = state_server.imu_state.position.x();
        lastKnownPositionNE.y() = state_server.imu_state.position.y();
        lastKnownPositionNE.z() = state_server.imu_state.position.z();
        last_cam_fusion_ms = imuSampleTime_ms;
    } else {
        perf_count(_perf_Update_failed, timestamp());
    }

    for (const auto& cam_id : rm_cam_state_ids) {
        int cam_sequence = std::distance(state_server.cam_states.begin(), state_server.cam_states.find(cam_id));
        int cam_state_start = 22 + 6*cam_sequence;
        int cam_state_end = cam_state_start + 6;

        // Remove the corresponding rows and columns in the state
        // covariance matrix.
        if (cam_state_end < state_server.state_cov.rows()) {
            state_server.state_cov.block(cam_state_start, 0, (state_server.state_cov.rows()-cam_state_end), state_server.state_cov.cols()) =
                state_server.state_cov.block(cam_state_end, 0, (state_server.state_cov.rows()-cam_state_end), state_server.state_cov.cols());

            state_server.state_cov.block(0, cam_state_start, state_server.state_cov.rows(), (state_server.state_cov.cols()-cam_state_end)) =
                state_server.state_cov.block(0, cam_state_end, state_server.state_cov.rows(), (state_server.state_cov.cols()-cam_state_end));

            state_server.state_cov.conservativeResize((state_server.state_cov.rows()-6), (state_server.state_cov.cols()-6));
        } else {
            state_server.state_cov.conservativeResize((state_server.state_cov.rows()-6), (state_server.state_cov.cols()-6));
        }

        // Remove this camera state in the state vector.
        state_server.cam_states.erase(cam_id);
    }
}