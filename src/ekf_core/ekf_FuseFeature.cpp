#include <ekf_core/ekf_Fusion.h>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

using namespace math_lib;
using namespace Eigen;
using namespace msckf_core;

void ekf_Fusion::MeasurementJacobian(const StateIDType& cam_state_id, const FeatureIDType& feature_id,
    Matrix<double, 4, 6>& H_x, Matrix<double, 4, 3>& H_f, Vector4d& r) 
{
    // Prepare all the required data.
    const CAMState& cam_state = state_server.cam_states[cam_state_id];
    const Feature& feature = map_server[feature_id];

    // Cam0 pose.
    Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
    const Vector3d& t_c0_w = cam_state.position;

    // Cam1 pose.
    Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
    Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
    Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

    // 3d feature position in the world frame.
    // And its observation with the stereo cameras.
    const Vector3d& p_w = feature.position;
    const Vector4d& z = feature.observations.find(cam_state_id)->second;

    // Convert the feature position from the world frame to
    // the cam0 and cam1 frame.
    Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
    Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

    // Compute the Jacobians.
    Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
    dz_dpc0(0, 0) = 1 / p_c0(2);
    dz_dpc0(1, 1) = 1 / p_c0(2);
    dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
    dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

    Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
    dz_dpc1(2, 0) = 1 / p_c1(2);
    dz_dpc1(3, 1) = 1 / p_c1(2);
    dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
    dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

    Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
    dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
    dpc0_dxc.rightCols(3) = -R_w_c0;

    Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
    dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
    dpc1_dxc.rightCols(3) = -R_w_c1;

    Matrix3d dpc0_dpg = R_w_c0;
    Matrix3d dpc1_dpg = R_w_c1;

    H_x = dz_dpc0*dpc0_dxc + dz_dpc1*dpc1_dxc;
    H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

    // Modifty the measurement Jacobian to ensure
    // observability constrain.
    Matrix<double, 4, 6> A = H_x;
    Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
    u.block<3, 1>(0, 0) = quaternionToRotation(cam_state.orientation_null) * ekf_Fusion::gravity;
    u.block<3, 1>(3, 0) = skewSymmetric(p_w-cam_state.position_null) * ekf_Fusion::gravity;
    H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
    H_f = -H_x.block<4, 3>(0, 3);

    // Compute the residual.
    r = z - Vector4d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2), p_c1(0)/p_c1(2), p_c1(1)/p_c1(2));

    return;
}

void ekf_Fusion::FeatureJacobian(const FeatureIDType& feature_id,const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) 
{
    const auto& feature = map_server[feature_id];

    // Check how many camera states in the provided camera
    // id camera has actually seen this feature.
    std::vector<StateIDType> valid_cam_state_ids(0);
    for (const auto& cam_id : cam_state_ids) {
        if (feature.observations.find(cam_id) == feature.observations.end()) {
            continue;
        }

        valid_cam_state_ids.push_back(cam_id);
    }

    int jacobian_row_size = 0;
    jacobian_row_size = 4 * valid_cam_state_ids.size();

    MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size, 22+state_server.cam_states.size()*6);
    MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
    VectorXd r_j = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    for (const auto& cam_id : valid_cam_state_ids) {
        Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
        Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
        Vector4d r_i = Vector4d::Zero();
        MeasurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

        auto cam_state_iter = state_server.cam_states.find(cam_id);
        int cam_state_cntr = std::distance(state_server.cam_states.begin(), cam_state_iter);

        // Stack the Jacobians.
        H_xj.block<4, 6>(stack_cntr, 22+6*cam_state_cntr) = H_xi;
        H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
        r_j.segment<4>(stack_cntr) = r_i;
        stack_cntr += 4;
    }

    // Project the residual and Jacobians onto the nullspace
    // of H_fj.
    JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
    MatrixXd A = svd_helper.matrixU().rightCols(jacobian_row_size - 3);

    H_x = A.transpose() * H_xj;
    r = A.transpose() * r_j;
}

bool ekf_Fusion::MeasurementUpdate(const MatrixXd& H, const VectorXd& r) 
{
    if (H.rows() == 0 || r.rows() == 0) {
        return false;
    }

    // Decompose the final Jacobian matrix to reduce computational
    // complexity as in Equation (28), (29).
    MatrixXd H_thin;
    VectorXd r_thin;

    if (H.rows() > H.cols()) {
        // Convert H to a sparse matrix.
        SparseMatrix<double> H_sparse = H.sparseView();

        // Perform QR decompostion on H_sparse.
        SPQR<SparseMatrix<double>> spqr_helper;
        spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
        spqr_helper.compute(H_sparse);

        MatrixXd H_temp;
        VectorXd r_temp;
        (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
        (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

        H_thin = H_temp.topRows(22+state_server.cam_states.size()*6);
        r_thin = r_temp.head(22+state_server.cam_states.size()*6);

        //HouseholderQR<MatrixXd> qr_helper(H);
        //MatrixXd Q = qr_helper.householderQ();
        //MatrixXd Q1 = Q.leftCols(22+state_server.cam_states.size()*6);

        //H_thin = Q1.transpose() * H;
        //r_thin = Q1.transpose() * r;
    } else {
        H_thin = H;
        r_thin = r;
    }

    // Compute the Kalman gain.
    const double obs_noise = sq(_observation_noise);
    const MatrixXd& P = state_server.state_cov;
    MatrixXd S = H_thin*P*H_thin.transpose() + obs_noise*MatrixXd::Identity(H_thin.rows(), H_thin.rows());
    //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
    MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
    MatrixXd K = K_transpose.transpose();

    // Compute the error of the state.
    VectorXd delta_x = K * r_thin;

    // Update the IMU state.
    VectorXd delta_x_imu = delta_x.head<22>();

    updateState(delta_x_imu);

    if(_estimate_external) {
        const Vector4d dq_extrinsic = smallAngleQuaternion(delta_x_imu.segment<3>(15));
        state_server.imu_state.R_imu_cam0 = quaternionToRotation(dq_extrinsic) * state_server.imu_state.R_imu_cam0;
        state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);
    }

    state_server.imu_state.cam_imu_dt += delta_x_imu(21);

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (uint32_t i = 0; i < state_server.cam_states.size(); ++i, ++cam_state_iter) {
        const VectorXd& delta_x_cam = delta_x.segment<6>(22+i*6);
        const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
        cam_state_iter->second.orientation = quaternionMultiplication(dq_cam, cam_state_iter->second.orientation);
        cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    // Update state covariance.
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
    // state_server.state_cov = I_KH*state_server.state_cov *I_KH.transpose() + K*obs_noise*K.transpose();
    state_server.state_cov = I_KH*state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    ConstrainVariances();

    return true;
}