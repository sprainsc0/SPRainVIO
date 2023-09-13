#include <ekf_core/ekf_Core.h>
#include <string>
#include <cstdio>
#include<iostream>

using namespace math_lib;

void ekf_Core::ControlAttFuse(void)
{
    bool attDataToFuse = false;
    uint16_t used_att_msg_cntr = 0;

    for (const auto& att_msg : att_msg_buffer) {
        if (att_msg.time_us < state_server.imu_state.timestamp) {
            if((state_server.imu_state.timestamp-att_msg.time_us) < 50000) {
                attDataToFuse = true;
                attDataDelayed = att_msg;
            }
            ++used_att_msg_cntr;
        } else {
            break;
        }
    }
    attitude_queue_mtx.lock();
    att_msg_buffer.erase(att_msg_buffer.begin(), att_msg_buffer.begin()+used_att_msg_cntr);
    attitude_queue_mtx.unlock();

    if(attDataToFuse) {
        perf_count(_perf_Att_Interval, timestamp());
        controlMagYawReset();
    }

    bool dataReady =  (attDataToFuse && statesInitialised && yawAlignComplete);

    if(dataReady) {
        perf_begin(_perf_FuseAttitude, timestamp());
        FuseExtAtt(attDataDelayed.quat, sq(_yawNoise));
        perf_end(_perf_FuseAttitude, timestamp());
    }
}

void ekf_Core::controlMagYawReset(void)
{
    Eigen::Vector3d deltaRotVecTemp;
    Eigen::Vector4d deltaQuatTemp;

    bool initialResetAllowed = false;
    if (!yawAlignComplete) {
        // 两次对准间的姿态角变化，仅允许低速下对准
        deltaQuatTemp = quaternion_division(state_server.imu_state.orientation, prevQuatMagReset);
        prevQuatMagReset = state_server.imu_state.orientation;

        deltaRotVecTemp = QuaternionToAxisAngle(deltaQuatTemp);

        bool angRateOK = deltaRotVecTemp.norm() < 0.1745;

        initialResetAllowed = angRateOK;
    }

    // 初始化的航向对准
    bool initialResetRequest = initialResetAllowed && !yawAlignComplete;

    magYawResetRequest = magYawResetRequest || // 外部地磁航向对准请求，倾斜角对准后
            initialResetRequest;               // 初始化地磁航向对准

    if (magYawResetRequest) {

        state_server.imu_state.orientation = attDataDelayed.quat;

        if (!yawAlignComplete) {
            printf("Nav initial yaw aligned \n");
        }

        // update the yaw reset completed status
        recordYawReset();

        // clear the yaw reset request flag
        magYawResetRequest = false;
    }
}

void ekf_Core::recordYawReset()
{
    yawAlignComplete = true;
}
