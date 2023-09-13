#include <ekf_core/ekf_Core.h>
#include <string>
#include <cstdio>
#include<iostream>

using namespace math_lib;

void ekf_Core::ControlConstFuse(void)
{
    bool dataReady =  ((imuSampleTime_ms-firstInitTime_ms) > 5000) && ((imuSampleTime_ms-last_cam_fusion_ms) > 200);
    double PosNoise = sq(0.5);
    if(dataReady) {
        perf_count(_perf_Const_pos, timestamp());
        if (tiltAlignComplete) {
            PosNoise = sq(_noaidHorizNoise);
        }
        FusePosNED(lastKnownPositionNE, PosNoise);
    }
}
