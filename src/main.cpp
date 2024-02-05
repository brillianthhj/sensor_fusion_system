#include "sensor_fusion_system/LaneKeepingSystem.hpp"

using PREC = float;

int32_t main(int32_t argc, char** argv)
{
    ros::init(argc, argv, "Lane Keeping System");
    Xycar::LaneKeepingSystem<PREC> laneKeepingSystem;
    laneKeepingSystem.run();

    return 0;
}
