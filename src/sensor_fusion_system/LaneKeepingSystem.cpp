#include "sensor_fusion_system/LaneKeepingSystem.hpp"

namespace Xycar {
template <typename PREC>
LaneKeepingSystem<PREC>::LaneKeepingSystem()
{
    std::string configPath;
    mNodeHandler.getParam("config_path", configPath);
    YAML::Node config = YAML::LoadFile(configPath);

    mPID = new PIDController<PREC>(config["PID"]["P_GAIN"].as<PREC>(), config["PID"]["I_GAIN"].as<PREC>(), config["PID"]["D_GAIN"].as<PREC>());
    mMovingAverage = new MovingAverageFilter<PREC>(config["MOVING_AVERAGE_FILTER"]["SAMPLE_SIZE"].as<uint32_t>());
    mCameraDetector = new CameraDetector<PREC>(config);
    /*
        create your lane detector.
    */
    setParams(config);

    mPublisher = mNodeHandler.advertise<xycar_msgs::xycar_motor>(mPublishingTopicName, mQueueSize);
    mSubscriber = mNodeHandler.subscribe(mSubscribedTopicName, mQueueSize, &LaneKeepingSystem::imageCallback, this);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::setParams(const YAML::Node& config)
{
    mPublishingTopicName = config["TOPIC"]["PUB_NAME"].as<std::string>();
    mSubscribedTopicName = config["TOPIC"]["SUB_NAME"].as<std::string>();
    mQueueSize = config["TOPIC"]["QUEUE_SIZE"].as<uint32_t>();
    mXycarSpeed = config["XYCAR"]["START_SPEED"].as<PREC>();
    mXycarMaxSpeed = config["XYCAR"]["MAX_SPEED"].as<PREC>();
    mXycarMinSpeed = config["XYCAR"]["MIN_SPEED"].as<PREC>();
    mXycarSpeedControlThreshold = config["XYCAR"]["SPEED_CONTROL_THRESHOLD"].as<PREC>();
    mAccelerationStep = config["XYCAR"]["ACCELERATION_STEP"].as<PREC>();
    mDecelerationStep = config["XYCAR"]["DECELERATION_STEP"].as<PREC>();
    mDebugging = config["DEBUG"].as<bool>();
}

template <typename PREC>
LaneKeepingSystem<PREC>::~LaneKeepingSystem()
{
    delete mPID;
    delete mMovingAverage;
    // delete your CameraDetector if you add your CameraDetector.
}

template <typename PREC>
void LaneKeepingSystem<PREC>::run()
{
    ros::Rate rate(kFrameRate);
    mCameraDetector->undistortMatrix();
    while (ros::ok())
    {
        ros::spinOnce();
        mCameraDetector->boundingBox(mFrame);
        
    }
}

template <typename PREC>
void LaneKeepingSystem<PREC>::imageCallback(const sensor_msgs::Image& message)
{
    cv::Mat src = cv::Mat(message.height, message.width, CV_8UC3, const_cast<uint8_t*>(&message.data[0]), message.step);
    cv::cvtColor(src, mFrame, cv::COLOR_RGB2BGR);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::speedControl(PREC steeringAngle)
{
    if (std::abs(steeringAngle) > mXycarSpeedControlThreshold)
    {
        mXycarSpeed -= mDecelerationStep;
        mXycarSpeed = std::max(mXycarSpeed, mXycarMinSpeed);
        return;
    }

    mXycarSpeed += mAccelerationStep;
    mXycarSpeed = std::min(mXycarSpeed, mXycarMaxSpeed);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::drive(PREC steeringAngle)
{
    xycar_msgs::xycar_motor motorMessage;
    motorMessage.angle = std::round(steeringAngle);
    motorMessage.speed = std::round(mXycarSpeed);

    mPublisher.publish(motorMessage);
}

template class LaneKeepingSystem<float>;
template class LaneKeepingSystem<double>;
} // namespace Xycar
