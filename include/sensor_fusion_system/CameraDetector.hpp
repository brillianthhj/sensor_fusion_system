#ifndef LANE_DETECTOR_HPP_
#define LANE_DETECTOR_HPP_

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>

/// create your lane detecter
/// Class naming.. it's up to you.
namespace Xycar {
template <typename PREC>
class CameraDetector final
{
public:
    using Ptr = CameraDetector*; /// < Pointer type of the class(it's up to u)

    static inline const cv::Scalar kRed = {0, 0, 255}; /// Scalar values of Red
    static inline const cv::Scalar kGreen = {0, 255, 0}; /// Scalar values of Green
    static inline const cv::Scalar kBlue = {255, 0, 0}; /// Scalar values of Blue

    CameraDetector(const YAML::Node& config) {setConfiguration(config);}
    void undistortAndDNNConfig();
    void boundingBox(const cv::Mat img);

private:
    int32_t mImageWidth;
    int32_t mImageHeight;
    cv::Size mImageSize;
    cv::Mat mCameraMatrix = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat mDistCoeffs = cv::Mat::eye(1, 5, CV_32F);
    cv::Mat mMap1, mMap2;
    cv::Mat mTemp, mFrame;

    cv::dnn::Net mNeuralNet;
    
    std::string mYoloConfig;
    std::string mYoloModel;
    std::string mYoloLabel;

    std::vector<std::string> mClassNames;
    std::vector<std::string> mOutputLayers;

    // Debug Image and flag
    cv::Mat mDebugFrame; /// < The frame for debugging
    void setConfiguration(const YAML::Node& config);
    bool mDebugging;
};
}

#endif // LANE_DETECTOR_HPP_