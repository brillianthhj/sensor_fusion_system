// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file CameraDerector.cpp
 * @author Jinwoo Mok
 * @version 1.0
 * @date 2024-02-06
 */

#include <numeric>
#include "sensor_fusion_system/CameraDetector.hpp"

namespace Xycar {

template <typename PREC>
void CameraDetector<PREC>::setConfiguration(const YAML::Node& config)
{
    mImageWidth = config["IMAGE"]["WIDTH"].as<int32_t>();
    mImageHeight = config["IMAGE"]["HEIGHT"].as<int32_t>();

    std::vector<std::vector<float>> matrixData;
    for (const auto& row : config["CAMERA"]["CAMERA_MATRIX"]) {
        std::vector<float> rowVector = row.as<std::vector<float>>();
        matrixData.push_back(rowVector);
    }

    // for(int i = 0; i < mCameraMatrix.row(); i++){
    //     for(int j = 0; j < mCameraMatrix.col(); j++){
    //         mCameraMatrix.at<float>(i, j) = matrixData[i][j];
    //     }
    // }
    // std::cout << mCameraMatrix << std::endl;


    mDebugging = config["DEBUG"].as<bool>();
}

template <typename PREC>
cv::Mat CameraDetector<PREC>::undistortImage(const cv::Mat img){
    if (img.empty()){
        std::cerr << "Not image" << std::endl;
    }
    else{
        // cv::initUndistortRectifyMap()

    }
}




template class CameraDetector<float>;
template class CameraDetector<double>;
} // namespace Xycar
