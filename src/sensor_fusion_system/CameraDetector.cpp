// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file CameraDetector.cpp
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
    mImageSize = cv::Size(mImageWidth, mImageHeight);
    
    // Camera Matrix
    std::vector<std::vector<float>> matrixData;
    for (const auto& row : config["CAMERA"]["CAMERA_MATRIX"]) {
        std::vector<float> rowVector;
        for (const auto& ele : row) {
            rowVector.emplace_back(ele.as<float>());
        }
        matrixData.push_back(rowVector);
    }

    for (int i = 0; i < mCameraMatrix.rows; ++i) {
        for (int j = 0; j < mCameraMatrix.cols; ++j) {
            mCameraMatrix.at<float>(i, j) = matrixData[i][j];
        }
    }
    
    // Dist Coeffs
    std::vector<float> distMatrixData;
    for (const auto& row : config["CAMERA"]["DIST_COEFF"]) {
        distMatrixData.emplace_back(row.as<float>());
    }
    mDistCoeffs = cv::Mat(distMatrixData, true);

    mDebugging = config["DEBUG"].as<bool>();
}

template <typename PREC>
void CameraDetector<PREC>::undistortMatrix(){
    cv::initUndistortRectifyMap(mCameraMatrix, mDistCoeffs, cv::Mat(), mCameraMatrix, mImageSize, CV_32FC1, mMap1, mMap2);
}

template <typename PREC>
void CameraDetector<PREC>::boundingBox(const cv::Mat img){
    if (img.empty()){
        std::cerr << "Not image" << std::endl;
    }
    else{
        mTemp = img.clone();
        cv::remap(img, mTemp, mMap1, mMap2, cv::INTER_LINEAR);
        cv::imshow("undistort_img", mTemp);
        cv::waitKey(1);
    }    
}




template class CameraDetector<float>;
template class CameraDetector<double>;
} // namespace Xycar
