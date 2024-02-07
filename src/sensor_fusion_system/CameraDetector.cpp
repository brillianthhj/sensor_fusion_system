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

    mYoloConfig = config["YOLO"]["CONFIG"].as<std::string>();
    mYoloModel = config["YOLO"]["MODEL"].as<std::string>();
    mYoloLabel = config["YOLO"]["LABEL"].as<std::string>();

    mDebugging = config["DEBUG"].as<bool>();
}

template <typename PREC>
void CameraDetector<PREC>::undistortAndDNNConfig()
{
    cv::initUndistortRectifyMap(mCameraMatrix, mDistCoeffs, cv::Mat(), mCameraMatrix, mImageSize, CV_32FC1, mMap1, mMap2);
    
    mNeuralNet = cv::dnn::readNet(mYoloConfig, mYoloModel);

    // Neural Net setting
    if(mNeuralNet.empty()){
        std::cerr << "Network load failed!" << std::endl;
    }

#if 0
        mNeuralNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        mNeuralNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
#else
        mNeuralNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        mNeuralNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
#endif

    std::ifstream classNamesFile(mYoloLabel);
    if (classNamesFile.is_open()) {
        std::string className = "";
        while(std::getline(classNamesFile, className)) {
            mClassNames.emplace_back(className);
        }
    }
    mOutputLayers = mNeuralNet.getUnconnectedOutLayersNames();
}

template <typename PREC>
void CameraDetector<PREC>::boundingBox(const cv::Mat img)
{
    if (img.empty()) {
        // std::cerr << "No image.. Wait.." << std::endl;
    }
    else {
        // undistort image
        mTemp = img.clone();
        cv::remap(img, mTemp, mMap1, mMap2, cv::INTER_LINEAR);
        cv::Mat blob = cv::blobFromImage();


        
        
        
        cv::imshow("undistort_img", mTemp);
        cv::waitKey(1);
    }    
}




template class CameraDetector<float>;
template class CameraDetector<double>;
} // namespace Xycar
