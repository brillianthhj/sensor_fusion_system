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

        bool isSuccess = cv::imwrite("/home/nvidia/hyejin_ws/src/sensor_fusion_system/image/undistort_img.png", mTemp);
    }
}

template <typename PREC>
void CameraDetector<PREC>::solvePnP(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point3f> objectPoints){

    std::cout << "There are " << imagePoints.size() << " imagePoints and " << objectPoints.size() << " objectPoints." << std::endl;
    std::cout << "Initial cameraMatrix: " << mCameraMatrix << std::endl;
    std::cout << "Initial distCoeffs: " << mDistCoeffs << std::endl;

    cv::Mat rvec(3,1,cv::DataType<double>::type);
    cv::Mat tvec(3,1,cv::DataType<double>::type);

    cv::solvePnP(objectPoints, imagePoints, mCameraMatrix, mDistCoeffs, rvec, tvec);

    std::cout << "rvec: " << rvec << std::endl;
    std::cout << "tvec: " << tvec << std::endl;

    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(objectPoints, rvec, tvec, mCameraMatrix, mDistCoeffs, projectedPoints);

    for(unsigned int i = 0; i < projectedPoints.size(); ++i)
    {
        std::cout << "Image point: " << imagePoints[i] << " Projected to " << projectedPoints[i] << std::endl;
    }
}

template <typename PREC>
std::vector<cv::Point2f> CameraDetector<PREC>::Generate2DPoints2()
{
  std::vector<cv::Point2f> points;

  points.push_back(cv::Point2f(84.857, 216.255));
  points.push_back(cv::Point2f(108.035, 215.645));
  points.push_back(cv::Point2f(192.209, 216.864));
  points.push_back(cv::Point2f(217.828, 216.864));
  points.push_back(cv::Point2f(300.172, 218.694));
  points.push_back(cv::Point2f(324.57, 218.084));
  points.push_back(cv::Point2f(408.134, 218.084));
  points.push_back(cv::Point2f(431.923, 219.304));
  points.push_back(cv::Point2f(514.267, 219.304));
  points.push_back(cv::Point2f(536.836, 219.304));
  points.push_back(cv::Point2f(84.857, 240.043));
  points.push_back(cv::Point2f(108.035, 240.653));

  for(unsigned int i = 0; i < points.size(); ++i)
    {
    std::cout << points[i] << std::endl;
    }

  return points;
}

template <typename PREC>
std::vector<cv::Point3f> CameraDetector<PREC>::Generate3DLidarPoints2()
{
  std::vector<cv::Point3f> points;

  points.push_back(cv::Point3f(-1.34259, -0.940092, 0.105));
  points.push_back(cv::Point3f(-1.34395, -0.840092, 0.105));
  points.push_back(cv::Point3f(-1.34139, -0.526456, 0.105));
  points.push_back(cv::Point3f(-1.34416, -0.426456, 0.105));
  points.push_back(cv::Point3f(-1.34638, -0.0840328, 0.105));
  points.push_back(cv::Point3f(-1.3479, 0.0159672, 0.105));
  points.push_back(cv::Point3f(-1.33974, 0.358982, 0.105));
  points.push_back(cv::Point3f(-1.34321, 0.451296, 0.105));
  points.push_back(cv::Point3f(-1.33782, 0.787527, 0.105));
  points.push_back(cv::Point3f(-1.33728, 0.887527, 0.105));
  points.push_back(cv::Point3f(-1.34259, -0.940092, 0));
  points.push_back(cv::Point3f(-1.34395, -0.840092, 0));

  for(unsigned int i = 0; i < points.size(); ++i)
    {
    std::cout << points[i] << std::endl;
    }

  return points;
}

template <typename PREC>
std::vector<cv::Point3f> CameraDetector<PREC>::Generate3DVCSPoints2()
{
  std::vector<cv::Point3f> points;

  points.push_back(cv::Point3f(1.8, 1.0, 0.105));
  points.push_back(cv::Point3f(1.8, 0.9, 0.105));
  points.push_back(cv::Point3f(1.8, 0.55, 0.105));
  points.push_back(cv::Point3f(1.8, 0.45, 0.105));
  points.push_back(cv::Point3f(1.8, 0.1, 0.105));
  points.push_back(cv::Point3f(1.8, 0.0, 0.105));
  points.push_back(cv::Point3f(1.8, -0.35, 0.105));
  points.push_back(cv::Point3f(1.8, -0.45, 0.105));
  points.push_back(cv::Point3f(1.8, -0.8, 0.105));
  points.push_back(cv::Point3f(1.8, -0.9, 0.105));
  points.push_back(cv::Point3f(1.8, 1.0, 0));
  points.push_back(cv::Point3f(1.8, 0.9, 0));

  for(unsigned int i = 0; i < points.size(); ++i)
    {
    std::cout << points[i] << std::endl;
    }

  return points;
}

// template <typename PREC>
// std::vector<cv::Point2f> CameraDetector<PREC>::Generate2DPoints()
// {
//   std::vector<cv::Point2f> points;

//   points.push_back(cv::Point2f(84.857, 216.255));
//   points.push_back(cv::Point2f(108.035, 215.645));
//   points.push_back(cv::Point2f(192.209, 216.864));
//   points.push_back(cv::Point2f(217.828, 216.864));
//   points.push_back(cv::Point2f(300.172, 218.694));
//   points.push_back(cv::Point2f(324.57, 218.084));
//   points.push_back(cv::Point2f(408.134, 218.084));
//   points.push_back(cv::Point2f(431.923, 219.304));
//   points.push_back(cv::Point2f(514.267, 219.304));
//   points.push_back(cv::Point2f(536.836, 219.304));
//   points.push_back(cv::Point2f(84.857, 240.043));
//   points.push_back(cv::Point2f(108.035, 240.653));
//   points.push_back(cv::Point2f(192.209, 241.873));
//   points.push_back(cv::Point2f(217.218, 242.483));
//   points.push_back(cv::Point2f(299.562, 242.483));
//   points.push_back(cv::Point2f(324.57, 242.483));
//   points.push_back(cv::Point2f(408.744, 243.703));
//   points.push_back(cv::Point2f(431.923, 244.313));
//   points.push_back(cv::Point2f(513.047, 244.923));
//   points.push_back(cv::Point2f(535.616, 243.093));

//   for(unsigned int i = 0; i < points.size(); ++i)
//     {
//     std::cout << points[i] << std::endl;
//     }

//   return points;
// }

// template <typename PREC>
// std::vector<cv::Point3f> CameraDetector<PREC>::Generate3DLidarPoints()
// {
//   std::vector<cv::Point3f> points;

//   points.push_back(cv::Point3f(-1.34259, -0.940092, 0.105));
//   points.push_back(cv::Point3f(-1.34395, -0.840092, 0.105));
//   points.push_back(cv::Point3f(-1.34139, -0.526456, 0.105));
//   points.push_back(cv::Point3f(-1.34416, -0.426456, 0.105));
//   points.push_back(cv::Point3f(-1.34638, -0.0840328, 0.105));
//   points.push_back(cv::Point3f(-1.3479, 0.0159672, 0.105));
//   points.push_back(cv::Point3f(-1.33974, 0.358982, 0.105));
//   points.push_back(cv::Point3f(-1.34321, 0.451296, 0.105));
//   points.push_back(cv::Point3f(-1.33782, 0.787527, 0.105));
//   points.push_back(cv::Point3f(-1.33728, 0.887527, 0.105));
//   points.push_back(cv::Point3f(-1.34259, -0.940092, 0));
//   points.push_back(cv::Point3f(-1.34395, -0.840092, 0));
//   points.push_back(cv::Point3f(-1.34139, -0.526456, 0));
//   points.push_back(cv::Point3f(-1.34416, -0.426456, 0));
//   points.push_back(cv::Point3f(-1.34638, -0.0840328, 0));
//   points.push_back(cv::Point3f(-1.3479, 0.0159672, 0));
//   points.push_back(cv::Point3f(-1.33974, 0.358982, 0));
//   points.push_back(cv::Point3f(-1.34321, 0.451296, 0));
//   points.push_back(cv::Point3f(-1.33782, 0.787527, 0));
//   points.push_back(cv::Point3f(-1.33728, 0.887527, 0));

//   for(unsigned int i = 0; i < points.size(); ++i)
//     {
//     std::cout << points[i] << std::endl;
//     }

//   return points;
// }

// template <typename PREC>
// std::vector<cv::Point3f> CameraDetector<PREC>::Generate3DVCSPoints()
// {
//   std::vector<cv::Point3f> points;

//   points.push_back(cv::Point3f(1.8, 1.0, 0.105));
//   points.push_back(cv::Point3f(1.8, 0.9, 0.105));
//   points.push_back(cv::Point3f(1.8, 0.55, 0.105));
//   points.push_back(cv::Point3f(1.8, 0.45, 0.105));
//   points.push_back(cv::Point3f(1.8, 0.1, 0.105));
//   points.push_back(cv::Point3f(1.8, 0.0, 0.105));
//   points.push_back(cv::Point3f(1.8, -0.35, 0.105));
//   points.push_back(cv::Point3f(1.8, -0.45, 0.105));
//   points.push_back(cv::Point3f(1.8, -0.8, 0.105));
//   points.push_back(cv::Point3f(1.8, -0.9, 0.105));
//   points.push_back(cv::Point3f(1.8, 1.0, 0));
//   points.push_back(cv::Point3f(1.8, 0.9, 0));
//   points.push_back(cv::Point3f(1.8, 0.55, 0));
//   points.push_back(cv::Point3f(1.8, 0.45, 0));
//   points.push_back(cv::Point3f(1.8, 0.1, 0));
//   points.push_back(cv::Point3f(1.8, 0.0, 0));
//   points.push_back(cv::Point3f(1.8, -0.35, 0));
//   points.push_back(cv::Point3f(1.8, -0.45, 0));
//   points.push_back(cv::Point3f(1.8, -0.8, 0));
//   points.push_back(cv::Point3f(1.8, -0.9, 0));

//   for(unsigned int i = 0; i < points.size(); ++i)
//     {
//     std::cout << points[i] << std::endl;
//     }

//   return points;
// }

template class CameraDetector<float>;
template class CameraDetector<double>;
} // namespace Xycar
