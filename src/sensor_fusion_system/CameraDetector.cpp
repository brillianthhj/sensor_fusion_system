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
    for (const auto& row : config["CAMERA"]["CAMERA_MATRIX2"]) {
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
    for (const auto& row : config["CAMERA"]["DIST_COEFF2"]) {
        distMatrixData.emplace_back(row.as<float>());
    }
    mDistCoeffs = cv::Mat(distMatrixData, true);

    mYoloConfig = config["YOLO"]["CONFIG"].as<std::string>();
    mYoloModel = config["YOLO"]["MODEL"].as<std::string>();
    mYoloLabel = config["YOLO"]["LABEL"].as<std::string>();

    mDebugging = config["DEBUG"].as<bool>();

    mRvec = cv::Mat(3, 1, cv::DataType<double>::type);
    mTvec = cv::Mat(3, 1, cv::DataType<double>::type);
}

template <typename PREC>
void CameraDetector<PREC>::undistortAndDNNConfig()
{
    cv::initUndistortRectifyMap(mCameraMatrix, mDistCoeffs, cv::Mat(), mCameraMatrix, mImageSize, CV_32FC1, mMap1, mMap2);

    mNeuralNet = cv::dnn::readNetFromDarknet(mYoloConfig, mYoloModel);
    // mNeuralNet = cv::dnn::readNetFromONNX(mYoloModel);

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
void CameraDetector<PREC>::boundingBox(const cv::Mat img, const std::vector<cv::Point2f> lidarImagePoints)
{
    if (img.empty()) {
        // std::cerr << "No image.. Wait.." << std::endl;
    }
    else {
        // undistort image
        mTemp = img.clone();
        cv::remap(img, mTemp, mMap1, mMap2, cv::INTER_LINEAR);
        
        // Convert Mat to batch of images
		cv::Mat blob = cv::dnn::blobFromImage(mTemp, 1 / 255.f, cv::Size(416, 416), cv::Scalar(), true);

		// Set the network input
		mNeuralNet.setInput(blob);

		// compute output
		std::vector<cv::Mat> outs;
		mNeuralNet.forward(outs, mOutputLayers);

		std::vector<double> layersTimings;
		double time_ms = mNeuralNet.getPerfProfile(layersTimings) * 1000 / cv::getTickFrequency();
		putText(mTemp, cv::format("FPS: %.2f ; time: %.2f ms", 1000.f / time_ms, time_ms),
			cv::Point(20, 30), 0, 0.75, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;

		for (auto& out : outs) {
			float* data = (float*)out.data;
			for (int j = 0; j < out.rows; ++j, data += out.cols) {
				cv::Mat scores = out.row(j).colRange(5, out.cols);
				double confidence;
				cv::Point classIdPoint;

				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

				if (confidence > mConfThreshold && classIdPoint.x == 4) {
					int cx = static_cast<int>(data[0] * mTemp.cols);
					int cy = static_cast<int>(data[1] * mTemp.rows);
					int bw = static_cast<int>(data[2] * mTemp.cols);
					int bh = static_cast<int>(data[3] * mTemp.rows);
					int sx = cx - bw / 2;
					int sy = cy - bh / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(cv::Rect(sx, sy, bw, bh));
				}
			}
		}

		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, confidences, mConfThreshold, mNmsThreshold, indices);

		for (size_t i = 0; i < indices.size(); ++i) {
			int idx = indices[i];
			int sx = boxes[idx].x;
			int sy = boxes[idx].y;

			rectangle(mTemp, boxes[idx], cv::Scalar(0, 255, 0));

			std::string label = cv::format("%.2f", confidences[idx]);
			label = mClassNames[classIds[idx]] + ":" + label;
			int baseLine = 0;
			cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			rectangle(mTemp, cv::Rect(sx, sy, labelSize.width, labelSize.height + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
			putText(mTemp, label, cv::Point(sx, sy + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1, cv::LINE_AA);
		}

        for (size_t i = 0; i < lidarImagePoints.size(); ++i) {
            int u = lidarImagePoints[i].x;
            int v = lidarImagePoints[i].y;

            circle(mTemp, cv::Point(u, v), 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }

    cv::imshow("undistort_img", mTemp);
    cv::waitKey(1);
    }
}

template <typename PREC>
void CameraDetector<PREC>::getExtrinsicMatrix(std::vector<cv::Point2f> imagePoints, std::vector<cv::Point3f> objectPoints){

    cv::solvePnP(objectPoints, imagePoints, mCameraMatrix, mDistCoeffs, mRvec, mTvec);

    // extrinsic matrix
    cv::Mat R;
    cv::Rodrigues(mRvec, R);

    mExtrinsicMatrix = cv::Mat::zeros(4, 4, R.type());
    R.copyTo(mExtrinsicMatrix(cv::Rect(0, 0, 3, 3)));
    mTvec.copyTo(mExtrinsicMatrix(cv::Rect(3, 0, 1, 3)));
    mExtrinsicMatrix.at<double>(3, 3) = 1.0;

    // std::cout << "There are " << imagePoints.size() << " imagePoints and " << objectPoints.size() << " objectPoints." << std::endl;
    // std::cout << "Initial cameraMatrix: " << mCameraMatrix << std::endl;
    // std::cout << "Initial distCoeffs: " << mDistCoeffs << std::endl;

    // std::cout << "rvec: " << mRvec << std::endl;
    // std::cout << "tvec: " << mTvec << std::endl;
    // std::cout << "extrinsic matrix: " << mExtrinsicMatrix << std::endl;
}

template <typename PREC>
std::vector<cv::Point2f> CameraDetector<PREC>::getProjectPoints(std::vector<cv::Point3f> objectPoints){
    std::vector<cv::Point2f> points;
    cv::projectPoints(objectPoints, mRvec, mTvec, mCameraMatrix, mDistCoeffs, points);

    std::vector<cv::Point2f> filteredPoints;
    for (int i=0; i<points.size(); ++i) {
        double x = points[i].x;
        double y = points[i].y;

        if (x > 0 && x < 640 && y > 0 && y < 480) {
            filteredPoints.push_back(cv::Point2f(x, y));
        }
    }

    return filteredPoints;
}


template <typename PREC>
void CameraDetector<PREC>::filterLidarFromBbox(std::vector<cv::Point3f> objectPoints, std::vector<cv::Rect> boxes){
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(objectPoints, mRvec, mTvec, mCameraMatrix, mDistCoeffs, projectedPoints);

    std::vector<cv::Point3f> filteredObjectPoints;

    for (const auto& box: boxes) {
        for (const auto& point: objectPoints) {
            if (point.x > box.x && point.x < box.x + box.width && point.y > box.y && point.y < box.y + box.height) {
                filteredObjectPoints.push_back(point);
            }
        }
    }

}

template <typename PREC>
std::vector<cv::Point2f> CameraDetector<PREC>::Generate2DPoints()
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
  points.push_back(cv::Point2f(192.209, 241.873));
  points.push_back(cv::Point2f(217.218, 242.483));
  points.push_back(cv::Point2f(299.562, 242.483));
  points.push_back(cv::Point2f(324.57, 242.483));
  points.push_back(cv::Point2f(408.744, 243.703));
  points.push_back(cv::Point2f(431.923, 244.313));
  points.push_back(cv::Point2f(513.047, 244.923));
  points.push_back(cv::Point2f(535.616, 243.093));

  for(unsigned int i = 0; i < points.size(); ++i)
    {
    std::cout << points[i] << std::endl;
    }

  return points;
}

template <typename PREC>
std::vector<cv::Point3f> CameraDetector<PREC>::Generate3DLidarPoints()
{
  std::vector<cv::Point3f> points;

  points.push_back(cv::Point3f(-0.940092, -0.105, 1.34259));
  points.push_back(cv::Point3f(-0.840092, -0.105, 1.34395));
  points.push_back(cv::Point3f(-0.526456, -0.105, 1.34139));
  points.push_back(cv::Point3f(-0.426456, -0.105, 1.34416));
  points.push_back(cv::Point3f(-0.0840328, -0.105, 1.346385));
  points.push_back(cv::Point3f(0.0159672, -0.105, 1.3479));
  points.push_back(cv::Point3f(0.358982, -0.105, 1.33974));
  points.push_back(cv::Point3f(0.451296, -0.105, 1.34321));
  points.push_back(cv::Point3f(0.787527, -0.105, 1.33782));
  points.push_back(cv::Point3f(0.887527, -0.105, 1.33728));
  points.push_back(cv::Point3f(-0.940092, 0, 1.34259));
  points.push_back(cv::Point3f(-0.840092, 0, 1.34395));
  points.push_back(cv::Point3f(-0.526456, 0, 1.34139));
  points.push_back(cv::Point3f(-0.426456, 0, 1.34416));
  points.push_back(cv::Point3f(-0.0840328, 0, 1.346385));
  points.push_back(cv::Point3f(0.0159672, 0, 1.3479));
  points.push_back(cv::Point3f(0.358982, 0, 1.33974));
  points.push_back(cv::Point3f(0.451296, 0, 1.34321));
  points.push_back(cv::Point3f(0.787527, 0, 1.33782));
  points.push_back(cv::Point3f(0.887527, 0, 1.33728));

  for(unsigned int i = 0; i < points.size(); ++i)
    {
    std::cout << points[i] << std::endl;
    }

  return points;
}

template <typename PREC>
std::vector<cv::Point3f> CameraDetector<PREC>::Generate3DVCSPoints()
{
  std::vector<cv::Point3f> points;

  points.push_back(cv::Point3f(-1.0, -0.105, 1.8));
  points.push_back(cv::Point3f(-0.9, -0.105, 1.8));
  points.push_back(cv::Point3f(-0.55, -0.105, 1.8));
  points.push_back(cv::Point3f(-0.45, -0.105, 1.8));
  points.push_back(cv::Point3f(-0.1, -0.105, 1.8));
  points.push_back(cv::Point3f(-0.0, -0.105, 1.8));
  points.push_back(cv::Point3f(0.35, -0.105, 1.8));
  points.push_back(cv::Point3f(0.45, -0.105, 1.8));
  points.push_back(cv::Point3f(0.8, -0.105, 1.8));
  points.push_back(cv::Point3f(0.9, -0.105, 1.8));
  points.push_back(cv::Point3f(-1.0, 0, 1.8));
  points.push_back(cv::Point3f(-0.9, 0, 1.8));
  points.push_back(cv::Point3f(-0.55, 0, 1.8));
  points.push_back(cv::Point3f(-0.45, 0, 1.8));
  points.push_back(cv::Point3f(-0.1, 0, 1.8));
  points.push_back(cv::Point3f(0.0, 0, 1.8));
  points.push_back(cv::Point3f(0.35, 0, 1.8));
  points.push_back(cv::Point3f(0.45, 0, 1.8));
  points.push_back(cv::Point3f(0.8, 0, 1.8));
  points.push_back(cv::Point3f(0.9, 0, 1.8));

  for(unsigned int i = 0; i < points.size(); ++i)
    {
    std::cout << points[i] << std::endl;
    }

  return points;
}

template class CameraDetector<float>;
template class CameraDetector<double>;
} // namespace Xycar
