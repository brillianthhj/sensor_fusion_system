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
void CameraDetector<PREC>::boundingBox(const cv::Mat img)
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

				if (confidence > mConfThreshold) {
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

        cv::imshow("undistort_img", mTemp);
        cv::waitKey(1);
    }    
}

template class CameraDetector<float>;
template class CameraDetector<double>;
} // namespace Xycar
