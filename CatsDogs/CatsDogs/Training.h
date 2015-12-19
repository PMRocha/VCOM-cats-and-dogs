#pragma once

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

class Training
{
private:
	Mat trainingDataMat;
	Mat labels;
	int filesNum, line;
	Ptr<SVM> svm;
public:
	Training(int filesNum, double area);
	~Training();

	//SUPPORT VECTOR MACHINE
	void initLabels();
	void supportVectorMachine(Mat catDog);
	void svmTrain();
};

