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
	Ptr<KNearest> knn;
public:
	Training(int filesNum, double area);
	~Training();
	void initLabels();

	//SUPPORT VECTOR MACHINE
	void setTrainingDataMat(Mat catDog);
	void svmTrain();
	float svmTest(Mat desc);
	void svmSave(string fileName = "svm_train.yml");
	void svmLoad(string fileName = "svm_train.yml");

	//K NEAREST NEIGHBOURS
	void knnTrain();
	float knnTest(Mat desc);
	void knnSave(string fileName = "knn_train.yml");
	void knnLoad(string fileName = "knn_train.yml");
};

