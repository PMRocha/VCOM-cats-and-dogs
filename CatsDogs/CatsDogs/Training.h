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
	int filesNum, line, dictionarySize;
	Ptr<SVM> svm;
	Ptr<KNearest> knn;
	Ptr<NormalBayesClassifier> bayes;
public:
	Training(int filesNum, double dictionarySize);
	~Training();
	void initLabels();
	void setTrainingDataMat(Mat catDog);

	//SUPPORT VECTOR MACHINE
	void svmTrain();
	float svmTest(Mat desc);
	void svmSave(string fileName = "svm_train.yml");
	void svmLoad(string fileName = "svm_train.yml");

	//K NEAREST NEIGHBOURS
	void knnTrain();
	float knnTest(Mat desc);
	void knnSave(string fileName = "knn_train.yml");
	void knnLoad(string fileName = "knn_train.yml");

	//BAYES
	void bayesTrain();
	float bayesTest(Mat desc);
	void bayesSave(string fileName = "bayes_train.yml");
	void bayesLoad(string fileName = "bayes_train.yml");
};

