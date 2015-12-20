#include "stdafx.h"
#include "Training.h"
#include "Image.h"
#include <opencv2/xfeatures2d.hpp>


Training::Training(int filesNum, double area){
	this->filesNum = filesNum;
	this->line = 0;
	trainingDataMat = Mat(filesNum, area, CV_32FC1);

	labels = Mat(filesNum, 0, CV_32SC1);
}

Training::~Training()
{
}

void Training::svmInitLabels() {
	for (int i = 0; i < filesNum; i++) {
		if (i<=filesNum/2) {
			labels.push_back(1);
		}else {
			labels.push_back(-1);
		}
	}
}

void Training::setTrainingDataMat(Mat catDog) {
	int ii = 0;
	for (int i = 0; i</*catDog.rows*/14; i++) {
		for (int j = 0; j < /*catDog.cols*/14; j++) {
			trainingDataMat.at<float>(line, ii) = catDog.at<uchar>(i, j);
			ii++;
		}
	}
	line++;
}

void Training::svmTrain() {
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::POLY);
	svm->setDegree(3);
	TermCriteria term_crit(CV_TERMCRIT_ITER, 100, 1e-6);
	svm->setTermCriteria(term_crit);

	// Train the SVM
	svm->train(trainingDataMat, ROW_SAMPLE, labels);
}

void Training::svmTest(Mat desc) {

	//just testing
	Mat descInLine = Mat(1, 196, CV_32FC1);
	int ii = 0;
	for (int i = 0; i < 14; i++) {
		for (int j = 0; j < 14; j++) {
			descInLine.at<float>(0, ii) = desc.at<uchar>(i, j);
			ii++;
		}
	}

	float res = svm->predict(descInLine);
	printf("res = %f\n", res);
}

void Training::svmSave(string fileName) {
	svm->save(fileName);
}

void Training::svmLoad(string fileName) {
	svm = StatModel::load<SVM>(fileName);
}