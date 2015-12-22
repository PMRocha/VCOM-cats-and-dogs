#include "stdafx.h"
#include "Training.h"
#include "Image.h"
#include <opencv2/xfeatures2d.hpp>


Training::Training(int filesNum, double dictionarySize){
	this->filesNum = filesNum;
	this->dictionarySize = dictionarySize;
	this->line = 0;
	trainingDataMat = Mat(0, dictionarySize, CV_32FC1);

	labels = Mat(filesNum, 0, CV_32SC1);
}

Training::~Training()
{
}

void Training::initLabels() {
	for (int i = 0; i < filesNum; i++) {
		if (i<=filesNum/2) {
			labels.push_back(1);
		}else {
			labels.push_back(0);
		}
	}
}

void Training::setTrainingDataMat(Mat catDog) {
	/*int ii = 0;
	for (int i = 0; i<14; i++) {
		for (int j = 0; j < 14; j++) {
			trainingDataMat.at<float>(line, ii) = catDog.at<uchar>(i, j);
			ii++;
		}
	}
	line++;*/

	trainingDataMat.push_back(catDog);
}

void Training::svmTrain() {
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	//svm->setDegree(3);
	TermCriteria term_crit(CV_TERMCRIT_ITER, 100, 1e-6);
	svm->setTermCriteria(term_crit);

	// Train the SVM
	//svm->train(trainingDataMat, ROW_SAMPLE, labels);
	Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labels);
	svm->trainAuto(trainData);
}

float Training::svmTest(Mat desc) {

	//just testing
	Mat descInLine = Mat(1, dictionarySize, CV_32FC1);
	int ii = 0;
	for (int i = 0; i < desc.rows; i++) {
		for (int j = 0; j < desc.cols; j++) {
			if (ii <= dictionarySize) {
				descInLine.at<float>(0, ii) = desc.at<uchar>(i, j);
				ii++;
			}
		}
	}
	return svm->predict(descInLine);
}

void Training::svmSave(string fileName) {
	svm->save(fileName);
}

void Training::svmLoad(string fileName) {
	svm = StatModel::load<SVM>(fileName);
}

void Training::knnTrain() {
	knn = KNearest::create();
	knn->setDefaultK(5);
	knn->setIsClassifier(true);
	knn->train(trainingDataMat, ROW_SAMPLE, labels);
}

float Training::knnTest(Mat desc) {
	Mat descInLine = Mat(1, dictionarySize, CV_32FC1);
	int ii = 0;
	for (int i = 0; i < desc.rows; i++) {
		for (int j = 0; j < desc.cols; j++) {
			if (ii <= dictionarySize) {
				descInLine.at<float>(0, ii) = desc.at<uchar>(i, j);
				ii++;
			}
		}
	}

	Mat res(0, 0, CV_32F);
	knn->findNearest(descInLine, knn->getDefaultK(), res);
	return res.at<float>(0,0);
}

void Training::knnSave(string fileName) {
	knn->save(fileName);
}

void Training::knnLoad(string fileName) {
	knn = StatModel::load<KNearest>(fileName);
}


void Training::bayesTrain() {
	bayes = NormalBayesClassifier::create();
	bayes->train(trainingDataMat, ROW_SAMPLE, labels);
}

float Training::bayesTest(Mat desc) {
	Mat descInLine = Mat(1, dictionarySize, CV_32FC1);
	int ii = 0;
	for (int i = 0; i < desc.rows; i++) {
		for (int j = 0; j < desc.cols; j++) {
			if (ii <= dictionarySize) {
				descInLine.at<float>(0, ii) = desc.at<uchar>(i, j);
				ii++;
			}
		}
	}

	return bayes->predict(descInLine);
}

void Training::bayesSave(string fileName) {
	bayes->save(fileName);
}

void Training::bayesLoad(string fileName) {
	bayes = StatModel::load<NormalBayesClassifier>(fileName);
}