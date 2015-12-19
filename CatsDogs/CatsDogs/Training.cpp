#include "stdafx.h"
#include "Training.h"


Training::Training(int filesNum, double area){
	this->filesNum = filesNum;
	this->line = 0;
	trainingDataMat = Mat(filesNum, area, CV_32FC1);
	labels = Mat(filesNum, 1, CV_32FC1);
}

Training::~Training()
{
}

void Training::initLabels() {
	for (int i = 0; i < filesNum; i++) {
		if (filesNum/2<=i) {
			labels.at<float>(i, 1) = 1;
		}else {
			labels.at<float>(i, 1) = -1;
		}
	}
}

void Training::supportVectorMachine(Mat catDog) {
	for (int i = 0; i<catDog.rows; i++) {
		for (int j = 0; j < catDog.cols; j++) {
			trainingDataMat.at<float>(line, i+j) = catDog.at<uchar>(i, j);
		}
	}
	line++;
}

void Training::svmTrain() {
	// Set up SVM's parameters
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	TermCriteria term_crit(CV_TERMCRIT_ITER, 100, 1e-6);
	svm->setTermCriteria(term_crit);

	// Train the SVM
	printf("2");
	svm->train(trainingDataMat, ROW_SAMPLE, labels);
	printf("3");

	//Mat query; // input, 1channel, 1 row (apply reshape(1,1) if nessecary)
	//Mat res;   // output
	//svm->predict(query, res);
}
