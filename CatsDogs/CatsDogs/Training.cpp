#include "stdafx.h"
#include "Training.h"
#include "Image.h"


Training::Training(int filesNum, double area){
	this->filesNum = filesNum;
	this->line = 0;
	trainingDataMat = Mat(filesNum, area, CV_32FC1);

	labels = Mat(filesNum, 0, CV_32SC1);
}

Training::~Training()
{
}

void Training::initLabels() {
	for (int i = 0; i < filesNum; i++) {
		if (filesNum/2<=i) {
			labels.push_back(1);
		}else {
			labels.push_back(0);
		}
	}

	for (int i = 0; i<labels.rows; i++)
		for (int j = 0; j<labels.cols; j++)
			printf("labels(%d, %d) = %d \n", i, j, labels.at<int>(i, j));
}

void Training::supportVectorMachine(Mat catDog) {
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
	printf("2");
	svm->train(trainingDataMat, ROW_SAMPLE, labels);
	printf("3");
	
	/*Mat res;   // output
	Image cat = Image("train/cat.0.jpg");
	svm->predict(cat.getImage(), res);
	imshow("output", res);
	waitKey(0);*/
}