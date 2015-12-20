#include "stdafx.h"
#include "Image.h"
#include "Training.h"

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;

void detectionSIFT(Mat img, Mat &desc) {
	vector<vector<KeyPoint>>keypoints = vector<vector<KeyPoint>>();

	//detect the keypoints of hand cards using SIFT Detector
	Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
	vector<KeyPoint> img_keypoints;
	detector->detect(img, img_keypoints);

	//calculate descriptors (feature vectors)
	Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create(196);
	extractor->compute(img, img_keypoints, desc);
}

void getBiggerArea(int line, int cols, int &biggerArea) {
	int area = line*cols;
	if (area>biggerArea) {
		biggerArea = area;
	}
}

int main(){
	//identify dogs
	vector<Mat> train_descriptors;
	vector<Mat> test_descriptors;

	//positive images - dogs
	for (int i = 0; i < 20; i++) {
		Image dog = Image("../x64/Release/train/dog."+to_string(i)+".jpg");
		//Image dog = Image("train/dog." + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(dog.getImage(), desc);
		train_descriptors.push_back(desc);
	}
	
	//negative images - cats
	for (int i = 0; i < 20; i++) {
		Image cat = Image("../x64/Release/train/cat." + to_string(i) + ".jpg");
		//Image cat = Image("train/cat." + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(cat.getImage(), desc);
		train_descriptors.push_back(desc);
	}

	Training train(40, 196);
	train.svmInitLabels();
	for (int i = 0; i < train_descriptors.size(); i++) {
		train.setTrainingDataMat(train_descriptors[i]);
	}
	train.svmTrain();
	train.svmSave();
	for (int i = 1; i < 50; i++) {
		Image catOrDog = Image("../x64/Release/test1/" + to_string(i) + ".jpg");
		//Image cat = Image("test1/" + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(catOrDog.getImage(), desc);
		train.svmTest(desc);
		imshow("catOrDog", catOrDog.getImage());
		waitKey(0);
	}

    return 0;
}

