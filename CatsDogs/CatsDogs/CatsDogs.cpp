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
	Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
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
	int biggerArea = 0;
	vector<Mat> descriptors;

	//positive images - dogs
	for (int i = 0; i < 10; i++) {
		Image dog = Image("../x64/Release/train/dog."+to_string(i)+".jpg");
		Mat desc;
		detectionSIFT(dog.getImage(), desc);
		descriptors.push_back(desc);
		getBiggerArea(desc.rows, desc.cols, biggerArea);
	}
	
	//negative images - cats
	for (int i = 0; i < 10; i++) {
		Image cat = Image("../x64/Release/train/cat." + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(cat.getImage(), desc);
		descriptors.push_back(desc);
		getBiggerArea(desc.rows, desc.cols, biggerArea);
	}

	Training train(20, biggerArea);
	train.initLabels();
	for (int i = 0; i < descriptors.size(); i++) {
		train.supportVectorMachine(descriptors[i]);
	}
	train.svmTrain();

    return 0;
}

