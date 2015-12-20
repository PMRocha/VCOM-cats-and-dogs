#include "stdafx.h"
#include "Image.h"
#include "Training.h"

#include <iostream>
#include <fstream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#define numFileTrain 20
#define numFileTest 20

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
	int option = 0;
	do {
		cout << "########################################" << endl;
		cout << "###### BEST TRAINING MACHINE EVER ######" << endl;
		cout << "########################################" << endl;
		cout << "1. Support Vector Machine" << endl;
		cout << "2. K Nearest Neighbours" << endl;
		cout << "3. Exit" << endl;
		cin >> option;

		switch (option) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			return 0;
		}
	} while (option == 0);

	//identify dogs
	vector<Mat> train_descriptors;
	vector<Mat> test_descriptors;
	int imageCounter = 0;

	ofstream myfile;
	myfile.open("results.csv",ios::app);
	myfile << "id,label\n";

	//positive images - dogs
	for (int i = 0; i < numFileTrain; i++) {
		Image dog = Image("../x64/Release/train/dog."+to_string(i)+".jpg");
		//Image dog = Image("train/dog." + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(dog.getImage(), desc);
		train_descriptors.push_back(desc);
		printf("%d /25000 \n", imageCounter);
		imageCounter++;
	}
	
	//negative images - cats
	for (int i = 0; i < numFileTrain; i++) {
		Image cat = Image("../x64/Release/train/cat." + to_string(i) + ".jpg");
		//Image cat = Image("train/cat." + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(cat.getImage(), desc);
		train_descriptors.push_back(desc);
		printf("%d /25000 \n", imageCounter);
		imageCounter++;
	}

	Training train(2* numFileTrain, 196);
	train.svmInitLabels();
	for (int i = 0; i < train_descriptors.size(); i++) {
		train.setTrainingDataMat(train_descriptors[i]);
	}
	//train.svmTrain();
	train.knnTrain();
	//train.svmSave();
	train.knnSave();
	imageCounter=0;
	float res;
	for (int i = 1; i < numFileTest; i++) {
		Image catOrDog = Image("../x64/Release/test1/" + to_string(i) + ".jpg");
		//Image cat = Image("test1/" + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(catOrDog.getImage(), desc);
		//res = train.svmTest(desc);
		res = train.knnTest(desc);
		myfile << i << "," << res << endl;
		printf("%d /12500 \n", imageCounter);
		imageCounter++;
		//imshow("catOrDog", catOrDog.getImage());
		//waitKey(0);
	}
	myfile.close();
    return 0;
}

