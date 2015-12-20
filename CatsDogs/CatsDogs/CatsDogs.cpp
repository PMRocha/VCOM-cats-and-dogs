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

vector<Mat> train_descriptors;
vector<Mat> test_descriptors;
int imageCounter = 0;

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

void getFeatures(vector<Mat> &descriptors) {
	//verificar se o vocabulario existe (ficheiro)
	//positive images - dogs
	for (int i = 0; i < numFileTrain; i++) {
		Image dog = Image("../x64/Release/train/dog." + to_string(i) + ".jpg");
		//Image dog = Image("train/dog." + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(dog.getImage(), desc);
		descriptors.push_back(desc);
		printf("%d /25000 \n", imageCounter);
		imageCounter++;
	}

	//negative images - cats
	for (int i = 0; i < numFileTrain; i++) {
		Image cat = Image("../x64/Release/train/cat." + to_string(i) + ".jpg");
		//Image cat = Image("train/cat." + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(cat.getImage(), desc);
		descriptors.push_back(desc);
		printf("%d /25000 \n", imageCounter);
		imageCounter++;
	}
}

void startTraining(int option) {
	//initialize training
	Training train(2 * numFileTrain, 196);
	train.initLabels();
	for (int i = 0; i < train_descriptors.size(); i++) {
		train.setTrainingDataMat(train_descriptors[i]);
	}
	//choosing train algorithm
	switch (option) {
		case 1:
			train.svmTrain();
			train.svmSave();
			break;
		case 2:
			train.knnTrain();
			train.knnSave();
			break;
		case 3:
			break;
	}

	ofstream myfile;
	myfile.open("results.csv", ios::app);
	myfile << "id,label\n";

	imageCounter = 0;
	float res;
	for (int i = 1; i < numFileTest; i++) {
		Image catOrDog = Image("../x64/Release/test1/" + to_string(i) + ".jpg");
		//Image cat = Image("test1/" + to_string(i) + ".jpg");
		Mat desc;
		detectionSIFT(catOrDog.getImage(), desc);
		switch (option) {
		case 1:
			res = train.svmTest(desc);
			break;
		case 2:
			res = train.knnTest(desc);
			break;
		}
		myfile << i << "," << res << endl;
		printf("%d /12500 \n", imageCounter);
		imageCounter++;
		//imshow("catOrDog", catOrDog.getImage());
		//waitKey(0);
	}
	myfile.close();
}

void menu_trainOrLoad(int &opt, bool &train) {
	//false = train   true = load
	int option = 0;
	do {
		cout << "########################################" << endl;
		cout << "###### BEST TRAINING MACHINE EVER ######" << endl;
		cout << "########################################" << endl;
		cout << "1. Train" << endl;
		cout << "2. Load from YML" << endl;
		cout << "3. Back" << endl;
		cin >> option;

		switch (option) {
		case 1:
			getFeatures(train_descriptors);
			startTraining(opt);
			break;
		case 2:
			break;
		case 3:
			opt = 0;
			return;
		}
	} while (option == 0);
}

void menu() {
	int option = 0;
	bool train = true;
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
			menu_trainOrLoad(option, train);
			break;
		case 2:
			menu_trainOrLoad(option, train);
			break;
		case 3:
			exit(0);
		}
	} while (option == 0);
}

int main(){
	//identify dogs
	menu();
    return 0;
}

