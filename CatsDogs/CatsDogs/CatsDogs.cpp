#include "stdafx.h"
#include "Image.h"
#include "Training.h"

#include <iostream>
#include <fstream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#define numFileTrain 100
#define numFileTest 12501
#define dictionarySize 196

using namespace cv;

Mat train_descriptors;
TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;
Mat dictionary;

vector<Mat> test_descriptors;
vector<Mat> bow_descriptors;
Training train = Training(2 * numFileTrain, dictionarySize);
int imageCounter = 1;

void detectionSIFT(Mat img, Mat &desc) {
	vector<vector<KeyPoint>>keypoints = vector<vector<KeyPoint>>();

	//detect the keypoints of hand cards using SIFT Detector
	Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
	vector<KeyPoint> img_keypoints;
	detector->detect(img, img_keypoints);

	//calculate descriptors (feature vectors)
	Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create(/*196*/dictionarySize);
	extractor->compute(img, img_keypoints, desc);
}

void getFeatures() {
	//verificar se o vocabulario existe (ficheiro)
	//positive images - dogs
	for (int i = 0; i < numFileTrain; i++) {
		Image dog = Image("../x64/Release/train/dog." + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		//Image dog = Image("train/dog." + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat desc;
		detectionSIFT(dog.getImage(), desc);
		train_descriptors.push_back(desc);
		printf("%d/%d \n", imageCounter, numFileTrain*2);
		imageCounter++;
	}

	//negative images - cats
	for (int i = 0; i < numFileTrain; i++) {
		Image cat = Image("../x64/Release/train/cat." + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		//Image cat = Image("train/cat." + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat desc;
		detectionSIFT(cat.getImage(), desc);
		train_descriptors.push_back(desc);
		printf("%d/%d \n", imageCounter, numFileTrain*2);
		imageCounter++;
	}
}

void getBoFdescriptor() {
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
	Ptr<DescriptorExtractor> extractor = xfeatures2d::SIFT::create();
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	bowDE.setVocabulary(dictionary);

	imageCounter = 1;
	for (int i = 0; i < numFileTrain; i++) {
		Image dog = Image("../x64/Release/train/dog." + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		//Image dog = Image("train/dog." + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		vector<KeyPoint> keypoints;
		detector->detect(dog.getImage(), keypoints);
		Mat bowDescriptor;
		bowDE.compute(dog.getImage(), keypoints, bowDescriptor); 
		train.setTrainingDataMat(bowDescriptor);
		printf("%d/%d \n", imageCounter, numFileTrain*2);
		imageCounter++;
	}

	//negative images - cats
	for (int i = 0; i < numFileTrain; i++) {
		Image cat = Image("../x64/Release/train/cat." + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		//Image cat = Image("train/cat." + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		vector<KeyPoint> keypoints;
		detector->detect(cat.getImage(), keypoints);
		Mat bowDescriptor;
		bowDE.compute(cat.getImage(), keypoints, bowDescriptor);
		train.setTrainingDataMat(bowDescriptor);
		printf("%d/%d \n", imageCounter, numFileTrain*2);
		imageCounter++;
	}
}

void constructBagOfWords() {
	FileStorage fs("dictionary.yml", FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "1st SIFT" << endl;
		getFeatures();
		cout << "BOW creation" << endl;
		//Create the BoW (or BoF) trainer
		BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
		//cluster the feature vectors
		dictionary = bowTrainer.cluster(train_descriptors);
		//store the vocabulary
		FileStorage fs("dictionary.yml", FileStorage::WRITE);
		fs << "vocabulary" << dictionary;
		fs.release();
	} else {
		cout << "loading file" << endl;
		fs["vocabulary"] >> dictionary;
		fs.release();
	}
	cout << "2nd SIFT" << endl;
	getBoFdescriptor();
}

void startTraining(int option) {
	//initialize training
	train.initLabels();
	/*for (int i = 0; i < bow_descriptors.size(); i++) {
		train.setTrainingDataMat(bow_descriptors[i]);
	}*/
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
			train.bayesTrain();
			train.bayesSave();
			break;
	}
}

void testing(int option) {
	printf("Testing \n");
	ofstream myfile;
	myfile.open("results.csv", ios::trunc);
	myfile << "id,label\n";

	imageCounter = 1;
	float res;
	for (int i = 1; i < numFileTest; i++) {
		Image catOrDog = Image("../x64/Release/test1/" + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		//Image cat = Image("test1/" + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat desc;
		detectionSIFT(catOrDog.getImage(), desc);
		switch (option) {
		case 1:
			res = train.svmTest(desc);
			break;
		case 2:
			res = train.knnTest(desc);
			break;
		case 3:
			res = train.bayesTest(desc);
			break;
		}
		myfile << i << "," << res << endl;
		printf("%d/%d \n", imageCounter, numFileTest);
		imageCounter++;
	}
	myfile.close();
}

void menu_trainOrLoad(int &opt, bool &load) {
	//false = train   true = load
	int option = 0;
	cout << "########################################" << endl;
	cout << "###### BEST TRAINING MACHINE EVER ######" << endl;
	cout << "########################################" << endl;
	cout << "1. Train" << endl;
	cout << "2. Load from YML" << endl;
	cout << "3. Back" << endl;
	cin >> option;

	switch (option) {
	case 1:
		constructBagOfWords();
		startTraining(opt);
		break;
	case 2:
		if (opt == 1) {//svm
			train.svmLoad();
		} else if (opt == 2) {//knn
			train.knnLoad();
		} else if (opt == 3) {//bayes
			train.bayesLoad();
		}
		break;
	case 3:
		opt = 0;
		return;
	}
	testing(opt);
}

void menu() {
	int option = 0;
	bool load = false;
	do {
		cout << "########################################" << endl;
		cout << "###### BEST TRAINING MACHINE EVER ######" << endl;
		cout << "########################################" << endl;
		cout << "1. Support Vector Machine" << endl;
		cout << "2. K Nearest Neighbours" << endl;
		cout << "3. Bayes Classifier" << endl;
		cout << "4. Exit" << endl;
		cin >> option;

		if (option == 4) {
			exit(0);
		} else if (option < 4) {
			menu_trainOrLoad(option, load);
		}else {
			option = 0;
		}
	} while (option == 0);
}

int main(){
	//identify dogs
	menu();
    return 0;
}

