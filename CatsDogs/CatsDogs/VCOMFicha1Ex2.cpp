#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//#include <Gdiplus.h>

using namespace cv;
using namespace std;
/*
int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cout << "Width: " << image.size().width << endl;
	cout << "Height: " << image.size().height << endl;

	cv::Size size = image.size();
	Mat img2, img3;
	img2 = image;
	image.copyTo(img3);
	flip(img3, img2, 1);

	Mat img = Mat(50, 200, CV_8UC1, Scalar(100));
	img.at<uchar>(25, 100) = 255;

	/*namedWindow("1", WINDOW_AUTOSIZE);
	namedWindow("2", WINDOW_AUTOSIZE);
	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("1", image);   
	imshow("2", image); 
	imshow("Display window", image); // Show our image inside it.*/
	/*namedWindow("2", WINDOW_AUTOSIZE);
	imshow("2", img);

	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;
}*/