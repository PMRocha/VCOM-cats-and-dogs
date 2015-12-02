// VCOMPagina1Exercicio4.cpp : Defines the entry point for the console application.
//

/*#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

/*using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	Mat capture;
	Mat channels[3];
	int psy = 0;

	VideoCapture cap(0); // open the video camera no. 0

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame size : " << dWidth << " x " << dHeight << endl;

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	namedWindow("GreyVideo", CV_WINDOW_AUTOSIZE);
	namedWindow("blue", CV_WINDOW_AUTOSIZE);

	while (1)
	{
		Mat frame, greyFrame,blueFrame;

		
		bool bSuccess = cap.read(frame); // read a new frame from video
		cvtColor(frame, greyFrame, CV_BGR2GRAY);

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}
		frame.copyTo(blueFrame);
		split(blueFrame, channels);

		channels[psy] = Mat::zeros(frame.rows, frame.cols, CV_8UC1);//Set blue channel to 0
		channels[(psy+1)/3] = Mat::zeros(frame.rows, frame.cols, CV_8UC1);//Set blue channel to 0

		merge(channels, 3, blueFrame);

		imshow("MyVideo", frame); //show the frame in "MyVideo" window
		imshow("blue", blueFrame);//grey capture
		imshow("GreyVideo", greyFrame);//grey capture
		if (psy == 2)
		{
			psy = 0;
		}
		else
		{
			psy++;
		}
		if (waitKey(1) == 27) //capture photo
		{
			cout << "test" << endl;
			frame.copyTo(capture);
			namedWindow("1", WINDOW_AUTOSIZE);// Create a window for display.
			imshow("1", capture);//red channel
		}

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	return 0;

}*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////






