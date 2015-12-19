#pragma once

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

class Image
{
private:
	Mat image;
public:
	Image(string imageDir);
	Image();
	~Image();

	//get
	Mat getImage();

	//set
	void setImage(Mat image);
};

