#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class Image
{
private:
	Mat image;
public:
	Image(string fileName);
	Image(int height, int width, int intensity);
	~Image();
};

