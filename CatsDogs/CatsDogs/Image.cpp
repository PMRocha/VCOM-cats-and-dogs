#include "Image.h"



Image::Image(string fileName)
{
	
}

Image::Image(int height, int width, int intensity)
{
	image = Mat(50, 200, CV_8UC1, Scalar(100));
}

Image::~Image()
{
}