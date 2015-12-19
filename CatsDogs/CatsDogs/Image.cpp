#include "stdafx.h"
#include "Image.h"


Image::Image(string imageDir) {
	image = imread(imageDir, 1);
}

Image::Image() {
	image = NULL;
}


Image::~Image()
{
}

Mat Image::getImage() {
	return image;
}

void Image::setImage(Mat image) {
	this->image = image;
}
