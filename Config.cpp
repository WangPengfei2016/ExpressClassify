#include "config.h"

TempleteOne::TempleteOne(cv::Rect bar) {
	this->bar.x = bar.x;
	this->bar.y = bar.y;
	this->bar.width = bar.width;
	this->bar.height = bar.height;
}

cv::Mat TempleteOne::cropy_region(cv::Mat region) {
	cv::Rect phone;
	phone.x = bar.x - bar.x/2;
	phone.y = bar.y - bar.height*5/2;

	if (phone.y < 0) {
		return region;
	}

	phone.width = region.cols-phone.x;
	phone.height = bar.height*1.5;
	return region(phone).clone();
}

TempleteTwo::TempleteTwo(cv::Rect bar) {
	this->bar.x = bar.x;
	this->bar.y = bar.y;
	this->bar.width = bar.width;
	this->bar.height = bar.height;
}

cv::Mat TempleteTwo::cropy_region(cv::Mat region) {
	cv::Rect phone;
	phone.x = bar.x;
	phone.y = bar.y + bar.height*3;
	phone.width = region.cols-phone.x;
	phone.height = bar.height;
	return region(phone).clone();
}


