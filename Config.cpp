#include "Config.h"

HorizonalExpress::HorizonalExpress(int width) {
    this->width = width;
}

bool HorizonalExpress::filter_region_by_shape(cv::Mat img) {
    if (img.cols > width/3 || img.cols < width/10)
            return false;
    if (img.cols > img.rows*12 || img.cols < img.rows*5)
            return false;
    return true;
}

VerticalExpress::VerticalExpress(int width) {
    this->width = width;
}

bool VerticalExpress::filter_region_by_shape(cv::Mat img) {
    if (img.cols > width/3 || img.cols < width/10)
            return false;
    if (img.cols > img.rows*12 || img.cols < img.rows*5)
            return false;
    return true;
}