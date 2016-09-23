#ifndef PROCESSER_H
#define PROCESSER_H

#include <zbar.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <tesseract/baseapi.h>

using namespace std;
using namespace cv;


class Processor {
// 手机号处理类
// 返回手机号

public:
    // 初始化处理器
    Processor(const char *path);
    // 析构函数
    ~Processor();
    // 提取手机号
    string extract_phone(Mat img, Rect rect);

private:
    // tesseract 识别程序接口
    tesseract::TessBaseAPI *api;
    // 识别手机号程序
    string recognize_num(Mat image);
};


#endif //PROCESSER_H
