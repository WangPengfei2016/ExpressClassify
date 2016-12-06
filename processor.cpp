#include "processor.h"
#include <iostream>
#include <stdlib.h>

using namespace cv;
using namespace std;

void show(string name, Mat img)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img);
    int key =  cv::waitKey();
    if (key == 115)
	{
	    cv::imwrite("./phone.bmp", img);
	}
	else
	{
	    cv::destroyWindow(name);
	}
}

static bool comp(const vector<Point> key1, const vector<Point> key2)
{
    // 比较轮廓面积
    return contourArea(key1) > contourArea(key2);
}


static void filter_noise(Mat image)
{
    vector< vector<Point> > subContours;
	vector< vector<Point> >::iterator iterator;
    findContours(image.clone(), subContours, RETR_LIST, CHAIN_APPROX_SIMPLE);

	for (iterator = subContours.begin(); iterator != subContours.end(); iterator++)
	{
		cv::Rect rect = cv::boundingRect(*iterator);
		if (rect.area() < 30 || rect.width < 4)
		{
			Mat tmp(image, rect);
			tmp -= tmp;
		}

	}

}

static bool phone_classify(Mat region, Mat* outArrary)
{
	Rect line;
	int height = 0;
	for (int i = 0; i < region.rows; i++)
	{
		int nonZero = countNonZero(region.row(i));

		if (nonZero)
		{
			if (nonZero < region.cols*0.8)
			{
				height += 1;
			}
		   	else
		   	{
				height = 0;
			}
		}
		else if (!nonZero && height)
		{

			if (line.height < height)
			{
				line.y = i-height;
				line.width = region.cols;
				line.height = height;
			}
			height = 0;

		}

	}

	if (line.area() == 0)
	{
		line.width = region.cols;
		line.height = region.rows;
	}

	Mat mainLine(region, line);

	int width = 0;
	Mat vec = Mat::zeros(1, mainLine.cols, CV_8UC1);
	vector<Vec2s> chars;
	for (int i = 0; i < mainLine.cols; i++)
	{
		int nonZero = countNonZero(mainLine.col(i));

		vec.at<uchar>(0, i) = nonZero;

		if (nonZero)
		{
			width++;
		}
		else if (!nonZero && width)
		{
			if (width < 2)
			{
				vec.at<uchar>(0, i) = 0;
				width = 0;
			}
			else
			{
				chars.push_back(Vec2s(i-width, i-1));
				width = 0;
			}
		}
	}


	if (!chars.size())
	{
		return false;
	}

	int amonut = 0;
	vector<char> outText;
	int meanWidth = countNonZero(vec)/chars.size();
	for (vector<Vec2s>::iterator item = chars.begin(); item != chars.end(); item++)
	{

		short start = (*item)[0], end = (*item)[1];
		int width = end-start;

		Mat charImg = mainLine.colRange(start, end);

		if (cv::countNonZero(charImg) < 30 || width < 3)
		{
			continue;
		}

		if (width > meanWidth*1.5)
		{
			amonut += 2;
		}
		else
		{
			amonut++;
		}

	}

	if (amonut < 10 || amonut > 17)
	{
		return false;
	}

	mainLine.copyTo(*outArrary);
	return true;
}

Processor::Processor(const char *path)
{
    // 创建识别api
	api = new tesseract::TessBaseAPI();
	api->Init(path, "eng");                                    // 初始化识别数据路径和语言
	api->SetVariable("tessedit_char_blacklist", ":,\".-");     // 设置识别黑名单
	api->SetVariable("tessedit_char_whitelist", "1234567890"); // 设置识别白名单
	api->SetVariable("save_blob_choices", "T");
	api->SetPageSegMode(tesseract::PSM_SINGLE_WORD); // 设置识别模式为单行文本
}

Processor::~Processor()
{
    // 清除api防止内存泄漏
    api->Clear();
    api = NULL;
    delete api;
}

/**
 * 手机号提取函数
 * img 面单
 * rect 条码位置矩形
 * type 面单类型
**/
string Processor::extract_phone(std::string path, int width, int height)
{
    Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);

    if (!img.data)
    {
		cout<< "can not find any image !!!" <<endl;
		return "";
    }

    Size crop = Size(width, height);

    // 初始化变量
    Mat baup, thr, med, last;

    // 图片宽
    uint cols = img.cols;
    uint rows = img.rows;

    baup = img.clone();

    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    cv::filter2D(baup, baup, baup.depth(), kernel);

    cv::medianBlur(baup, baup, 3);
    adaptiveThreshold(baup, thr, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 20);

    Mat closing = getStructuringElement(MORPH_RECT, Size(14, 1));
    morphologyEx(thr, med, MORPH_CLOSE, closing);

    Mat opening = getStructuringElement(MORPH_RECT, Size(4, 4));
    morphologyEx(med, last, MORPH_OPEN, opening);

    // 获取所有轮廓，根据轮廓面积排序
    vector< vector<Point> > contours;
    findContours(last, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), comp);

    for (int i = 0; i < contours.size(); ++i)
    {
		int limit = crop.area();

		cv::RotatedRect rotatedRect = cv::minAreaRect(contours[i]);

		float angle = rotatedRect.angle;
		Size area = rotatedRect.size;
		Point center = rotatedRect.center;

		if (angle < -45.) {
			angle += 90.0;
			int tmp = area.width;
			area.width = area.height;
			area.height = tmp;
		}

		// 第一次粗过滤
		// 根据形状和面积过滤
		if (area.area() > limit*0.01 || area.area() < limit*0.001  || area.width < 5*area.height) {
			continue;
		}
		if (area.width > 300 || area.width < 50) {
			continue;
		}

		// 扩展切割区域，提高识别率
		if (center.x - area.width/2 > 4)
		{
			area.width += 8;
		}
		if (center.y - area.height/2 > 3)
		{
			area.height += 4;
		}

		// 在原始图片中取出手机号, 旋转
		Mat M, rotated, candidate_region;
		M = cv::getRotationMatrix2D(rotatedRect.center, angle, 1.0);
		cv::warpAffine(baup, rotated, M, baup.size(), cv::INTER_CUBIC);
		cv::getRectSubPix(rotated, area, rotatedRect.center, candidate_region);

		// 放大图像
		if (candidate_region.rows < 25 || candidate_region.cols < 120)
		{
			resize(candidate_region, candidate_region, Size(), 1.8, 1.8, INTER_LANCZOS4);
		}

		// 二值化，抑制干扰
		cv::adaptiveThreshold(candidate_region, candidate_region, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 13, 5);

		filter_noise(candidate_region);

		// 精细过滤
		cv::Mat numMat;
		if (!phone_classify(candidate_region, &numMat))
		{
			continue;
		}

		// 图片添加边框，提高识别率
		cv::copyMakeBorder(numMat, numMat, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

		// 开始识别手机号
		string num = recognize_num(numMat);
		if (num != "NO")
		{
			size_t pos = string::npos;
			pos = path.find(num.substr(0, 11));
			if (pos == string::npos) {
				return num;
			}
		}
    }

    // 识别失败
    return "";
}

string Processor::recognize_num(Mat image)
{
    /**
     * 手机号数字识别程序
     **/

    string outText = "";
    bool findPhone = false;
    vector<float> confidence;

    api->SetImage(image.data, image.size().width, image.size().height, image.channels(), (int)image.step1());
    api->Recognize(NULL);

    // 逐个符号识别，获取每一个Confidence
    tesseract::ResultIterator *ri = api->GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_SYMBOL;

    if (ri != 0)
    {
        do
        {
            float conf = ri->Confidence(level);
			if (conf < 65 && confidence.size() > 0 && confidence.back() < 65)
			{
				outText.clear();
				confidence.clear();
				continue;
			}
            confidence.push_back(conf);
            const char *symbol = ri->GetUTF8Text(level);
            if (symbol == 0 || string(symbol) == " ")
            {
                continue;
            }
            outText += symbol;
            delete[] symbol;
        } while (ri->Next(level));
    }

    // 根据手机号特征，判别识别结果是否为手机号
    size_t length = outText.length();

    if (length > 15 || length < 11)
    {
        return "NO";
    }

    size_t front;
    for (front = 0; front <= length - 11; front++)
    {
        if (outText[front] != '1')
        {
            continue;
        }

        if (outText[front + 1] == '3' || outText[front + 1] == '8')
        {
            findPhone = true;
            goto finally;
        }
        else if (outText[front + 1] == '5')
        {
            if (outText[front + 2] == '4')
                continue;

            findPhone = true;
            goto finally;
        }
        else if (outText[front + 1] == '4')
        {
            if (outText[front + 2] == '5' || outText[front + 2] == '7')
                findPhone = true;
            goto finally;
        }
		else if (outText[front + 1] == '7')
		{
			if (outText[front + 2] == '8' || outText[front + 2] == '6' || outText[front +2] == '0')
				findPhone = true;
			goto finally;
		}
    }

// 识别过程结束清理
finally:
    // 清空识别数据
    api->Clear();

    float average_confidence, total = 0;
    if (findPhone)
    {
        // 计算可信度总和
        for (size_t i = front; i < front + 11; i++)
        {
            total += confidence[i];
        }
    }
    // 平均可信度
    average_confidence = total / 11;
    if (average_confidence > 75)
    {
		string substr = outText.substr(front, 11);
		return substr+":"+to_string(average_confidence);
	}
    else
    {
        return "NO";
    }
}
