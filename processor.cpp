#include "processor.h"
#include <algorithm>
#include <ctime>
#include <iostream>
#include <stack>
#include <stdlib.h>

using namespace cv;
using namespace std;

void saveimg(Mat img)
{
    Mat saveimg;
    resize(img.clone(), saveimg, Size(10, 16), 0, 0, INTER_CUBIC);
    threshold(saveimg, saveimg, 0, 255, THRESH_OTSU | THRESH_BINARY);
    time_t result = time(nullptr);
    string path = "/Users/lambda/Project/reglib/num/" + to_string(result) + ".bmp";
    imwrite(path, saveimg);
}

void show(string name, Mat img)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img);
    char key = cv::waitKey();
    if (key >= '0' && key <= '9')
    {
		saveimg(img);
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

/**
 * 文本锐化，抑制模糊
**/
static Mat usm(Mat imgSrc)
{
    //double sigma = 3;
    //int threshold = 0;
    //float amount = 1.0;
    //Mat imgBlurred, imgDst;
    //GaussianBlur(imgSrc, imgBlurred, Size(), sigma, sigma);
    //Mat lowContrastMask = abs(imgSrc - imgBlurred) < threshold;
    //imgDst = imgSrc * (1 + amount) + imgBlurred * (-amount);
    //imgSrc.copyTo(imgDst, lowContrastMask);
    //return imgDst;
	Mat imgEnhance;
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	filter2D(imgSrc, imgEnhance, imgSrc.depth(), kernel);
	return imgEnhance;
}

/**
 * 根据轮廓面积关系过滤不符合要求的轮廓
**/
static void filterNoise(Mat image)
{
    int cols = image.cols;
    int rows = image.rows;
    vector< vector<Point> > subContours;
    vector< vector<Point> >::iterator iterator;
    findContours(image.clone(), subContours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    for (iterator = subContours.begin(); iterator != subContours.end(); iterator++)
    {
		cv::Rect rect = cv::boundingRect(*iterator);
		if (rect.area() < cols * cols / 150 && rect.width < cols / 50 && rect.height < rows / 6)
		{
			Mat tmp(image, rect);
			tmp -= tmp;
		}
    }
}

static short searchMinimum(short rec[], short size)
{
    short pos = 0, value = 50;
    for (int i = 1; i < size - 1; i++)
    {
		if (rec[i] < rec[i - 1] && rec[i] <= rec[i + 1])
		{
			if (value > rec[i])
			{
				pos = i;
				value = rec[i];
			}
		}
    }

    return pos;
}

static list<Mat> phone_classify(Mat region)
{
    list<Mat> blockList;
    Rect line;
    int height = 0;
    for (int i = 0; i < region.rows; i++)
    {
		int nonZero = countNonZero(region.row(i));

		if (nonZero)
		{
			if (nonZero < region.cols * 0.9)
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
				line.y = i - height;
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

    if (line.height < 8)
    {
		blockList.clear();
		return blockList;
    }

    Mat mainLine(region, line);
    int width = 0;
    list<Range> chars;
    short *cols = new short[line.width];

    for (int i = 0; i < line.width - 1; i++)
    {
		int nonZero = countNonZero(mainLine.col(i));
		int nextNonZero = i < line.width - 1 ? countNonZero(mainLine.col(i + 1)) : 0;
		cols[i] = nonZero;

		if (nonZero)
		{
			width++;
		}
		else if (!nonZero && width)
		{
			if (width < 10 && nextNonZero)
			{
				width++;
			}
			else
			{
				chars.push_back(Range(i - width, i));
				width = 0;
			}
		}
    }

    if (!chars.size())
    {
		return blockList;
    }

    vector<char> outText;
    while (!chars.empty())
    {
		vector<short> widthArray;
		for (list<Range>::iterator item = chars.begin(); item != chars.end(); item++)
		{
			short width = item->end - item->start;
			widthArray.push_back(width);
		}
		sort(widthArray.begin(), widthArray.end());
		int median = widthArray[widthArray.size() / 2];
		Range top = chars.back();
		short start = top.start, end = top.end;
		int width = end - start;

		Mat charImg = mainLine(Range::all(), top);

		if (width*width < median * median / 6 || width < median / 6)
		{
			chars.pop_back();
			continue;
		}

		if (width > median * 1.8 || width > line.height)
		{
			short pos = searchMinimum(cols + start, width);
			if (width - pos > 8 && pos > 8)
			{
				chars.pop_back();
				chars.push_back(Range(start, start + pos + 1));
				chars.push_back(Range(start + pos + 1, end));
			}
			else
			{
				cv::copyMakeBorder(charImg.clone(), charImg, 3, 3, 3, 3, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
				blockList.push_front(charImg);
				chars.pop_back();
			}
		}
		else
		{
			cv::copyMakeBorder(charImg.clone(), charImg, 3, 3, 3, 3, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
			blockList.push_front(charImg);
			chars.pop_back();
		}
    }

    return blockList;
}

Processor::Processor(const char *path)
{
    // 创建识别api
    api = new tesseract::TessBaseAPI();
    api->Init(path, "eng");				       // 初始化识别数据路径和语言
    api->SetVariable("tessedit_char_blacklist", ":,\".-");     // 设置识别黑名单
    api->SetVariable("tessedit_char_whitelist", "1234567890"); // 设置识别白名单
    api->SetVariable("save_blob_choices", "T");
    api->SetPageSegMode(tesseract::PSM_SINGLE_LINE); // 设置识别模式为单行文本
}

Processor::~Processor()
{
    // 清除api防止内存泄漏
    api->End();
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
		cerr << "image is empty" << endl;
		return "";
    }

    Size crop = Size(width, height);

    // 初始化变量
    Mat baup, thr, med, last;

    baup = img.clone();

    adaptiveThreshold(baup, thr, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 9, 9);

    Mat closing = getStructuringElement(MORPH_RECT, Size(13, 1));
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

		if (angle < -45.)
		{
			angle += 90.0;
			int tmp = area.width;
			area.width = area.height;
			area.height = tmp;
		}

		// 第一次粗过滤
		// 根据形状和面积过滤
		if (area.area() > limit * 0.01 || area.area() < limit * 0.001 || area.width < 5 * area.height)
		{
			continue;
		}

		if (area.width > 300 || area.width < 50)
		{
			continue;
		}

		// 扩展切割区域，提高识别率
		if (center.x - area.width / 2 > 4)
		{
			area.width += 8;
		}
		if (center.y - area.height / 2 > 3)
		{
			area.height += 4;
		}

		// 在原始图片中取出手机号, 旋转
		Mat M, rotated, candidate_region;
		M = cv::getRotationMatrix2D(rotatedRect.center, angle, 1.0);
		cv::warpAffine(baup, rotated, M, baup.size(), cv::INTER_CUBIC);
		cv::getRectSubPix(rotated, area, rotatedRect.center, candidate_region);

		candidate_region = usm(candidate_region);
		// 放大图像
		if (candidate_region.rows < 25 || candidate_region.cols < 120)
		{
			resize(candidate_region, candidate_region, Size(), 1.8, 1.8, INTER_CUBIC);
		}

		// 二值化，抑制干扰
		if (candidate_region.cols < 160)
		{
			cv::adaptiveThreshold(candidate_region, candidate_region, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 5, 3);
		}
		else if (candidate_region.cols < 240)
		{
			cv::adaptiveThreshold(candidate_region, candidate_region, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 7);
		}
		else
		{
			cv::adaptiveThreshold(candidate_region, candidate_region, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 9);
		}

		filterNoise(candidate_region);
		if (candidate_region.cols > 420 || candidate_region.cols < 140)
		{
			continue;
		}

		// 切割图像字符区域
		list<Mat> blockList = phone_classify(candidate_region);
		if (blockList.empty() || blockList.size() < 10 || blockList.size() > 16)
		{
			continue;
		}

		// 拼接图像
		Mat combine = *(blockList.begin());
		for (list<Mat>::iterator i = ++(blockList.begin()); i != blockList.end(); i++)
		{
			hconcat(combine, *i, combine);
		}

		// 识别手机号
		string num = decodeNum(combine);
		if (num != "NO")
		{
			size_t pos = string::npos;
			pos = path.find(num.substr(0, 11));
			if (pos == string::npos)
			{
			return num;
			}
		}
    }

    // 识别失败
    return "";
}

/**
 * 手机号数字识别程序
 **/
string Processor::decodeNum(Mat image)
{
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
			if (outText[front + 2] == '8' || outText[front + 2] == '6' || outText[front + 2] == '0' || outText[front + 2] == '7')
			findPhone = true;
			goto finally;
		}
    }
// 识别过程结束清理
finally:
    // 清空识别数据
    api->Clear();
    api->ClearAdaptiveClassifier();
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
		return substr + ":" + to_string(average_confidence);
    }
    else
    {
		return "NO";
    }
}
