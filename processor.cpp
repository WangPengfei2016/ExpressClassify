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

bool comp(const vector<Point> key1, const vector<Point> key2)
{
    // 比较轮廓面积
    return contourArea(key1) > contourArea(key2);
}


bool phone_classify(Mat region)
{
    list<Rect> faultage;
    /* 获取垂直方向投影，确定文本行 */
    int *v = new int[region.rows]();
    for (int i = 0; i <region.rows; ++i)
    {
        uchar *p = region.ptr<uchar>(i);
        for (int j = 0; j < region.cols; ++j)
        {
            if (p[j] == 255) {
                v[i]++;
            }
        }
    }

    int start = 0;
    bool block = false;
    for (int i = 1; i < region.rows; i++)
    {
        if (!block) {
            if (v[i] != 0 && v[i] < region.cols*0.6)
            {
                block = true;
                start = i;
            }
            continue;
		} else if (v[i] > region.cols*0.6) {
			block = false;
			start = 0;
			continue;
		}
        if (v[i] < 10)
        {
            Rect rect;
            rect.y = start;
            rect.height = i-start;
            rect.width = region.cols;
            faultage.push_back(rect);
            start = 0;
            block = false;
        }
    }

    Rect final;
    for (list<Rect>::iterator rect = faultage.begin(); rect != faultage.end(); rect++)
    {
        if (final.height < rect->height)
        {
            final.x = rect->x;
            final.y = rect->y;
            final.width = rect->width;
            final.height = rect->height;
        }
    }

	if (final.area() ==0 ) {
		return false;
	}

    Mat hand(region, final);
    /* 获取水平方向投影，取得垂直最大像素个数已确定分割字符阈值 */
    /* 记录每一行像素突变次数 */
	short **changes = new short*[hand.rows];
	for (int i = 0; i < hand.rows; i++) {
		changes[i] = new short[hand.cols];
	}
    int *a = new int[hand.cols]();
    uint last_pix = 0;
    // 行遍历
    for (int i = 0; i < hand.rows; ++i)
    {
        uchar *p = hand.ptr<uchar>(i);
        // 列遍历
        for (int j = 0; j < hand.cols; ++j)
        {
            if (p[j] == 255)
            {
                a[j]++;
            }
            if (p[j] != last_pix)
            {
                last_pix = p[j];
				changes[i][j] = 1;
			} else {
				changes[i][j] = 0;
			}
        }
    }

    uint meanWidth = 0;
    int front = 0;
    list<Rect> rectList;
    for (int k = 0; k < hand.cols - 1; ++k)
    {
        if (a[k] ==0 && a[k+1] > a[k])
        {
            front = k;
        }
        else if (a[k] > a[k+1] && a[k+1] == 0)
        {
			if (k-front+1 > 2) {
				rectList.push_back(Rect(Point(front, 0), Size(k-front+1, hand.rows)));
				meanWidth += k-front+1;
			}
        }
    }

	if (!rectList.size()) {
		return false;
	}

    meanWidth /= rectList.size();
	list<Rect> chars;
    for (list<Rect>::iterator rect = rectList.begin(); rect != rectList.end(); ++rect) {

		int offset = rect->x;
		float change = 0.0;
		int sum = 0;

		for(int i = rect->y; i < rect->y+rect->height; i++)
		{
			for (int j = rect->x; j < rect->x+rect->width; j++)
			{
				sum += changes[i][j];
			}

		}

		change = sum*1.0/rect->height;
		if (rect->width > meanWidth*2 && change/2 < 6)
		{
			short count;
			int *extr = a+rect->x;
			list<int> tmp;

			for (int i = 1; i < rect->width-1; ++i)
			{
				if (extr[i] < extr[i-1] && extr[i] < extr[i+1])
				{
					tmp.push_back(i);
				}
			}

			int offset = rect->x;
			int min_pos = *(tmp.begin());
			for(list<int>::iterator item = tmp.begin(); item != tmp.end(); item++)
			{
				if (a[min_pos+offset] > a[offset+(*item)])
				{
					min_pos = *item;
				}

			}

			if (min_pos+offset < hand.rows && a[min_pos+offset] < 4){
				Rect first(Point(rect->x, 0), Size(min_pos, rect->height));
				Rect second(Point(rect->x+min_pos, 0), Size(rect->width-min_pos, rect->height));
				chars.push_back(first);
				chars.push_back(second);
			}
		} else if (change < 4){
			chars.push_back(*rect);
		} else {
			chars.clear();
		}
    }

	for (int i = 1; i < hand.rows; i++) {
		delete[] changes[i];
	}
	delete[] changes;

    if (chars.size() < 10 || chars.size() > 19)
    {
        return false;
    }
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
		cout<< "no data" <<endl;
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

	cv::blur(baup, baup, Size(3, 3));
    adaptiveThreshold(baup, thr, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 20);

    Mat closing = getStructuringElement(MORPH_RECT, Size(12, 1));
    morphologyEx(thr, med, MORPH_CLOSE, closing);

    Mat opening = getStructuringElement(MORPH_RECT, Size(4, 4));
    morphologyEx(med, last, MORPH_OPEN, opening);

    // 获取所有轮廓，根据轮廓面积排序
    vector< vector<Point> > contours;
    findContours(last, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), comp);

	int count = 0;
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
		if (center.x - area.width/2 > 6)
		{
			area.width += 12;
		}

		if (center.y - area.height/2 > 3)
		{
			area.height += 4;
		}

		// 在原始图片中取出手机号
		Mat M, rotated, candidate_region;
		M = cv::getRotationMatrix2D(rotatedRect.center, angle, 1.0);
		cv::warpAffine(baup, rotated, M, baup.size(), cv::INTER_CUBIC);
		cv::getRectSubPix(rotated, area, rotatedRect.center, candidate_region);

		if (candidate_region.rows < 25)
		{
			resize(candidate_region, candidate_region, Size(), 1.5, 1.5, INTER_LANCZOS4);
		}

		// 二值化，抑制干扰
		cv::adaptiveThreshold(candidate_region, candidate_region, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 7, 5);

		// 图片添加边框，提高识别率
		cv::copyMakeBorder(candidate_region, candidate_region, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

		// 精细过滤
		if (!phone_classify(candidate_region))
		{
			continue;
		}

		// 开始识别手机号
		count++;
		string num = recognize_num(candidate_region);
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
			if (conf < 65 && confidence.back() < 65) {
				outText = "";
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
        return outText.substr(front, front + 11)+":"+to_string(average_confidence);
    }
    else
    {
        return "NO";
    }
}
