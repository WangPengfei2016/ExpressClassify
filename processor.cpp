#include "processor.h"
#include <iostream>

using namespace cv;
using namespace std;

void show(string name, Mat img)
{
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
    cv::imshow(name, img);
    cv::waitKey();
    cv::destroyAllWindows();
}

bool comp(const vector<Point> key1, const vector<Point> key2)
{
    // 比较轮廓面积
    return contourArea(key1) > contourArea(key2);
}

bool phone_classify(Mat region)
{
    /* 获取水平方向投影，取得垂直最大像素个数已确定分割字符阈值 */
    /* 记录每一行像素突变次数，取得最大值 */
    int *a = new int[region.cols](), max_pixs = 0;
    uint max_change_times = 0, last_pix = 0, change_times = 0;
    // 行遍历
    for (int i = 0; i < region.rows; ++i)
    {
        uchar *p = region.ptr<uchar>(i);
        // 列遍历
        for (int j = 0; j < region.cols; ++j)
        {
            if (p[j] == 255)
            {
                a[j]++;
                if (max_pixs < a[j])
                {
                    max_pixs = a[j];
                }
            }
            if (p[j] != last_pix)
            {
                last_pix = p[j];
                change_times++;
            }
        }
        max_change_times = change_times > max_change_times ? change_times : max_change_times;
        change_times = 0;
    }

    // 像素突变次数大于100次，判定不是手机号
    if (max_change_times > 50 || max_change_times < 20)
    {
        return false;
    }

    // 根据投影切割字符，判断是否包含手机号
    // 循环切割，避免字符粘连造成分割不足
    float rate = 0.1; // 初始阈值比例
    while (true)
    {
        int limit = max_pixs * rate;
        vector<Rect> s;
        bool isChar = false;
        int start = 0;
        for (int k = 0; k < region.cols; ++k)
        {
            if (a[k] > limit && !isChar)
            {
                start = k;
                isChar = true;
            }
            else if (a[k] < limit && isChar)
            {
                s.push_back(Rect(Point(start, 0), Size(k - start, region.rows)));
                isChar = false;
            }
        }

        // 判断字符个数
        if (s.size() > 9 && s.size() < 15)
        {
            return true;
        }
        // 阈值增长终止条件
        if (rate > 0.3)
            break;
        rate += 0.05;
    }

    return false;
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
    Mat img = imread(path);

	if (!img.data)
	{
		return "";
	}
    Rect rect = Rect(Point(0, 0), cv::Size(width, height));
    // 初始化变量
    Mat baup, gray, thr, med, last;
    // 条码宽度
    uint cols = (uint)rect.width;

	baup = img.clone();

    cvtColor(baup, gray, cv::COLOR_RGB2GRAY);
    adaptiveThreshold(gray, thr, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 20);

    Mat closing = getStructuringElement(MORPH_RECT, Size(cols / 30, 1));
    morphologyEx(thr, med, MORPH_CLOSE, closing);

    Mat opening = getStructuringElement(MORPH_RECT, Size(cols / 100, cols / 200));
    morphologyEx(med, last, MORPH_OPEN, opening);

    // 获取所有轮廓，根据轮廓面积排序
    vector< vector<Point> > contours;
    findContours(last, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), comp);

    for (int i = 0; i < contours.size(); ++i)
    {

        Rect candidate_rect = boundingRect(contours[i]);

        // 第一次粗过滤
        // 根据形状和面积过滤
        if (candidate_rect.height < rect.height / 8 || candidate_rect.height > rect.height / 2)
        {
            continue;
        }

        // 扩大手机号候选区域
        if (candidate_rect.y > 3)
        {
            candidate_rect.y -= 3;
            candidate_rect.height += 3;
        }

        // 在原始图片中取出手机号
        Mat orgin_mat(gray, candidate_rect);
        Mat candidate_region = orgin_mat.clone();

        // 二值化，抑制干扰
		cv::adaptiveThreshold(candidate_region, candidate_region, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 7, 5);

        // 精细过滤
        if (!phone_classify(candidate_region))
        {
            continue;
        }

        // 开始识别手机号
        string num = recognize_num(candidate_region);
        if (num != "NO")
        {
            return num;
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
            if (conf < 60)
            {
                continue;
            }
            confidence.push_back(conf);
            const char *symbol = ri->GetUTF8Text(level);
            if (symbol == 0)
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
        return outText.substr(front, front + 11);
    }
    else
    {
        return "NO";
    }
}
