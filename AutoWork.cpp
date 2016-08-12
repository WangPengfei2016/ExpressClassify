//
// Created by 黄金 on 2016/8/2.
//

#include "AutoWork.h"
#include "config.h"
#include <numeric>
#include <map>
#include <iostream>

void show(string name, Mat image) {
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, image);
    int k = waitKey(0);
    if (k == 27) {
        cv::destroyWindow(name);
    } else if (k == (int)'s') {
        printf("save");
        imwrite("/Users/hj/Downloads/save.jpg", image);
        cv::destroyWindow(name);
    }
}

Mat rotate_img(Point center, float angle, Mat image, Size size) {
    Mat tmp;
    Mat rot_mat = getRotationMatrix2D(center, angle, 1);
    warpAffine(image, tmp, rot_mat, size);
    return tmp;
}

bool phone_classify(Mat region) {
//    获取水平方向投影
    int *a = new int[region.cols]();
    int max = 0;
    for (int i = 0; i < region.rows; ++i) {
        uchar *p = region.ptr<uchar>(i);
        for (int j = 0; j < region.cols; ++j) {
            if (p[j] == 255) {
                a[j]++;
                if (max < a[j]) {
                    max = a[j];
                }
            }

        }
    }
//    根据投影切割字符，判断是否包含手机号
//    循环切割，避免字符粘连造成分割不足
    float rate = 0.05;     // 初始阈值
    while (true) {
        int limit = max*rate;
        vector<Rect> s;
        bool isChar = false;
        int start = 0;
        for (int k = 0; k < region.cols; ++k) {
            if (a[k] > limit && !isChar){
                start = k;
                isChar = true;
            } else if (a[k] < limit && isChar) {
                s.push_back(Rect(Point(start, 0), Size(k-start, region.rows)));
                isChar = false;
            }
        }

//        判断字符个数
        if (s.size() > 9 && s.size() < 15){
            return true;
        }
        //        阈值增长终止条件
        if (rate > 0.15)
            break;
        rate += 0.05;
    }

    return false;

}

bool comp(const vector<Point> key1, const vector<Point> key2) {
    return contourArea(key1) > contourArea(key2);
}

void AutoWork::init(const char *path) {

    api = new TessBaseAPI();
    api->Init(NULL, "eng");
    api->SetVariable("tessedit_char_blacklist", ":,\".-");
    api->SetVariable("tessedit_char_whitelist", "1234567890");
    api->SetPageSegMode(PSM_SINGLE_LINE);

}

Mat AutoWork::locate_express(Mat src) {
    Mat gray, thr, med, last;

    vector<Mat> channels;
    split(src, channels);
    gray = channels[2];

    threshold(gray, thr, 150, 255, THRESH_BINARY);

    Mat opening = getStructuringElement(MORPH_RECT, Size(10, 10));
    morphologyEx(thr, med, MORPH_OPEN, opening);


    Mat closing = getStructuringElement(MORPH_RECT, Size(50, 50));
    morphologyEx(med, last, MORPH_CLOSE, closing);

    vector< vector<Point> > contours;
    findContours(last.clone(), contours, RETR_LIST, CHAIN_APPROX_NONE);
    sort(contours.begin(), contours.end(), comp);

    RotatedRect roteRect = minAreaRect(contours[0]);
    Point center = roteRect.center;
    float angle = roteRect.angle;
    gray = rotate_img(center, angle, gray, med.size());
    last = rotate_img(center, angle, last, med.size());

    contours.clear();
    findContours(last.clone(), contours, RETR_LIST, CHAIN_APPROX_NONE);
    sort(contours.begin(), contours.end(), comp);
    Rect rect = boundingRect(contours[0]);

    Mat area(gray, rect);
    return area;
}

string AutoWork::extract_bar(Mat *obj) {
    Mat thr, med, last;

    threshold(*obj, thr, 90, 255, THRESH_BINARY_INV);

    Mat cl = getStructuringElement(MORPH_RECT, Size(15, 15));
    morphologyEx(thr, med, MORPH_CLOSE, cl, Point(-1, -1), 2);

    Mat op = getStructuringElement(MORPH_RECT, Size(15, 15));
    morphologyEx(med, last, MORPH_OPEN, op, Point(-1, -1), 2);

    vector< vector<Point> > contours;
    findContours(last, contours, RETR_LIST, CHAIN_APPROX_NONE);
    sort(contours.begin(), contours.end(), comp);

    vector<Point> approxCurve;
    int i;
    for (i=0; i < contours.size(); i++) {
        approxPolyDP(contours[i], approxCurve, double(contours[i].size())*0.05, true);
        if (approxCurve.size() == 4) {
            break;
        }
    }
    Rect rect = boundingRect(contours[i]);

    int bar_width = rect.width;
    int bar_height = rect.height;

    rect.x -= bar_width/20;
    rect.width += bar_width/10;
    rect.y -= bar_height/20;
    rect.height += bar_height/10;
    
    Mat bar(obj->clone(), rect);
    int width = rect.width;
    int height = rect.height;

    if (width < height) {
        Mat tmp;
        transpose(bar, tmp);
        flip(tmp, bar, 1);
        transpose(*obj, tmp);
        flip(tmp, *obj, 1);
    }

    Mat barShape;
    threshold(bar, barShape, 0, 255, THRESH_OTSU|THRESH_BINARY_INV);

    Mat k = getStructuringElement(MORPH_RECT, Size(bar.cols/10, 1));
    morphologyEx(barShape, med, MORPH_CLOSE, k);

    findContours(med, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), comp);


    if (contours.size() > 1) {
        Rect one = boundingRect(contours[0]);
        Rect two = boundingRect(contours[1]);

        if (two.y < one.y) {
            Mat tmp;
            transpose(bar, tmp);
            flip(tmp, tmp, 1);
            transpose(tmp, tmp);
            flip(tmp, bar, 1);

            transpose(*obj, tmp);
            flip(tmp, tmp, 1);
            transpose(tmp, tmp);
            flip(tmp, *obj, 1);
        }
    }

    return recognize_bar(bar);
}

string AutoWork::extract_phone(Mat *obj) {
    uint cols = (uint)obj->cols;
    ConfigFactory *factory = new ConfigFactory(); 
    Config *config = factory->createConfig(VERTICAL, obj->cols);
    Mat baup, thr, med, last;
    baup = obj->clone();
    adaptiveThreshold(*obj, thr, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 20);

    Mat closing = getStructuringElement(MORPH_RECT, Size(cols/40, 1));
    morphologyEx(thr, med, MORPH_CLOSE, closing);

    Mat opening = getStructuringElement(MORPH_RECT, Size(cols/100, cols/200));
    morphologyEx(med, last, MORPH_OPEN, opening);

    vector< vector<Point> > contours;
    findContours(last, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    sort(contours.begin(), contours.end(), comp);

    for (int i = 0; i < contours.size(); ++i) {
        Rect like_phone = boundingRect(contours[i]);
        
        if (like_phone.x > 10) {
            like_phone.x -= 10;
            like_phone.width += 10;
        }
        
        // 在原始图片中取出手机号
        Mat phone_region(baup, like_phone);

        if (!config->filter_region_by_shape(phone_region)) {
            continue;
        }

        // 手机号区域过小时，进行插值运算，放大手机号区域
        if (like_phone.width < 110) {
            resize(phone_region, phone_region, Size(like_phone.width*3, like_phone.height*3), 0, 0, INTER_LANCZOS4);
        }
        // 局部阈值二值化，抑制干扰
        adaptiveThreshold(phone_region, phone_region, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 7, 15);

        if (!phone_classify(phone_region.clone())) {
            continue;
        }

        string num = recognize_num(phone_region);

        if (num != "NO") {
            return num;
        }
    }

    return "NO";
}



string AutoWork::recognize_bar(Mat area) {
    
    Mat bar;

    bar = area.clone();

    int width = bar.cols;
    int height = bar.rows;

    Image image(width, height, "Y800", bar.data, width*height);

    ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);

    scanner.scan(image);

    Image::SymbolIterator symbol = image.symbol_begin();

    if (symbol != image.symbol_end()) {
        string data = symbol->get_data();
        return symbol->get_data();
    }

    return "NO";
}

string AutoWork::recognize_num(Mat image, bool isPhone) {

    string outText = "";
    vector<float> confidence;

    api->SetImage(image.data, image.size().width, image.size().height, image.channels(), (int)image.step1());
    api->Recognize(0);

    tesseract::ResultIterator* ri = api->GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_SYMBOL;

    if (ri != 0) {
        do {
            float conf = ri->Confidence(level);

            if (conf < 61) {
                continue;
            }
            confidence.push_back(conf);
            const char* symbol = ri->GetUTF8Text(level);
            if (symbol == 0) {
                continue;
            }
            outText += symbol;
            delete[] symbol;
        } while (ri->Next(level));

    }

    size_t length = outText.length();

    // 判断是否是手机号
    if ( length < 15 && length > 10) {
        for (uint j = 0; j <= length-11; j++) {
            size_t start = length-(11+j);
            string out = outText.substr(start, 11);
            if (out[0] == '1') {
                switch (out[1]) {
                    case '3':
                    case '5':
                    case '7':
                    case '8': {
                        double t = accumulate(confidence.begin(), confidence.end(), 0.0);
                        if ((t/11) > 75) {
                            cout<< out <<endl;
                            api->Clear();
                            return out;
                        }
                        break;
                    }
                    default:
                        break;
                }
            }
        }

    }
    api->Clear();
    return "NO";
}

