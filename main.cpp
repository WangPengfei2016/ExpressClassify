#include <iostream>
#include "processor.h"

int main() {

  Mat obj;
  Processor *processor = new Processor(NULL);

  Mat src = imread("/Users/hj/Downloads/img/17.jpg");

  if (src.empty()) {
    cout << "no image" << endl;
    exit(0);
  }

  cv::Rect rect = cv::Rect(cv::Point(10, 10), cv::Size(100, 100));

  int k = 1;
  while (k) {
    processor->extract_phone(obj, rect);
    k--;
  }

  return 0;
}
