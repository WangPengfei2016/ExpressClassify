#include <iostream>
#include "Processor.h"

int main() {

  cv::Mat obj;
  Processor *processor;
  processor->init(NULL);

  cv::Mat src = cv::imread("/Users/hj/Downloads/img/17.jpg");
  
  if (src.empty()) {
    cout << "no image" << endl;
    exit(0);
  }

  double t = (double)cv::getTickCount();
  int k = 1;
  while (k) {
    obj = processor->locate_express(src);
    processor->extract_bar(&obj);

    processor->extract_phone(&obj);
    k--;
  }
  cout << ((double)cv::getTickCount() - t) / cv::getTickFrequency() << "sec" << endl;
  return 0;
}