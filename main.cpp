#include <iostream>
#include "processor.h"

int main() {

  cv::Mat obj;
  Processor *processor = new Processor(NULL);

  Mat src = imread("/Users/hj/Downloads/img/17.jpg");
  
  if (src.empty()) {
    cout << "no image" << endl;
    exit(0);
  }

  double t = (double)getTickCount();
  int k = 1;
  while (k) {
    obj = processor->locate_express(src);
    processor->extract_bar(&obj);

    processor->extract_phone(&obj);
    k--;
  }
  cout << ((double)getTickCount() - t) / getTickFrequency() << "sec" << endl;
  return 0;
}