#include "processor.h"
#include <iostream>

int main() {

  Processor *processor = new Processor(NULL);

  Mat src = imread("/Users/lambda/Downloads/bar.jpg");

  if (src.empty()) {
    cout << "no image" << endl;
    exit(0);
  }

  cv::Rect rect = cv::Rect(cv::Point(40, 255), cv::Size(350, 90));

  int k = 1;
  while (k) {
	std::string phone = processor->extract_phone(src, rect, 1);
	cout<< "-----phone: " << phone << "-----" <<endl;
    k--;
  }

  return 0;
}
