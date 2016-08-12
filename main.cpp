#include "AutoWork.h"
#include <iostream>

int main() {

  Mat obj;
  AutoWork *api;
  api->init(NULL);

  Mat src = imread("/Users/hj/Downloads/img/17.jpg");
  if (src.empty()) {
    cout << "no image" << endl;
    exit(0);
  }

  double t = (double)getTickCount();
  int k = 1;
  while (k) {
    obj = api->locate_express(src);
    api->extract_bar(&obj);

    api->extract_phone(&obj);
    k--;
  }
  cout << ((double)getTickCount() - t) / getTickFrequency() << "sec" << endl;
  return 0;
}