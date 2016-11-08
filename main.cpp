#include "processor.h"
#include <iostream>

int main() {

  Processor *processor = new Processor(NULL);

  std::string path = "/Users/lambda/Downloads/3316275218734,08183036.bmp";

  int k = 1;
  while (k) {
	std::string phone = processor->extract_phone(path, 54, 475, 454, 98, 1);
	cout<< "-----phone: " << phone << "-----" <<endl;
    k--;
  }

  return 0;
}
