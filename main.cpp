#include "processor.h"
#include <iostream>

int main() {

  Processor *processor = new Processor(NULL);

  std::string path = "/Users/lambda/Downloads/882419904842751278,05125613.bmp";

  int k = 1;
  while (k) {
	std::string phone = processor->extract_phone(path, 60, 380, 400, 100, 1);
	cout<< "-----phone: " << phone << "-----" <<endl;
    k--;
  }

  return 0;
}
