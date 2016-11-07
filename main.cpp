#include "processor.h"
#include <iostream>

int main() {

  Processor *processor = new Processor(NULL);

  std::string path = "/Users/lambda/Downloads/882419904842751278,07155403.bmp";

  int k = 1;
  while (k) {
	std::string phone = processor->extract_phone(path, 20, 400, 340, 95, 1);
	cout<< "-----phone: " << phone << "-----" <<endl;
    k--;
  }

  return 0;
}
