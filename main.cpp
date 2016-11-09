#include "processor.h"
#include <iostream>

int main() {

  Processor *processor = new Processor(NULL);

  std::string path = "/Users/lambda/Downloads/376073476287,08185041.bmp";

  int k = 20;
  while (k) {
	std::string phone = processor->extract_phone(path, 54, 475, 454, 98, 1);
	cout<< "-----phone: " << phone << "-----" <<endl;
    k--;
  }

  return 0;
}
