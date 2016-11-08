#include "processor.h"
#include <iostream>

int main() {

  Processor *processor = new Processor(NULL);

  std::string path = "/Users/lambda/Downloads/70493001194206,08111336.bmp";

  int k = 1;
  while (k) {
	std::string phone = processor->extract_phone(path, 43, 480, 477, 88, 1);
	cout<< "-----phone: " << phone << "-----" <<endl;
    k--;
  }

  return 0;
}
