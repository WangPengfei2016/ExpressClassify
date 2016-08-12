cmake_minimum_required(VERSION 3.5)
project(autoClassify)

include_directories("/usr/local/Cellar/opencv3/3.1.0_3/include/" "/usr/local/Cellar/tesseract/3.04.01_1/include/" "/usr/local/Cellar/zbar/0.10_1/include/")
link_directories("/usr/local/Cellar/opencv3/3.1.0_3/lib/" "/usr/local/Cellar/tesseract/3.04.01_1/lib/" "/usr/local/Cellar/zbar/0.10_1/lib/")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lopencv_highgui -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_flann -lopencv_xfeatures2d  -ltesseract -lzbar")

set(SOURCE_FILES main.cpp AutoWork.cpp AutoWork.h Config.cpp Config.h)
add_executable(autoClassify ${SOURCE_FILES})