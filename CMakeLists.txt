cmake_minimum_required(VERSION 3.19)
project(prob)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES
        main.cpp
        pixelModifications.cpp
        segmentation.cpp
        kernelModifications.cpp coefficientCalculation.cpp coefficientCalculation.h)

add_executable(prob ${SOURCES})

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(prob ${OpenCV_LIBS})