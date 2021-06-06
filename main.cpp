#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <map>
#include <set>

#include "pixelModifications.h"
#include "segmentation.h"
#include "kernelModifications.h"
#include "coefficientCalculation.h"

namespace fs = std::filesystem;



int main() {
//    cv::Mat mat = cv::imread("../images/014820.jpg");
////    cv::Mat mat = cv::imread("../images/ideal.jpg");
//    cv::namedWindow("Window1");
//    cv::imshow("Window1", mat);
////    cv::Mat3b editableMat = mat;
////    levelHistogramRgb(editableMat);
////    cv::Mat1d editableMat = getOnlyPixelsWithGivenValues(mat, {95, 145}, {50, 255}, {20, 190});
////    std::vector<SegmentationResult> segmentationResult = getSegments(editableMat, 20, 20);
//    cv::Mat3b hsv_mat;
//    cv::cvtColor(mat, hsv_mat, cv::COLOR_BGR2HSV);
//    std::vector<SegmentationResult> segmentationResult =
//            getSegmentsStartingWithGivenColor(hsv_mat, {100, 130}, {60, 255}, {60, 170},
//                                              {5, 15, 10},
//                                              {10,10});
////    std::vector<SegmentationResult> segmentationResult =
////            getSegmentsStartingWithSeedsGrey(hsv_mat, {100, 130}, {60, 255}, {60, 170},
////                                              5,
////                                              {10,10});
//    for(const auto& item : segmentationResult) {
//        cv::namedWindow("Window2");
//        cv::imshow("Window2", item.object);
//        cv::waitKey(-1);
//    }
//
////    cv::Rect roi = {1,1,3,3};
////    cv::Mat resized = cv::Mat(mat, roi);
////    cv::namedWindow("Window2");
////    cv::imshow("Window2", resized);
////    levelHistogram(resized);
//
////    cv::Mat3b hsvMat;
////    cv::cvtColor(mat, hsvMat, cv::COLOR_BGR2HSV);
////    levelHistogram(hsvMat);
////    cv::cvtColor(hsvMat, mat, cv::COLOR_HSV2BGR);
////    cv::namedWindow("Window2");
////    cv::imshow("Window2", editableMat);
////    cv::waitKey(-1);
//    return 0;





    std::string path("../images");
    std::vector<std::string> files;
    for (auto &p : fs::recursive_directory_iterator(path))
    {
        files.push_back(p.path().lexically_normal().string());
        std::cout << p.path().lexically_normal().string() << '\n';
    }

    for(const auto& file : files) {
        cv::Mat mat = cv::imread(file);
        cv::Mat3b editableMat = mat;
//        cv::namedWindow("Window1");
//        cv::imshow("Window1", mat);
//        changeContrast(editableMat, 1.5);
//        mat = getOnlyPixelsWithGivenValues(editableMat, {95, 145}, {50, 255}, {20, 190});
        cv::Mat3b hsvMat;
        cv::cvtColor(mat, hsvMat, cv::COLOR_BGR2HSV);
        std::set<IntPair> seeds = getSegmentationSeedsHsv(hsvMat,{100, 127},
                                                          {163, 204}, {45, 153});
        cv::Mat3b matWithSeeds = mat.clone();
        for(const auto& seed : seeds) {
            matWithSeeds(seed.first, seed.second) = {255, 0 , 255};
        }
//        cv::namedWindow("Window2");
//        cv::imshow("Window2", matWithSeeds);

        cv::Mat1b grayMap;
        cv::cvtColor(mat, grayMap, cv::COLOR_BGR2GRAY);
//        cv::namedWindow("Window3");
//        cv::imshow("Window3", grayMap);

//        grayMap = maxFilter(grayMap, 3, MedianValue());
//        cv::namedWindow("Window3i");
//        cv::imshow("Window3i", grayMap);

//        levelHistogramGray(grayMap);
//        cv::namedWindow("Window3a");
//        cv::imshow("Window3a", grayMap);

//        grayMap = maxFilter(grayMap, 3, MinPixelValue());
        changeContrast(grayMap, 2);
        cv::namedWindow("Window3b");
        cv::imshow("Window3b", grayMap);
//        cv::waitKey(-1);

        std::vector<SegmentationResult> segments = getSegmentsStartingWithSeedsGrey(grayMap, seeds, 10, {18, 18});
        for(const auto& segment : segments) {
            cv::namedWindow("Window4");
            cv::imshow("Window4", segment.object);
            ImageMoments imageMoments = calculateImageMoments(segment.object);
            double m1 = calculateM1(imageMoments);
            double m7 = calculateM7(imageMoments);
            std::cout << "m1: " << m1 << "\tm7: " << m7 << std::endl;
            cv::waitKey(-1);
        }




    }
    return 0;


//    std::cout << "Hello, World!" << std::endl;
//    return 0;
}
