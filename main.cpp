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


std::vector<cv::Rect> findObjects(const cv::Mat& mat) {
    cv::Mat3b hsvMat;
    cv::cvtColor(mat, hsvMat, cv::COLOR_BGR2HSV);
    std::set<IntPair> seeds = getSegmentationSeedsHsv(hsvMat,{100, 127},
                                                      {150, 204}, {45, 153});
    cv::Mat1b grayMap;
    cv::cvtColor(mat, grayMap, cv::COLOR_BGR2GRAY);
    grayMap = kernelFilter(grayMap, 3, MinPixelValue());
    uchar avgLuminosity = getAverageLuminosity(grayMap);
    changeBrightness(grayMap, 127 - avgLuminosity);
    changeContrast(grayMap, 3);

    std::vector<SegmentationResult> segments = getSegments8DirGrey2(grayMap, seeds, 10, {18, 18});
    std::vector<cv::Rect> objectsFound;
    for(const auto& segment : segments) {
        ImageMoments imageMoments = calculateImageMoments(segment.object);
        double m1 = calculateM1(imageMoments);
        double m7 = calculateM7(imageMoments);
        if(m1 > 0.35 && m1 < 0.68 && m7 > 0.029 && m7 < 0.084) {
            objectsFound.push_back(segment.roi);
        }
    }
    return objectsFound;
}


int main() {

    std::string path("../images");
    std::vector<std::string> files;
    for (auto &p : fs::recursive_directory_iterator(path)) {
        files.push_back(p.path().lexically_normal().string());
        std::cout << p.path().lexically_normal().string() << '\n';
    }

    for (const auto &file : files) {
        cv::Mat mat = cv::imread(file);
        std::vector<cv::Rect> objectsFound = findObjects(mat);
        cv::Mat3b matWithRoi = mat.clone();
        for (const auto &roi : objectsFound) {
            cv::rectangle(matWithRoi, roi, {255, 0, 255}, 2);
        }
        cv::namedWindow("picture");
        cv::imshow("picture", matWithRoi);
        cv::waitKey(-1);
    }

    return 0;
}
