//
// Created by Michal on 05.06.2021.
//

#include <opencv2/imgproc.hpp>
#include <map>

#include "pixelModifications.h"
#include "typedefs.h"

#include <iostream>


cv::Mat3b filterPixelsWithGivenValues(const cv::Mat3b& mat, const UcharPair& hueRange, const UcharPair& saturationRange,
                                      const UcharPair& valueRange) {
    cv::Mat3b hsv_mat;
    cv::cvtColor(mat, hsv_mat, cv::COLOR_BGR2HSV);
    for(cv::Vec3b& pixel : hsv_mat) {
        if(pixel[0] < hueRange.first || pixel[0] > hueRange.second ||
           pixel[1] < saturationRange.first || pixel[1] > saturationRange.second ||
           pixel[2] < valueRange.first || pixel[2] > saturationRange.second) {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
        }
    }
    cv::Mat3b resultMat;
    cv::cvtColor(hsv_mat, resultMat, cv::COLOR_HSV2BGR);
    return resultMat;
}

cv::Mat1d getOnlyPixelsWithGivenValues(const cv::Mat3b &mat, const UcharPair &hueRange, const UcharPair &saturationRange,
                             const UcharPair &valueRange) {
    cv::Mat3b hsv_mat;
    cv::cvtColor(mat, hsv_mat, cv::COLOR_BGR2HSV);
    cv::Mat1d resultMat = cv::Mat1d(hsv_mat.rows, hsv_mat.cols);
    for(int i = 0; i < hsv_mat.total(); ++i) {
        if(hsv_mat(i)[0] < hueRange.first || hsv_mat(i)[0] > hueRange.second ||
           hsv_mat(i)[1] < saturationRange.first || hsv_mat(i)[1] > saturationRange.second ||
           hsv_mat(i)[2] < valueRange.first || hsv_mat(i)[2] > saturationRange.second) {
            resultMat(i) = 0;
        } else {
            resultMat(i) = 1;
        }
    }
    return resultMat;
}

void levelHistogram(cv::Mat3b &hsvMat) {
    if(hsvMat.empty()) return;
    unsigned i = 1;
    std::map<uchar, unsigned> nOfPixels;
    for(const auto& pixel : hsvMat) {
        ++nOfPixels[pixel[2]];
    }
    if(nOfPixels.size() <= 1) {
        return;
    }
    unsigned d0 = (*nOfPixels.begin()).second;
    unsigned sum = 0;
    std::map<uchar, uchar> lut;
    for(const auto& item : nOfPixels) {
        sum += item.second;
        lut[item.first] = (sum - d0) * (256 - 1) / (hsvMat.total() - d0);
    }
    for(auto& pixel : hsvMat) {
        pixel[2] = lut[pixel[2]];
    }
}


void levelHistogramRgb(cv::Mat3b &mat) {
    if(mat.empty()) return;
    unsigned i = 1;
    std::map<unsigned, unsigned> nOfPixels;
    for(const auto& pixel : mat) {
        ++nOfPixels[pixel[0] + pixel[1] + pixel[2]];
    }
    if(nOfPixels.size() <= 1) {
        return;
    }
    unsigned d0 = (*nOfPixels.begin()).second;
    unsigned sum = 0;
    std::map<unsigned, double> lut;
    for(const auto& item : nOfPixels) {
        sum += item.second;
        unsigned targetValue = (sum - d0) * (255 * 3) / (mat.total() - d0);
        lut[item.first] = (double) targetValue / item.first;
    }
    for(auto& pixel : mat) {
        double coefficient = lut[pixel[0] + pixel[1] + pixel[2]];
        pixel[0] = uchar(coefficient * pixel[0]);
        pixel[1] = uchar(coefficient * pixel[1]);
        pixel[2] = uchar(coefficient * pixel[2]);
    }
}

void changeContrast(cv::Mat3b &mat, double coefficient) {
    const uchar HALF_MAX = 127;
    for(auto& pixel : mat) {
        for(int i = 0; i < 3; ++i) {
            double value = coefficient * (pixel[i] - HALF_MAX) + HALF_MAX;
            value = value < 0 ? 0 : (value > 255 ? 255 : value);
            pixel[i] = value; // NOLINT(cppcoreguidelines-narrowing-conversions)
        }
    }
}
