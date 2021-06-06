//
// Created by Michal on 05.06.2021.
//

#include <utility>
#include <vector>
#include <set>
#include <map>
#include <opencv2/imgproc.hpp>

#include "segmentation.h"

#include <iostream>

std::set<IntPair>
getSegmentationSeedsHsv(const cv::Mat3b &hsv_mat, const UcharPair &hueRange, const UcharPair &saturationRange,
                        const UcharPair &valueRange) {
    std::set<IntPair> seeds;
    for(int i = 0; i < hsv_mat.rows; ++i) {
        for(int j = 0; j < hsv_mat.cols; ++j) {
            if(hsv_mat(i, j)[0] >= hueRange.first && hsv_mat(i, j)[0] <= hueRange.second &&
               hsv_mat(i, j)[1] >= saturationRange.first && hsv_mat(i, j)[1] <= saturationRange.second &&
               hsv_mat(i, j)[2] >= valueRange.first && hsv_mat(i, j)[2] <= saturationRange.second) {
                seeds.insert({i, j});
            }
        }
    }
    return seeds;
}

std::vector<SegmentationResult> getSegments(const cv::Mat1d &mat, int minWidth, int minHeight) {
    std::set<IntPair> pixelsLeft;
    for(int i = 0; i < mat.rows; ++i) {
        for(int j = 0; j < mat.cols; ++j) {
            if(mat(i, j) == 1) {
                pixelsLeft.insert({i, j});
            }
        }
    }
    std::vector<SegmentationResult> result;
    while(!pixelsLeft.empty()) {
        cv::Mat1d segment = cv::Mat1d(mat.rows, mat.cols, 0.0);
        int minRow = mat.rows, minCol = mat.cols, maxRow = 0, maxCol = 0;
        std::set<IntPair> pixelsToProceed;
        pixelsToProceed.insert(*pixelsLeft.begin());
        while(!pixelsToProceed.empty()) {
            std::set<IntPair> newPixelsToProceed;
            for(const auto& pixel : pixelsToProceed) {
                segment(pixel.first, pixel.second) = 1.0;
                if(pixel.first > maxRow) maxRow = pixel.first;
                if(pixel.first < minRow) minRow = pixel.first;
                if(pixel.second > maxCol) maxCol = pixel.second;
                if(pixel.second < minCol) minCol = pixel.second;
                pixelsLeft.erase(pixel);
                if(pixel.first < mat.rows && pixelsLeft.find({pixel.first + 1, pixel.second}) != pixelsLeft.end()) {
                    newPixelsToProceed.insert({pixel.first + 1, pixel.second});
                }
                if(pixel.first > 0 && pixelsLeft.find({pixel.first - 1, pixel.second}) != pixelsLeft.end()) {
                    newPixelsToProceed.insert({pixel.first - 1, pixel.second});
                }
                if(pixel.second < mat.cols && pixelsLeft.find({pixel.first, pixel.second + 1}) != pixelsLeft.end()) {
                    newPixelsToProceed.insert({pixel.first, pixel.second + 1});
                }
                if(pixel.second > 0 && pixelsLeft.find({pixel.first, pixel.second - 1}) != pixelsLeft.end()) {
                    newPixelsToProceed.insert({pixel.first, pixel.second - 1});
                }
            }
            pixelsToProceed = std::move(newPixelsToProceed);
        }
        int width = maxCol - minCol + 1;
        int height = maxRow - minRow + 1;
        if(width >= minWidth && height >= minHeight) {
            cv::Rect roi = {minCol, minRow, width, height};
            result.emplace_back(cv::Mat1d(segment, roi), roi);
        }
    }
    return result;
}

bool isPixelInTolerance(const cv::Vec3b& originalPix, const cv::Vec3b& newPix, const cv::Vec3b& tolerance) {
    return
        abs(originalPix[0] - newPix[0]) <= tolerance[0] &&
        abs(originalPix[1] - newPix[1]) <= tolerance[1] &&
        abs(originalPix[2] - newPix[2]) <= tolerance[2];
}

std::vector<SegmentationResult>
getSegmentsStartingWithGivenColor(const cv::Mat3b &hsv_mat, const UcharPair &hueRange, const UcharPair &saturationRange,
                                  const UcharPair &valueRange, const cv::Vec3b &tolerance, const IntPair &minSize) {
    std::set<IntPair> pixelsToStartWith;
    for(int i = 0; i < hsv_mat.rows; ++i) {
        for(int j = 0; j < hsv_mat.cols; ++j) {
            if(hsv_mat(i, j)[0] >= hueRange.first && hsv_mat(i, j)[0] <= hueRange.second &&
               hsv_mat(i, j)[1] >= saturationRange.first && hsv_mat(i, j)[1] <= saturationRange.second &&
               hsv_mat(i, j)[2] >= valueRange.first && hsv_mat(i, j)[2] <= saturationRange.second) {
                pixelsToStartWith.insert({i, j});
            }
        }
    }
    std::set<IntPair> usedPixels;
    std::vector<SegmentationResult> result;
    while(!pixelsToStartWith.empty()) {
        IntPair startPixel = *pixelsToStartWith.begin();
        pixelsToStartWith.erase(startPixel);
        if(usedPixels.find(startPixel) != usedPixels.end()) {
            continue;
        }
        std::set<IntPair> pixelsToProceed;
        pixelsToProceed.insert(startPixel);
        std::set<IntPair> segmentPixels;
        int minRow = hsv_mat.rows, minCol = hsv_mat.cols, maxRow = 0, maxCol = 0;

        while(!pixelsToProceed.empty()) {

            std::set<IntPair> newPixelsToProceed;
            for(const auto& pixel : pixelsToProceed) {
                segmentPixels.insert(pixel);
                if(pixel.first > maxRow) maxRow = pixel.first;
                if(pixel.first < minRow) minRow = pixel.first;
                if(pixel.second > maxCol) maxCol = pixel.second;
                if(pixel.second < minCol) minCol = pixel.second;
                usedPixels.insert(pixel);

                cv::Vec3b imagePixel = hsv_mat(pixel.first, pixel.second);
                IntPair newPixel = {pixel.first + 1, pixel.second};
                if(
                        pixel.first < hsv_mat.rows - 1 &&
                        isPixelInTolerance(imagePixel, hsv_mat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                    newPixelsToProceed.insert({newPixel.first, newPixel.second});
                }
                newPixel = {pixel.first - 1, pixel.second};
                if(
                        newPixel.first > 1 &&
                        isPixelInTolerance(imagePixel, hsv_mat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                    newPixelsToProceed.insert({newPixel.first, newPixel.second});
                }
                newPixel = {pixel.first, pixel.second + 1};
                if(
                        newPixel.second < hsv_mat.cols - 1 &&
                        isPixelInTolerance(imagePixel, hsv_mat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                    newPixelsToProceed.insert({newPixel.first, newPixel.second});
                }
                newPixel = {pixel.first, pixel.second - 1};
                if(
                        newPixel.second > 1  &&
                        isPixelInTolerance(imagePixel, hsv_mat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                    newPixelsToProceed.insert({newPixel.first, newPixel.second});
                }
            }
            pixelsToProceed = std::move(newPixelsToProceed);
        }
        int width = maxCol - minCol + 1;
        int height = maxRow - minRow + 1;
        if(width >= minSize.first && height >= minSize.second) {
            cv::Mat1d segment = cv::Mat1d(height, width, 0.0);
            for(const auto& pixel : segmentPixels) {
                segment(pixel.first - minRow, pixel.second - minCol) = 1.0;
            }
            cv::Rect roi = {minCol, minRow, width, height};
            result.emplace_back(segment, roi);
        }
    }
    return result;
}

bool isPixelInToleranceGrey(const uchar originalPix, const uchar newPix, const uchar tolerance) {
    return
            abs(originalPix - newPix) <= tolerance;
}

std::vector<SegmentationResult>
getSegmentsStartingWithGivenColorGrey(const cv::Mat3b &mat, const UcharPair &hueRange, const UcharPair &saturationRange,
                                      const UcharPair &valueRange, const uchar tolerance, const IntPair &minSize) {
    cv::Mat3b hsvMat;
    cv::cvtColor(mat, hsvMat, cv::COLOR_BGR2HSV);
    std::set<IntPair> pixelsToStartWith;
    for(int i = 0; i < hsvMat.rows; ++i) {
        for(int j = 0; j < hsvMat.cols; ++j) {
            if(hsvMat(i, j)[0] >= hueRange.first && hsvMat(i, j)[0] <= hueRange.second &&
               hsvMat(i, j)[1] >= saturationRange.first && hsvMat(i, j)[1] <= saturationRange.second &&
               hsvMat(i, j)[2] >= valueRange.first && hsvMat(i, j)[2] <= saturationRange.second) {
                pixelsToStartWith.insert({i, j});
            }
        }
    }
    cv::Mat1b greyMat;
    cv::cvtColor(mat, greyMat, cv::COLOR_BGR2GRAY);
    std::set<IntPair> usedPixels;
    std::vector<SegmentationResult> result;
    while(!pixelsToStartWith.empty()) {
        IntPair startPixel = *pixelsToStartWith.begin();
        pixelsToStartWith.erase(startPixel);
        if(usedPixels.find(startPixel) != usedPixels.end()) {
            continue;
        }
        std::set<IntPair> pixelsToProceed;
        pixelsToProceed.insert(startPixel);
        std::set<IntPair> segmentPixels;
        int minRow = greyMat.rows, minCol = greyMat.cols, maxRow = 0, maxCol = 0;

        while(!pixelsToProceed.empty()) {

            std::set<IntPair> newPixelsToProceed;
            for(const auto& pixel : pixelsToProceed) {
                segmentPixels.insert(pixel);
                if(pixel.first > maxRow) maxRow = pixel.first;
                if(pixel.first < minRow) minRow = pixel.first;
                if(pixel.second > maxCol) maxCol = pixel.second;
                if(pixel.second < minCol) minCol = pixel.second;
                usedPixels.insert(pixel);

                uchar imagePixel = greyMat(pixel.first, pixel.second);
                IntPair newPixel = {pixel.first + 1, pixel.second};
                if(
                        pixel.first < greyMat.rows - 1 &&
                        isPixelInToleranceGrey(imagePixel, greyMat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                    newPixelsToProceed.insert({newPixel.first, newPixel.second});
                }
                newPixel = {pixel.first - 1, pixel.second};
                if(
                        newPixel.first > 1 &&
                        isPixelInToleranceGrey(imagePixel, greyMat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                    newPixelsToProceed.insert({newPixel.first, newPixel.second});
                }
                newPixel = {pixel.first, pixel.second + 1};
                if(
                        newPixel.second < greyMat.cols - 1 &&
                        isPixelInToleranceGrey(imagePixel, greyMat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                    newPixelsToProceed.insert({newPixel.first, newPixel.second});
                }
                newPixel = {pixel.first, pixel.second - 1};
                if(
                        newPixel.second > 1 &&
                        isPixelInToleranceGrey(imagePixel, greyMat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                    newPixelsToProceed.insert({newPixel.first, newPixel.second});
                }
            }
            pixelsToProceed = std::move(newPixelsToProceed);
        }
        int width = maxCol - minCol + 1;
        int height = maxRow - minRow + 1;
        if(width >= minSize.first && height >= minSize.second) {
            cv::Mat1d segment = cv::Mat1d(height, width, 0.0);
            for(const auto& pixel : segmentPixels) {
                segment(pixel.first - minRow, pixel.second - minCol) = 1.0;
            }
            cv::Rect roi = {minCol, minRow, width, height};
            result.emplace_back(segment, roi);
        }
    }
    return result;
}
