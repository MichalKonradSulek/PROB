//
// Created by Michal on 05.06.2021.
//

#include <utility>
#include <vector>
#include <set>
#include <map>
#include <opencv2/imgproc.hpp>

#include "segmentation.h"

std::set<IntPair>
getSegmentationSeedsHsv(const cv::Mat3b &hsvMat, const UcharPair &hueRange, const UcharPair &saturationRange,
                        const UcharPair &valueRange) {
    std::set<IntPair> seeds;
    for(int i = 0; i < hsvMat.rows; ++i) {
        for(int j = 0; j < hsvMat.cols; ++j) {
            if(hsvMat(i, j)[0] >= hueRange.first && hsvMat(i, j)[0] <= hueRange.second &&
               hsvMat(i, j)[1] >= saturationRange.first && hsvMat(i, j)[1] <= saturationRange.second &&
               hsvMat(i, j)[2] >= valueRange.first && hsvMat(i, j)[2] <= saturationRange.second) {
                seeds.insert({i, j});
            }
        }
    }
    return seeds;
}

std::set<IntPair> getSegmentationSeedsBgr(const cv::Mat3b &rgbMat, const UcharPair &bRange, const UcharPair &gRange,
                                          const UcharPair &rRange) {
    std::set<IntPair> seeds;
    for(int i = 0; i < rgbMat.rows; ++i) {
        for(int j = 0; j < rgbMat.cols; ++j) {
            if(rgbMat(i, j)[0] >= bRange.first && rgbMat(i, j)[0] <= bRange.second &&
               rgbMat(i, j)[1] >= gRange.first && rgbMat(i, j)[1] <= gRange.second &&
               rgbMat(i, j)[2] >= rRange.first && rgbMat(i, j)[2] <= gRange.second) {
                seeds.insert({i, j});
            }
        }
    }
    return seeds;
}

bool isPixelInToleranceGrey(const uchar originalPix, const uchar newPix, const uchar tolerance) {
    return
            abs(originalPix - newPix) <= tolerance;
}


std::vector<SegmentationResult>
getSegments8DirGrey(const cv::Mat1b &greyMat, std::set<IntPair> seeds, const uchar tolerance, const IntPair &minSize) {
    std::set<IntPair> usedPixels;
    std::vector<SegmentationResult> result;
    while (!seeds.empty()) {
        IntPair startPixel = *seeds.begin();
        seeds.erase(startPixel);
        if (usedPixels.find(startPixel) != usedPixels.end()) {
            continue;
        }
        std::set<IntPair> pixelsToProceed;
        pixelsToProceed.insert(startPixel);
        std::set<IntPair> segmentPixels;
        int minRow = greyMat.rows, minCol = greyMat.cols, maxRow = 0, maxCol = 0;

        while (!pixelsToProceed.empty()) {

            std::set<IntPair> newPixelsToProceed;
            for (const auto &pixel : pixelsToProceed) {
                segmentPixels.insert(pixel);
                if (pixel.first > maxRow) maxRow = pixel.first;
                if (pixel.first < minRow) minRow = pixel.first;
                if (pixel.second > maxCol) maxCol = pixel.second;
                if (pixel.second < minCol) minCol = pixel.second;
                usedPixels.insert(pixel);

                std::set<IntPair> potentialPixelsToProceed = {
                        {pixel.first - 1, pixel.second - 1},
                        {pixel.first - 1, pixel.second},
                        {pixel.first - 1, pixel.second + 1},
                        {pixel.first,     pixel.second - 1},
                        {pixel.first,     pixel.second + 1},
                        {pixel.first + 1, pixel.second - 1},
                        {pixel.first + 1, pixel.second},
                        {pixel.first + 1, pixel.second + 1},
                };

                if (pixel.first == 0) {
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second - 1});
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second});
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second + 1});
                }
                if (pixel.first == greyMat.rows - 1) {
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second - 1});
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second});
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second + 1});
                }
                if (pixel.second == 0) {
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second - 1});
                    potentialPixelsToProceed.erase({pixel.first, pixel.second - 1});
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second - 1});
                }
                if (pixel.second == greyMat.cols - 1) {
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second + 1});
                    potentialPixelsToProceed.erase({pixel.first, pixel.second + 1});
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second + 1});
                }

                uchar imagePixel = greyMat(pixel.first, pixel.second);
                for (const auto &newPixel : potentialPixelsToProceed) {
                    if (isPixelInToleranceGrey(imagePixel, greyMat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels.find({newPixel.first, newPixel.second}) == usedPixels.end()) {
                        newPixelsToProceed.insert({newPixel.first, newPixel.second});
                    }
                }
            }
            pixelsToProceed = std::move(newPixelsToProceed);
        }
        int width = maxCol - minCol + 1;
        int height = maxRow - minRow + 1;
        if (width >= minSize.first && height >= minSize.second) {
            cv::Mat1d segment = cv::Mat1d(height, width, 0.0);
            for (const auto &pixel : segmentPixels) {
                segment(pixel.first - minRow, pixel.second - minCol) = 1.0;
            }
            cv::Rect roi = {minCol, minRow, width, height};
            result.emplace_back(segment, roi);
        }
    }
    return result;
}

std::vector<SegmentationResult>
getSegments8DirGrey2(const cv::Mat1b &greyMat, std::set<IntPair> seeds, const uchar tolerance, const IntPair &minSize) {
    cv::Mat1b usedPixels = cv::Mat1b(greyMat.rows, greyMat.cols, (uchar)0);
    std::vector<SegmentationResult> result;
    while (!seeds.empty()) {
        IntPair startPixel = *seeds.begin();
        seeds.erase(startPixel);
        if (usedPixels(startPixel.first, startPixel.second) != 0) {
            continue;
        }
        std::set<IntPair> pixelsToProceed;
        pixelsToProceed.insert(startPixel);
        std::set<IntPair> segmentPixels;
        int minRow = greyMat.rows, minCol = greyMat.cols, maxRow = 0, maxCol = 0;

        while (!pixelsToProceed.empty()) {

            std::set<IntPair> newPixelsToProceed;
            for (const auto &pixel : pixelsToProceed) {
                segmentPixels.insert(pixel);
                if (pixel.first > maxRow) maxRow = pixel.first;
                if (pixel.first < minRow) minRow = pixel.first;
                if (pixel.second > maxCol) maxCol = pixel.second;
                if (pixel.second < minCol) minCol = pixel.second;
                usedPixels(pixel.first, pixel.second) = 1;

                std::set<IntPair> potentialPixelsToProceed = {
                        {pixel.first - 1, pixel.second - 1},
                        {pixel.first - 1, pixel.second},
                        {pixel.first - 1, pixel.second + 1},
                        {pixel.first,     pixel.second - 1},
                        {pixel.first,     pixel.second + 1},
                        {pixel.first + 1, pixel.second - 1},
                        {pixel.first + 1, pixel.second},
                        {pixel.first + 1, pixel.second + 1},
                };

                if (pixel.first == 0) {
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second - 1});
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second});
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second + 1});
                }
                if (pixel.first == greyMat.rows - 1) {
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second - 1});
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second});
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second + 1});
                }
                if (pixel.second == 0) {
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second - 1});
                    potentialPixelsToProceed.erase({pixel.first, pixel.second - 1});
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second - 1});
                }
                if (pixel.second == greyMat.cols - 1) {
                    potentialPixelsToProceed.erase({pixel.first - 1, pixel.second + 1});
                    potentialPixelsToProceed.erase({pixel.first, pixel.second + 1});
                    potentialPixelsToProceed.erase({pixel.first + 1, pixel.second + 1});
                }

                uchar imagePixel = greyMat(pixel.first, pixel.second);
                for (const auto &newPixel : potentialPixelsToProceed) {
                    if (isPixelInToleranceGrey(imagePixel, greyMat(newPixel.first, newPixel.second), tolerance) &&
                        usedPixels(newPixel.first, newPixel.second) == 0) {
                        newPixelsToProceed.insert({newPixel.first, newPixel.second});
                    }
                }
            }
            pixelsToProceed = std::move(newPixelsToProceed);
        }
        int width = maxCol - minCol + 1;
        int height = maxRow - minRow + 1;
        if (width >= minSize.first && height >= minSize.second) {
            cv::Mat1d segment = cv::Mat1d(height, width, 0.0);
            for (const auto &pixel : segmentPixels) {
                segment(pixel.first - minRow, pixel.second - minCol) = 1.0;
            }
            cv::Rect roi = {minCol, minRow, width, height};
            result.emplace_back(segment, roi);
        }
    }
    return result;
}
