//
// Created by Michal on 05.06.2021.
//

#ifndef PROB_SEGMENTATION_H
#define PROB_SEGMENTATION_H

#include <opencv2/core/core.hpp>

#include "typedefs.h"

struct SegmentationResult {
    cv::Mat1d object;
    cv::Rect roi;

    SegmentationResult(cv::Mat1d object, cv::Rect roi) : object(std::move(object)), roi(std::move(roi)) {}
};

std::set<IntPair> getSegmentationSeedsHsv(const cv::Mat3b &hsvMat, const UcharPair &hueRange,
                                          const UcharPair &saturationRange, const UcharPair &valueRange);

std::set<IntPair> getSegmentationSeedsBgr(const cv::Mat3b &rgbMat, const UcharPair &bRange, const UcharPair &gRange,
                                          const UcharPair &rRange);

std::vector<SegmentationResult>
getSegments8DirGrey(const cv::Mat1b &greyMat, std::set<IntPair> seeds, uchar tolerance,
                    const IntPair &minSize);

std::vector<SegmentationResult>
getSegments8DirGrey2(const cv::Mat1b &greyMat, std::set<IntPair> seeds, uchar tolerance,
                    const IntPair &minSize);

#endif //PROB_SEGMENTATION_H
