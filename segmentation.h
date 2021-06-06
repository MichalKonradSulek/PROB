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

std::set<IntPair> getSegmentationSeedsHsv(const cv::Mat3b &hsv_mat, const UcharPair &hueRange,
                                          const UcharPair &saturationRange, const UcharPair &valueRange);

std::vector<SegmentationResult> getSegments(const cv::Mat1d& mat, int minWidth=0, int minHeight=0);

std::vector<SegmentationResult>
getSegmentsStartingWithGivenColor(const cv::Mat3b &hsv_mat, const UcharPair &hueRange, const UcharPair &saturationRange,
                                  const UcharPair &valueRange, const cv::Vec3b &tolerance = {5, 5, 5},
                                  const IntPair &minSize = {1, 1});

std::vector<SegmentationResult>
getSegmentsStartingWithGivenColorGrey(const cv::Mat3b &mat, const UcharPair &hueRange, const UcharPair &saturationRange,
                                      const UcharPair &valueRange, uchar tolerance, const IntPair &minSize);


#endif //PROB_SEGMENTATION_H