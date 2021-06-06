//
// Created by Michal on 05.06.2021.
//

#ifndef PROB_PIXELMODIFICATIONS_H
#define PROB_PIXELMODIFICATIONS_H

#include <opencv2/core/core.hpp>

#include "typedefs.h"

cv::Mat3b filterPixelsWithGivenValues(const cv::Mat3b& mat, const UcharPair& hueRange, const UcharPair& saturationRange,
                                      const UcharPair& valueRange);

cv::Mat1d getOnlyPixelsWithGivenValues(const cv::Mat3b& mat, const UcharPair& hueRange, const UcharPair& saturationRange,
                                       const UcharPair& valueRange);

void levelHistogram(cv::Mat3b& hsvMat);

void levelHistogramRgb(cv::Mat3b& mat);

void changeContrast(cv::Mat3b& mat, double coefficient);


#endif //PROB_PIXELMODIFICATIONS_H