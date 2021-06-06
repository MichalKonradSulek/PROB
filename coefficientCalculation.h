//
// Created by Michal on 06.06.2021.
//

#ifndef PROB_COEFFICIENTCALCULATION_H
#define PROB_COEFFICIENTCALCULATION_H

#include <opencv2/core/core.hpp>

struct ImageMoments {
    double m00 = 0;
    double m01 = 0;
    double m10 = 0;
    double m11 = 0;
    double m02 = 0;
    double m20 = 0;
};

ImageMoments calculateImageMoments(const cv::Mat1d& image);

double calculateM1(const ImageMoments& moments);

double calculateM7(const ImageMoments& moments);

#endif //PROB_COEFFICIENTCALCULATION_H
