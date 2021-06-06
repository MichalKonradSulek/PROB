//
// Created by Michal on 06.06.2021.
//

#ifndef PROB_KERNELMODIFICATIONS_H
#define PROB_KERNELMODIFICATIONS_H


#include <opencv2/core/core.hpp>

class KernelFilter {
public:
    virtual uchar operator()(const cv::Mat1b& area) const = 0;
};

class MaxPixelValue : public KernelFilter {
    uchar operator()(const cv::Mat1b& area) const override;
};

class MinPixelValue : public KernelFilter {
    uchar operator()(const cv::Mat1b& area) const override;
};

class MedianValue : public KernelFilter {
    uchar operator()(const cv::Mat1b& area) const override;
};

cv::Mat1b maxFilter(const cv::Mat1b& mat, int kernelSize, const KernelFilter& kernelFilter);


#endif //PROB_KERNELMODIFICATIONS_H
