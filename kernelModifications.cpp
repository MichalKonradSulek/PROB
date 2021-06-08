//
// Created by Michal on 06.06.2021.
//

#include "kernelModifications.h"

uchar MaxPixelValue::operator()(const cv::Mat1b &area) const {
    uchar maxVal = 0;
    for(const auto& pixel : area) {
        if(pixel > maxVal) maxVal = pixel;
    }
    return maxVal;
}

uchar MinPixelValue::operator()(const cv::Mat1b &area) const {
    uchar minVal = 255;
    for(const auto& pixel : area) {
        if(pixel < minVal) minVal = pixel;
    }
    return minVal;
}

uchar MedianValue::operator()(const cv::Mat1b &area) const {
    CV_Assert(area.total() > 0);
    int rank = area.total() / 2; // NOLINT(cppcoreguidelines-narrowing-conversions)
    std::vector<std::pair<uchar, int>> arrayOfPixels;
    arrayOfPixels.reserve(area.total());
    for (int i = 0; i < area.total(); ++i) {
        arrayOfPixels.emplace_back(area(i), i);
    }
    std::sort(arrayOfPixels.begin(), arrayOfPixels.end());
    return arrayOfPixels.at(rank).first;
}

cv::Vec3b getPixelWithGivenRank(const cv::Mat_<cv::Vec3b>& area, const unsigned rank) {
    std::vector<std::pair<unsigned, unsigned>> arrayOfPixels;
    arrayOfPixels.reserve(((unsigned)area.rows) * area.cols);
    for (unsigned i = 0; i < area.rows; ++i) {
        for (unsigned j = 0; j < area.cols; ++j) {
            unsigned index = i * area.cols + j;
            unsigned luminosity = area(i, j)[0] + area(i, j)[1] + area(i, j)[2];
            arrayOfPixels.emplace_back(luminosity, index);
        }
    }
    std::sort(arrayOfPixels.begin(), arrayOfPixels.end());
    unsigned index = arrayOfPixels.at(rank).second;
    return area(index / 7, index % 7);
}



cv::Mat1b kernelFilter(const cv::Mat1b &mat, int kernelSize, const KernelFilter& kernelFilter) {
    CV_Assert(kernelSize % 2 == 1 && kernelSize > 0);
    const int kernelHalfSize = kernelSize / 2;
    cv::Mat1b result = cv::Mat1b(mat.rows, mat.cols);
    for (int row = kernelHalfSize; row < mat.rows - kernelHalfSize; ++row) {
        for (int col = kernelHalfSize; col < mat.cols - kernelHalfSize; ++col) {
            cv::Mat1b area = cv::Mat1b(mat, {col - kernelHalfSize, row - kernelHalfSize, kernelSize, kernelSize});
            uchar newValue = kernelFilter(area);
            result(row, col) = newValue;
        }
    }
    return result;
}
