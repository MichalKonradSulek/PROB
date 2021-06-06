//
// Created by Michal on 06.06.2021.
//

#include "coefficientCalculation.h"

ImageMoments calculateImageMoments(const cv::Mat1d &image) {
    ImageMoments result;
    for (int imageRow = 0; imageRow < image.rows; ++imageRow) {
        for (int imageCol = 0; imageCol < image.cols; ++imageCol) {
            result.m00 += image(imageRow, imageCol);
            result.m01 += image(imageRow, imageCol) * (imageRow + 0.5);
            result.m10 += image(imageRow, imageCol) * (imageCol + 0.5);
            result.m11 += image(imageRow, imageCol) * (imageCol + 0.5) * (imageRow + 0.5);
            result.m02 += image(imageRow, imageCol) * pow((imageRow + 0.5), 2);
            result.m20 += image(imageRow, imageCol) * pow((imageCol + 0.5), 2);
        }
    }
    return result;
}

double calculateM1(const ImageMoments &moments) {
    const double M02 = moments.m02 - pow(moments.m01, 2) / moments.m00;
    const double M20 = moments.m20 - pow(moments.m10, 2) / moments.m00;
    return (M20 + M02) / pow(moments.m00, 2);
}

double calculateM7(const ImageMoments &moments) {
    const double M11 = moments.m11 - moments.m10 * moments.m01 / moments.m00;
    const double M02 = moments.m02 - pow(moments.m01, 2) / moments.m00;
    const double M20 = moments.m20 - pow(moments.m10, 2) / moments.m00;
    return (M20 * M02 - pow(M11, 2)) / pow(moments.m00, 4);
}
