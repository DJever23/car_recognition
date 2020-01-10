#include "PossibleChar.h"

PossibleChar::PossibleChar(std::vector<cv::Point> _contour) {
    contour = _contour;

    boundingRect = cv::boundingRect(contour);//计算轮廓的垂直边界最小矩形

    intCenterX = (boundingRect.x + boundingRect.x + boundingRect.width) / 2;
    intCenterY = (boundingRect.y + boundingRect.y + boundingRect.height) / 2;

    dblDiagonalSize = sqrt(pow(boundingRect.width, 2) + pow(boundingRect.height, 2));  //对角线长度

    dblAspectRatio = (float)boundingRect.width / (float)boundingRect.height;   //长宽比

}
