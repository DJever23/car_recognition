#include "Preprocess.h"
#include<iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/types_c.h>

void preprocess(cv::Mat &imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh) 
{
    imgGrayscale = extractValue(imgOriginal);                           // 从HSV空间，提取灰度图

    cv::Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);       // 增加对比度

    cv::Mat imgBlurred;
    
    cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);          // gaussian blur


	//cv::imshow("效果图1", imgMaxContrastGrayscale);
	//cv::imshow("效果图2", imgBlurred);

	// 自适应阈值二值化
    cv::adaptiveThreshold(imgBlurred, imgThresh, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);

	//cv::imshow("效果图3", imgThresh);
	//cvWaitKey();
}

cv::Mat extractValue(cv::Mat &imgOriginal) {
    cv::Mat imgHSV;
    std::vector<cv::Mat> vectorOfHSVImages;
    cv::Mat imgValue;

    cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);  //从RGB转换到HSV颜色空间

	//cv::imshow("效果图1", imgHSV);

    cv::split(imgHSV, vectorOfHSVImages);  //分为H,S,V三个通道

	//cv::imshow("效果图2", vectorOfHSVImages[0]);
	//cv::imshow("效果图3", vectorOfHSVImages[1]);
	//cv::imshow("效果图4", vectorOfHSVImages[2]);

	//cvWaitKey();

    imgValue = vectorOfHSVImages[2];  //vectorOfHSVImages[2]  从视觉效果来看是灰度图  V代表明度

    return imgValue;
}

cv::Mat maximizeContrast(cv::Mat &imgGrayscale) {
    cv::Mat imgTopHat;
    cv::Mat imgBlackHat;
    cv::Mat imgGrayscalePlusTopHat;
    cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

    cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));  //CV_SHAPE_RECT, 矩形核
    //高级形态学变化
    cv::morphologyEx(imgGrayscale, imgTopHat, CV_MOP_TOPHAT, structuringElement);    //CV_MOP_TOPHAT，礼帽  突出原图像中比周围亮的区域 
    cv::morphologyEx(imgGrayscale, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);//CV_MOP_BLACKHAT，黑帽   突出原图像中比周围暗的区域

    //cv::imshow("效果图4", imgGrayscale);
    //cv::imshow("效果图5", imgTopHat);
    //cv::imshow("效果图6", imgBlackHat);

    imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
    imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

	//cv::imshow("效果图7", imgGrayscalePlusTopHat);
	//cv::imshow("效果图8", imgGrayscalePlusTopHatMinusBlackHat);

	//cvWaitKey();

    return imgGrayscalePlusTopHatMinusBlackHat;
}


