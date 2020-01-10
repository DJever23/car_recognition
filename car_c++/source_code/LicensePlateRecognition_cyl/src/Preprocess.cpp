#include "Preprocess.h"
#include<iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/types_c.h>

void preprocess(cv::Mat &imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh) 
{
    imgGrayscale = extractValue(imgOriginal);                           // ��HSV�ռ䣬��ȡ�Ҷ�ͼ

    cv::Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);       // ���ӶԱȶ�

    cv::Mat imgBlurred;
    
    cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);          // gaussian blur


	//cv::imshow("Ч��ͼ1", imgMaxContrastGrayscale);
	//cv::imshow("Ч��ͼ2", imgBlurred);

	// ����Ӧ��ֵ��ֵ��
    cv::adaptiveThreshold(imgBlurred, imgThresh, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);

	//cv::imshow("Ч��ͼ3", imgThresh);
	//cvWaitKey();
}

cv::Mat extractValue(cv::Mat &imgOriginal) {
    cv::Mat imgHSV;
    std::vector<cv::Mat> vectorOfHSVImages;
    cv::Mat imgValue;

    cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);  //��RGBת����HSV��ɫ�ռ�

	//cv::imshow("Ч��ͼ1", imgHSV);

    cv::split(imgHSV, vectorOfHSVImages);  //��ΪH,S,V����ͨ��

	//cv::imshow("Ч��ͼ2", vectorOfHSVImages[0]);
	//cv::imshow("Ч��ͼ3", vectorOfHSVImages[1]);
	//cv::imshow("Ч��ͼ4", vectorOfHSVImages[2]);

	//cvWaitKey();

    imgValue = vectorOfHSVImages[2];  //vectorOfHSVImages[2]  ���Ӿ�Ч�������ǻҶ�ͼ  V��������

    return imgValue;
}

cv::Mat maximizeContrast(cv::Mat &imgGrayscale) {
    cv::Mat imgTopHat;
    cv::Mat imgBlackHat;
    cv::Mat imgGrayscalePlusTopHat;
    cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

    cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));  //CV_SHAPE_RECT, ���κ�
    //�߼���̬ѧ�仯
    cv::morphologyEx(imgGrayscale, imgTopHat, CV_MOP_TOPHAT, structuringElement);    //CV_MOP_TOPHAT����ñ  ͻ��ԭͼ���б���Χ�������� 
    cv::morphologyEx(imgGrayscale, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);//CV_MOP_BLACKHAT����ñ   ͻ��ԭͼ���б���Χ��������

    //cv::imshow("Ч��ͼ4", imgGrayscale);
    //cv::imshow("Ч��ͼ5", imgTopHat);
    //cv::imshow("Ч��ͼ6", imgBlackHat);

    imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
    imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

	//cv::imshow("Ч��ͼ7", imgGrayscalePlusTopHat);
	//cv::imshow("Ч��ͼ8", imgGrayscalePlusTopHatMinusBlackHat);

	//cvWaitKey();

    return imgGrayscalePlusTopHatMinusBlackHat;
}


