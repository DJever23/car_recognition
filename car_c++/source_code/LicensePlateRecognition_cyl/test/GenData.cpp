#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/types_c.h>

#include<iostream>
#include<vector>

const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

int main(int argc, char** argv)
{
    cv::Mat imgTrainingNumbers;         //输入图像
    cv::Mat imgGrayscale;
    cv::Mat imgBlurred;
    cv::Mat imgThresh;
    cv::Mat imgThreshCopy;

    std::vector<std::vector<cv::Point> > ptContours;        //申明轮廓向量
    std::vector<cv::Vec4i> v4iHierarchy;                    //申明输出轮廓的层次结构

    cv::Mat matClassificationInts;
    cv::Mat matTrainingImagesAsFlattenedFloats;

    //可能的字符
    std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z' };

    if(argc != 2)
    {
        std::cout<<"usage: ./DatasetForTrain img"<<std::endl;
        return 1;
    }
    imgTrainingNumbers = cv::imread(argv[1]);//读取训练图片
    if (imgTrainingNumbers.empty()) 
    {
        std::cout << "error: image not read from file\n\n";
        return 0;
    }

    cv::cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY);//转成灰度图

    cv::GaussianBlur(imgGrayscale, imgBlurred, cv::Size(5, 5), 0);//高斯模糊，0表示自动选择模糊程度                                 

    //图像二值化
    cv::adaptiveThreshold(imgBlurred,
        imgThresh,
        255,                                    //超过门限则为白色
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         //使用高斯寻找门限
        cv::THRESH_BINARY_INV,                  //前景为白色，背景为黑色
        11,                                     //一个像素的周围邻域大小来计算门限
        2);                                     //从均值或加权均值减去的常数

    cv::imshow("imgThresh", imgThresh);         //显示二值化图像

    imgThreshCopy = imgThresh.clone();

    cv::findContours(imgThreshCopy, ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//寻找最外层轮廓

    for (int i = 0; i < ptContours.size(); i++)//对每一个轮廓
    {
        if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA)//如果轮廓大小大于阈值
        {
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                //画框

            cv::rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);      //对每个字符位置画框

            cv::Mat matROI = imgThresh(boundingRect);           //得到框出来的字符图像

            cv::Mat matROIResized;
            cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // 设置为一致的图像大小

            //cv::imshow("matROI", matROI);
            cv::imshow("matROIResized", matROIResized);
            cv::imshow("imgTrainingNumbers", imgTrainingNumbers);       // show training numbers image, this will now have red rectangles drawn on it

            int intChar = cv::waitKey(0);           // get key press

            if (intChar == 27) {        //如果检测到esc，则退出
                return 0;
            }
            else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end())
            {
                matClassificationInts.push_back(intChar);       //将此字符的类别记录下来

                cv::Mat matImageFloat;
                matROIResized.convertTo(matImageFloat, CV_32FC1);//把字符图像转为浮点类型

                cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       //把字符图像转为一行，第一个参数是通道数

                matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);//转为KNN能够支持的图像类型
            }
        }
    }

    std::cout << "training dataset complete\n\n";
    
    //保存到文件
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);
    if (fsClassifications.isOpened() == false) {
        std::cout << "error, unable to open training classifications file, exiting program\n\n";
        return 0;
    }

    fsClassifications << "classifications" << matClassificationInts;
    fsClassifications.release();//关闭文件
    
    //保存训练图片
    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);
    if (fsTrainingImages.isOpened() == false) {
        std::cout << "error, unable to open training images file, exiting program\n\n";
        return 0;
    }

    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;
    fsTrainingImages.release();

    return 0;
}
