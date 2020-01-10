#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>

#include "DetectPlates.h"
#include "PossiblePlate.h"
#include "DetectChars.h"

void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);      //scalar是将图像设置成单一灰度和颜色，此为黑色
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

int main(int argc, char** argv)
{
    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();//读取文件，进行KNN训练
    
    if (blnKNNTrainingSuccessful == false) {
        std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
        return 0;
    }
    
    if(argc != 2)
    {
        std::cout<<"usage: ./LicensePlate_Recognition img"<<std::endl;
        return 1;
    }
    
    cv::Mat imgOriginalScene;           //输入图像
    imgOriginalScene = cv::imread(argv[1]);
    
    if (imgOriginalScene.empty()) {
        std::cout << "error: image not read from file\n\n";
        return(0);
    }
    
    std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);          //检测平面
    
    vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                               //检测在平面上的字符
    
    cv::imshow("imgOriginalScene", imgOriginalScene);
    
    if (vectorOfPossiblePlates.empty()) {                                                               //如果没有检测到平面
        std::cout << std::endl << "no license plates were detected" << std::endl;
    }
    else {
        std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);//按照字符数量从大到小对平面排序
        
        PossiblePlate licPlate = vectorOfPossiblePlates.front();//假设字符数量最多的平面是真正的车牌平面
        
        cv::imshow("imgPlate", licPlate.imgPlate);            // 显示车牌平面和二值图
        cv::imshow("imgThresh", licPlate.imgThresh);
        
        if (licPlate.strChars.length() == 0) {                                                      //如果没有找到字符
            std::cout << std::endl << "no characters were detected" << std::endl << std::endl;
            return 0;
        }
        
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate);                // 对车牌画出红框
        
        std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;     // 输出字符
        std::cout << std::endl << "-----------------------------------------" << std::endl;
        
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);              // 在图片上写出车牌
        
        cv::imshow("imgOriginalScene", imgOriginalScene);
        
        cv::imwrite("imgOriginalScene.png", imgOriginalScene);
    }
    
    cv::waitKey(0);
    
    return 0;
}


void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate)
{
    cv::Point2f p2fRectPoints[4];
    
    licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);            // 得到斜的方框的四个顶点
    
    for (int i = 0; i < 4; i++) {                                       // 画4条红线
        cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
    }
}


void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate)
{
    cv::Point ptCenterOfTextArea;                   // 字符串中心的位置
    cv::Point ptLowerLeftTextOrigin;                // 字符串左下角的位置
    
    int intFontFace = cv::FONT_HERSHEY_SIMPLEX;                              // 选择纯简字体 
    double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;            // 根据平面高度的字体大小 
    int intFontThickness = (int)std::round(dblFontScale * 1.5);             // 根据字体比例确定字体粗细 
    int intBaseline = 0;
    
    cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);
    
    ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         // 文字中心位置的x坐标和平面一样
    
    if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      // 如果车牌是在图片的3/4之上
        // 把字符串写在车牌下面
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }
    else {
        // 否则写在车牌上面
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }
    
    ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));
    ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));
    
    // 在图片中写出字符串
    cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}


