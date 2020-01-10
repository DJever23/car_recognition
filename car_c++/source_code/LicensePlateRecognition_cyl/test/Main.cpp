#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>

#include "DetectPlates.h"
#include "PossiblePlate.h"
#include "DetectChars.h"

void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate);

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);      //scalar�ǽ�ͼ�����óɵ�һ�ҶȺ���ɫ����Ϊ��ɫ
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

int main(int argc, char** argv)
{
    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();//��ȡ�ļ�������KNNѵ��
    
    if (blnKNNTrainingSuccessful == false) {
        std::cout << std::endl << std::endl << "error: error: KNN traning was not successful" << std::endl << std::endl;
        return 0;
    }
    
    if(argc != 2)
    {
        std::cout<<"usage: ./LicensePlate_Recognition img"<<std::endl;
        return 1;
    }
    
    cv::Mat imgOriginalScene;           //����ͼ��
    imgOriginalScene = cv::imread(argv[1]);
    
    if (imgOriginalScene.empty()) {
        std::cout << "error: image not read from file\n\n";
        return(0);
    }
    
    std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);          //���ƽ��
    
    vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                               //�����ƽ���ϵ��ַ�
    
    cv::imshow("imgOriginalScene", imgOriginalScene);
    
    if (vectorOfPossiblePlates.empty()) {                                                               //���û�м�⵽ƽ��
        std::cout << std::endl << "no license plates were detected" << std::endl;
    }
    else {
        std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);//�����ַ������Ӵ�С��ƽ������
        
        PossiblePlate licPlate = vectorOfPossiblePlates.front();//�����ַ���������ƽ���������ĳ���ƽ��
        
        cv::imshow("imgPlate", licPlate.imgPlate);            // ��ʾ����ƽ��Ͷ�ֵͼ
        cv::imshow("imgThresh", licPlate.imgThresh);
        
        if (licPlate.strChars.length() == 0) {                                                      //���û���ҵ��ַ�
            std::cout << std::endl << "no characters were detected" << std::endl << std::endl;
            return 0;
        }
        
        drawRedRectangleAroundPlate(imgOriginalScene, licPlate);                // �Գ��ƻ������
        
        std::cout << std::endl << "license plate read from image = " << licPlate.strChars << std::endl;     // ����ַ�
        std::cout << std::endl << "-----------------------------------------" << std::endl;
        
        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);              // ��ͼƬ��д������
        
        cv::imshow("imgOriginalScene", imgOriginalScene);
        
        cv::imwrite("imgOriginalScene.png", imgOriginalScene);
    }
    
    cv::waitKey(0);
    
    return 0;
}


void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate)
{
    cv::Point2f p2fRectPoints[4];
    
    licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);            // �õ�б�ķ�����ĸ�����
    
    for (int i = 0; i < 4; i++) {                                       // ��4������
        cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
    }
}


void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate)
{
    cv::Point ptCenterOfTextArea;                   // �ַ������ĵ�λ��
    cv::Point ptLowerLeftTextOrigin;                // �ַ������½ǵ�λ��
    
    int intFontFace = cv::FONT_HERSHEY_SIMPLEX;                              // ѡ�񴿼����� 
    double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;            // ����ƽ��߶ȵ������С 
    int intFontThickness = (int)std::round(dblFontScale * 1.5);             // �����������ȷ�������ϸ 
    int intBaseline = 0;
    
    cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);
    
    ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         // ��������λ�õ�x�����ƽ��һ��
    
    if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      // �����������ͼƬ��3/4֮��
        // ���ַ���д�ڳ�������
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }
    else {
        // ����д�ڳ�������
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }
    
    ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));
    ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));
    
    // ��ͼƬ��д���ַ���
    cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}


