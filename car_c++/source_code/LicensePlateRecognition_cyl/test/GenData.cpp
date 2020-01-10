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
    cv::Mat imgTrainingNumbers;         //����ͼ��
    cv::Mat imgGrayscale;
    cv::Mat imgBlurred;
    cv::Mat imgThresh;
    cv::Mat imgThreshCopy;

    std::vector<std::vector<cv::Point> > ptContours;        //������������
    std::vector<cv::Vec4i> v4iHierarchy;                    //������������Ĳ�νṹ

    cv::Mat matClassificationInts;
    cv::Mat matTrainingImagesAsFlattenedFloats;

    //���ܵ��ַ�
    std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z' };

    if(argc != 2)
    {
        std::cout<<"usage: ./DatasetForTrain img"<<std::endl;
        return 1;
    }
    imgTrainingNumbers = cv::imread(argv[1]);//��ȡѵ��ͼƬ
    if (imgTrainingNumbers.empty()) 
    {
        std::cout << "error: image not read from file\n\n";
        return 0;
    }

    cv::cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY);//ת�ɻҶ�ͼ

    cv::GaussianBlur(imgGrayscale, imgBlurred, cv::Size(5, 5), 0);//��˹ģ����0��ʾ�Զ�ѡ��ģ���̶�                                 

    //ͼ���ֵ��
    cv::adaptiveThreshold(imgBlurred,
        imgThresh,
        255,                                    //����������Ϊ��ɫ
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         //ʹ�ø�˹Ѱ������
        cv::THRESH_BINARY_INV,                  //ǰ��Ϊ��ɫ������Ϊ��ɫ
        11,                                     //һ�����ص���Χ�����С����������
        2);                                     //�Ӿ�ֵ���Ȩ��ֵ��ȥ�ĳ���

    cv::imshow("imgThresh", imgThresh);         //��ʾ��ֵ��ͼ��

    imgThreshCopy = imgThresh.clone();

    cv::findContours(imgThreshCopy, ptContours, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//Ѱ�����������

    for (int i = 0; i < ptContours.size(); i++)//��ÿһ������
    {
        if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA)//���������С������ֵ
        {
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                //����

            cv::rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);      //��ÿ���ַ�λ�û���

            cv::Mat matROI = imgThresh(boundingRect);           //�õ���������ַ�ͼ��

            cv::Mat matROIResized;
            cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // ����Ϊһ�µ�ͼ���С

            //cv::imshow("matROI", matROI);
            cv::imshow("matROIResized", matROIResized);
            cv::imshow("imgTrainingNumbers", imgTrainingNumbers);       // show training numbers image, this will now have red rectangles drawn on it

            int intChar = cv::waitKey(0);           // get key press

            if (intChar == 27) {        //�����⵽esc�����˳�
                return 0;
            }
            else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end())
            {
                matClassificationInts.push_back(intChar);       //�����ַ�������¼����

                cv::Mat matImageFloat;
                matROIResized.convertTo(matImageFloat, CV_32FC1);//���ַ�ͼ��תΪ��������

                cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       //���ַ�ͼ��תΪһ�У���һ��������ͨ����

                matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);//תΪKNN�ܹ�֧�ֵ�ͼ������
            }
        }
    }

    std::cout << "training dataset complete\n\n";
    
    //���浽�ļ�
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);
    if (fsClassifications.isOpened() == false) {
        std::cout << "error, unable to open training classifications file, exiting program\n\n";
        return 0;
    }

    fsClassifications << "classifications" << matClassificationInts;
    fsClassifications.release();//�ر��ļ�
    
    //����ѵ��ͼƬ
    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);
    if (fsTrainingImages.isOpened() == false) {
        std::cout << "error, unable to open training images file, exiting program\n\n";
        return 0;
    }

    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;
    fsTrainingImages.release();

    return 0;
}
