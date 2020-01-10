#include "DetectChars.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/types_c.h>

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);      //scalar�ǽ�ͼ�����óɵ�һ�ҶȺ���ɫ����Ϊ��ɫ
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// global variables
cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();

//#define SHOW_STEPS

bool loadKNNDataAndTrainKNN(void)
{
    //��ȡ���
    cv::Mat matClassificationInts;
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);
    if (fsClassifications.isOpened() == false) {
        std::cout << "error, unable to open training classifications file, exiting program\n\n";
        return false;
    }
    fsClassifications["classifications"] >> matClassificationInts;
    fsClassifications.release();

    //��ȡ����ѵ����ͼ�����
    cv::Mat matTrainingImagesAsFlattenedFloats;
    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);
    if (fsTrainingImages.isOpened() == false) {
        std::cout << "error, unable to open training images file, exiting program\n\n";
        return false;
    }
    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;
    fsTrainingImages.release();

    //ѵ��
    kNearest->setDefaultK(1);

    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    return true;
}

std::vector<PossiblePlate> detectCharsInPlates(std::vector<PossiblePlate> &vectorOfPossiblePlates)
{
    int intPlateCounter = 0;				// this is only for showing steps
    cv::Mat imgContours;
    std::vector<std::vector<cv::Point> > contours;
    cv::RNG rng;

    if (vectorOfPossiblePlates.empty()) {
        return(vectorOfPossiblePlates);
    }
    
    // ��ʱ������һ��ƽ��
    for (auto &possiblePlate : vectorOfPossiblePlates)
    {
        preprocess(possiblePlate.imgPlate, possiblePlate.imgGrayscale, possiblePlate.imgThresh);        // Ԥ����õ��Ҷ�ͼ�Ͷ�ֵ��ͼ

#ifdef SHOW_STEPS
        //cv::imshow("5a", possiblePlate.imgPlate);
        std::string filename;
        filename = filename+ "5a_" + (char)('0' + intPlateCounter) + ".jpg";
        cv::imwrite(filename, possiblePlate.imgPlate);
        //cv::imshow("5b", possiblePlate.imgGrayscale);
        std::string filename1;
        filename1 = filename1+ "5b_" + (char)('0' + intPlateCounter) + ".jpg";
        cv::imwrite(filename1, possiblePlate.imgGrayscale);
        //cv::imshow("5c", possiblePlate.imgThresh);
        std::string filename2;
        filename2 = filename2+ "5c_" + (char)('0' + intPlateCounter) + ".jpg";
        cv::imwrite(filename2, possiblePlate.imgThresh);
#endif	// SHOW_STEPS

        // �Ŵ�1.6����Ϊ�˸��õ�ʶ��
        cv::resize(possiblePlate.imgThresh, possiblePlate.imgThresh, cv::Size(), 1.6, 1.6);

		//cv::imshow("6c", possiblePlate.imgThresh);

        // �ٴζ�ֵ���������Ҷ��ж�ģ������
        //cv::threshold(possiblePlate.imgThresh, possiblePlate.imgThresh, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cv::threshold(possiblePlate.imgThresh, possiblePlate.imgThresh, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);	

	    //cv::imshow("7a", possiblePlate.imgThresh);
		//cv::waitKey();

#ifdef SHOW_STEPS
        //cv::imshow("5d", possiblePlate.imgThresh);
        std::string filename4;
        filename4 = filename4+ "5d_" + (char)('0' + intPlateCounter) + ".jpg";
        cv::imwrite(filename4, possiblePlate.imgThresh);
#endif	// SHOW_STEPS

        // Ѱ����ƽ�������п��ܵ��ַ�
        //����Ѱ������������Ȼ�����������ַ�������
        std::vector<PossibleChar> vectorOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh);

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK); //����ȫ��ͼ
        contours.clear();

        for (auto &possibleChar : vectorOfPossibleCharsInPlate) {
            contours.push_back(possibleChar.contour);
        }

        cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);

        //cv::imshow("6", imgContours);
        std::string filename5;
        filename5 = filename5+ "6_" + (char)('0' + intPlateCounter) + ".jpg";
        cv::imwrite(filename5, imgContours);
#endif	// SHOW_STEPS

        // �����е��ַ���������
        std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInPlate = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInPlate);

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK); //����ȫ��ͼ

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {
            int intRandomBlue = rng.uniform(0, 256);
            int intRandomGreen = rng.uniform(0, 256);
            int intRandomRed = rng.uniform(0, 256);
            
            contours.clear();

            for (auto &matchingChar : vectorOfMatchingChars) {
                contours.push_back(matchingChar.contour);
            }
            cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
        }
        //cv::imshow("7", imgContours);
        std::string filename6;
        filename6 = filename6+ "7_" + (char)('0' + intPlateCounter) + ".jpg";
        cv::imwrite(filename6, imgContours);
#endif	// SHOW_STEPS

        if (vectorOfVectorsOfMatchingCharsInPlate.size() == 0) {                // �����ƽ����û���ַ���������
#ifdef SHOW_STEPS
            std::cout << "chars found in plate number " << intPlateCounter << " = (none), click on any image and press a key to continue . . ." << std::endl;
            intPlateCounter++;
            //cv::destroyWindow("8");
            //cv::destroyWindow("9");
            //cv::destroyWindow("10");
            //cv::waitKey(0);
#endif	// SHOW_STEPS
            possiblePlate.strChars = "";            // ����ƽ���ʾ���ַ�����Ϊ��
            continue;                               // ���أ�����һ��ƽ���ٲ���
        }

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {                                         // ��ÿһ���ַ�������
            std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);      // ���մ�����λ������
            vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                     // �����ص����ַ�
        }

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK); //����ȫ��ͼ

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {
            int intRandomBlue = rng.uniform(0, 256);
            int intRandomGreen = rng.uniform(0, 256);
            int intRandomRed = rng.uniform(0, 256);

            contours.clear();

            for (auto &matchingChar : vectorOfMatchingChars) {
                contours.push_back(matchingChar.contour);
            }
            cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
        }
        //cv::imshow("8", imgContours);
        std::string filename7;
        filename7 = filename7+ "8_" + (char)('0' + intPlateCounter) + ".jpg";
        cv::imwrite(filename7, imgContours);
#endif	// SHOW_STEPS

        // ��һ��ƽ���У���Ϊ��������ַ��������ַ���������ʵ�ʵ��ַ�������
        unsigned int intLenOfLongestVectorOfChars = 0;
        unsigned int intIndexOfLongestVectorOfChars = 0;
        // �����е��ַ������������Ѱ��ӵ������ַ������������ַ�������������
        for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsInPlate.size(); i++) {
            if (vectorOfVectorsOfMatchingCharsInPlate[i].size() > intLenOfLongestVectorOfChars) {
                intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsInPlate[i].size();
                intIndexOfLongestVectorOfChars = i;
            }
        }
        // ��һ��ƽ���У���Ϊ��������ַ��������ַ���������ʵ�ʵ��ַ�������
        std::vector<PossibleChar> longestVectorOfMatchingCharsInPlate = vectorOfVectorsOfMatchingCharsInPlate[intIndexOfLongestVectorOfChars];

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK); //����ȫ��ͼ

        contours.clear();

        for (auto &matchingChar : longestVectorOfMatchingCharsInPlate) {
            contours.push_back(matchingChar.contour);
        }
        cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);

        //cv::imshow("9", imgContours);
        std::string filename8;
        filename8 = filename8+ "9_" + (char)('0' + intPlateCounter) + ".jpg";
        cv::imwrite(filename8, imgContours);
#endif	// SHOW_STEPS

        // ������ַ�������������ַ�ʶ��
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestVectorOfMatchingCharsInPlate, intPlateCounter);

#ifdef SHOW_STEPS
        std::cout << "chars found in plate number " << intPlateCounter << " = " << possiblePlate.strChars << ", click on any image and press a key to continue . . ." << std::endl;
        intPlateCounter++;
        //cv::waitKey(0);
#endif	// SHOW_STEPS

    }

#ifdef SHOW_STEPS
    std::cout << std::endl << "char detection complete, click on any image and press a key to continue . . ." << std::endl;
    //cv::waitKey(0);
#endif	// SHOW_STEPS

    return vectorOfPossiblePlates;
}

std::vector<PossibleChar> findPossibleCharsInPlate(cv::Mat &imgGrayscale, cv::Mat &imgThresh)
{
    std::vector<PossibleChar> vectorOfPossibleChars;

    cv::Mat imgThreshCopy;

    std::vector<std::vector<cv::Point> > contours;

    imgThreshCopy = imgThresh.clone();

    cv::findContours(imgThreshCopy, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (auto &contour : contours) {
        PossibleChar possibleChar(contour);

        if (checkIfPossibleChar(possibleChar)) {                // �ּ�⣬�Ƿ�������ַ�
            vectorOfPossibleChars.push_back(possibleChar);      // ����������
        }
    }

    return vectorOfPossibleChars;
}

bool checkIfPossibleChar(PossibleChar &possibleChar) {
    // ���ڵ�һ�δ�Ѱ�ң��жϸ������Ƿ�����Ǹ��ַ���û�н��з���
    if (possibleChar.boundingRect.area() > MIN_PIXEL_AREA &&
        possibleChar.boundingRect.width > MIN_PIXEL_WIDTH && possibleChar.boundingRect.height > MIN_PIXEL_HEIGHT &&
        MIN_ASPECT_RATIO < possibleChar.dblAspectRatio && possibleChar.dblAspectRatio < MAX_ASPECT_RATIO) {
        return true;
    }
    else {
        return false;
    }
}

std::vector<std::vector<PossibleChar> > findVectorOfVectorsOfMatchingChars(const std::vector<PossibleChar> &vectorOfPossibleChars)
{
    // ��֮ǰ��⵽�Ŀ������ַ����������з��飬δ��������ַ�����
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingChars;

    for (auto &possibleChar : vectorOfPossibleChars)
    {
        //forѭ�����뵱ǰ�ַ�������ӽ���char,�ӽ�ָ���Ǵ�С�ǶȽӽ���
        std::vector<PossibleChar> vectorOfMatchingChars = findVectorOfMatchingChars(possibleChar, vectorOfPossibleChars);

        vectorOfMatchingChars.push_back(possibleChar);          // ���Լ�Ҳ�ֽ������

        if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) {  //�����ǰ��������3���ַ�������
            continue;
        }
        // ��ͬһ������ַ������Ž������У���Ϊһ���飬vectorOfMatchingCharsΪһ����possibleChar�ӽ���char�������Լ���
        vectorOfVectorsOfMatchingChars.push_back(vectorOfMatchingChars);
        
        //���Ѿ��ֹ�����ַ������������ַ�������ɾ���������Ͳ�����η�����
        std::vector<PossibleChar> vectorOfPossibleCharsWithCurrentMatchesRemoved;  

        for (auto &possChar : vectorOfPossibleChars) {    //��big vector����ȥ����vectorOfMatchingChars
            if (std::find(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possChar) == vectorOfMatchingChars.end()) {
                vectorOfPossibleCharsWithCurrentMatchesRemoved.push_back(possChar);
            }
        }
        // ����һ���ݹ�
        std::vector<std::vector<PossibleChar> > recursiveVectorOfVectorsOfMatchingChars;

        // ��ɸѡ֮��ʣ�µ������ַ�������ȥ����
        recursiveVectorOfVectorsOfMatchingChars = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsWithCurrentMatchesRemoved);//���еݹ�

        for (auto &recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) {
            vectorOfVectorsOfMatchingChars.push_back(recursiveVectorOfMatchingChars);               // ���ݹ��зֳ����ַ���Ҳ�Ž�����������
        }

        break;
    }

    return vectorOfVectorsOfMatchingChars;//����һ��һ��������ַ�
}

std::vector<PossibleChar> findVectorOfMatchingChars(const PossibleChar &possibleChar, const std::vector<PossibleChar> &vectorOfChars)
{
    //Ϊ��ǰ�ַ��ҵ�һ����ӽ����ַ�
    std::vector<PossibleChar> vectorOfMatchingChars;

    for (auto &possibleMatchingChar : vectorOfChars) {
        if (possibleMatchingChar == possibleChar) {         //���Լ�����
            continue;
        }
        // ��������֮��Ĳ������жϷ���
        double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);  //����char֮��ľ���
        double dblAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar);      //char֮��ĽǶ�
        double dblChangeInArea = (double)abs(possibleMatchingChar.boundingRect.area() - possibleChar.boundingRect.area()) / (double)possibleChar.boundingRect.area();//����ڵ�ǰ�ַ���������������
        double dblChangeInWidth = (double)abs(possibleMatchingChar.boundingRect.width - possibleChar.boundingRect.width) / (double)possibleChar.boundingRect.width;//����ڵ�ǰ�ַ���������Կ�Ȳ�
        double dblChangeInHeight = (double)abs(possibleMatchingChar.boundingRect.height - possibleChar.boundingRect.height) / (double)possibleChar.boundingRect.height;//����ڵ�ǰ�ַ���������Ը߶Ȳ�

        // �����ж�
        if (dblDistanceBetweenChars < (possibleChar.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) && //С�ڶԽ��߳���5��
            dblAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS &&   //12
            dblChangeInArea < MAX_CHANGE_IN_AREA && //С���ַ������1.5��
            dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&  //С���ַ���ȵ�1.8��
            dblChangeInHeight < MAX_CHANGE_IN_HEIGHT) {  //С���ַ��߶ȵ�1.2��
            vectorOfMatchingChars.push_back(possibleMatchingChar);      // if the chars are a match, add the current char to vector of matching chars
        }
    }

    return(vectorOfMatchingChars);          // return result
}

// �ù��ɶ���������
double distanceBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar)
{
    int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
    int intY = abs(firstChar.intCenterY - secondChar.intCenterY);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

// ��arctan���������ַ���������֮��Ƕ�
double angleBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar)
{
    double dblAdj = abs(firstChar.intCenterX - secondChar.intCenterX);
    double dblOpp = abs(firstChar.intCenterY - secondChar.intCenterY);

    double dblAngleInRad = atan(dblOpp / dblAdj);

    double dblAngleInDeg = dblAngleInRad * (180.0 / CV_PI);

    return(dblAngleInDeg);
}

// ����������ַ��ص���˴�̫�����������ǵ������ַ���ɾ����С���ַ���
// Ϊ�˷�ֹͬһ���ַ���������������
// ���磬������ĸ��O�����ڻ����⻷��������Ϊ�����ҵ���������Ӧ��ֻ�����ַ�һ�Ρ�
std::vector<PossibleChar> removeInnerOverlappingChars(std::vector<PossibleChar> &vectorOfMatchingChars)
{
    std::vector<PossibleChar> vectorOfMatchingCharsWithInnerCharRemoved(vectorOfMatchingChars);

    for (auto &currentChar : vectorOfMatchingChars) {
        for (auto &otherChar : vectorOfMatchingChars) {
            if (currentChar != otherChar) {
                // �����ǰ�ַ������������ַ����������ĵ�ӽ�ͬһ��λ�ã�����С�ڶԽ��ߵ�0.3��
                if (distanceBetweenChars(currentChar, otherChar) < (currentChar.dblDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY)) {
                    // �����������ҵ����ص����ַ���Ȼ���Ƴ���С���ַ�����
                    // �����ǰ�ַ������������ַ�����С
                    if (currentChar.boundingRect.area() < otherChar.boundingRect.area()) {
                        // ��������Ѱ�ҵ�ǰ�ַ�����
                        std::vector<PossibleChar>::iterator currentCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), currentChar);
                        // ���������û�е������ô����ַ�������ԭ�����������ҵ���
                        if (currentCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(currentCharIterator);       // �Ƴ���ǰ�ַ�
                        }
                    }
                    else {        // ����������ַ��ȵ�ǰ�ַ�С
                                  // ��������Ѱ����������ַ�����
                        std::vector<PossibleChar>::iterator otherCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), otherChar);
                        // ���������û�е������ô����ַ�������ԭ�����������ҵ���
                        if (otherCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(otherCharIterator);         // �Ƴ���������ַ�
                        }
                    }
                }
            }
        }
    }

    return vectorOfMatchingCharsWithInnerCharRemoved;
}

// ���ַ��������н����ַ�ʶ��
std::string recognizeCharsInPlate(cv::Mat &imgThresh, std::vector<PossibleChar> &vectorOfMatchingChars, int &intPlateCounter)
{
    std::string strChars;               // ��󷵻صĽ������ƽ���ϵ��ַ���

    cv::Mat imgThreshColor;

    // ���ַ�����������ַ����������������������
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    cv::cvtColor(imgThresh, imgThreshColor, CV_GRAY2BGR);

	//cv::imshow("a", imgThresh);
	//v::imshow("b", imgThreshColor);

    for (auto &currentChar : vectorOfMatchingChars) {           // ��ʵ���ַ����������ÿ���ַ�
        cv::rectangle(imgThreshColor, currentChar.boundingRect, SCALAR_GREEN, 2);       // ���ַ������̿�
        //cv::imshow("c", imgThreshColor);
        
        cv::Mat imgROItoBeCloned = imgThresh(currentChar.boundingRect);                 // �Ѹ��ַ�������Ӷ�ֵ��ƽ������ȡ����
        
        cv::Mat imgROI = imgROItoBeCloned.clone();      // clone ROI image so we don't change original when we resize
        
        cv::Mat imgROIResized;
        // �Ѹ��ַ�ͼƬ�ߴ��������ѵ��ͼƬһ�£�20x30������ʶ��
        cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));
        
        cv::Mat matROIFloat;

        imgROIResized.convertTo(matROIFloat, CV_32FC1);         // תΪ�������ͣ���ΪKNNʶ�����Ҫ

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // ת��һ������

        cv::Mat matCurrentChar(0, 0, CV_32F);                   // ����һ���������ڻ�ȡʶ����ַ����

        kNearest->findNearest(matROIFlattenedFloat, 3, matCurrentChar);     // KNNʶ��

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       // ����ǰ�ַ��Ӿ���תΪ��������

        strChars = strChars + char(int(fltCurrentChar));        // תΪchar���ͼ����ܵ��ַ�������
    }

#ifdef SHOW_STEPS
    //cv::imshow("10", imgThreshColor);
    std::string filename9;
    filename9 = filename9+ "10_" + (char)('0' + intPlateCounter) + ".jpg";
    cv::imwrite(filename9, imgThreshColor);
#endif	// SHOW_STEPS

    return strChars;               // ����ʶ����ַ���
}

