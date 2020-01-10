#include "DetectPlates.h"

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);      //scalar�ǽ�ͼ�����óɵ�һ�ҶȺ���ɫ����Ϊ��ɫ
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

//#define SHOW_STEPS

std::vector<PossiblePlate> detectPlatesInScene(cv::Mat &imgOriginalScene)
{
    std::vector<PossiblePlate> vectorOfPossiblePlates;

    cv::Mat imgGrayscaleScene;//�ҶȻ����ͼ
    cv::Mat imgThreshScene;   //��ֵ�����ͼ
    cv::Mat imgContours(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);  //����ɫͼ��

    cv::RNG rng;

    cv::destroyAllWindows();//ɾ��������ȫ������

#ifdef SHOW_STEPS
    //cv::imshow("0", imgOriginalScene);
    cv::imwrite("0.jpg", imgOriginalScene);
#endif	// SHOW_STEPS

    preprocess(imgOriginalScene, imgGrayscaleScene, imgThreshScene);        // �õ��Ҷ�ͼ�Ͷ�ֵ��ͼ

#ifdef SHOW_STEPS
    //cv::imshow("1a", imgGrayscaleScene);
    //cv::imshow("1b", imgThreshScene);
    cv::imwrite("1a.jpg", imgGrayscaleScene);
    cv::imwrite("1b.jpg", imgThreshScene);
#endif	// SHOW_STEPS

    // �ҵ����е��ַ�
    // �����ҵ�����������Ȼ��ֻ���������ַ�������
    std::vector<PossibleChar> vectorOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene);//���������п������ַ���һЩ����  //ÿһ����������һ��㼯

#ifdef SHOW_STEPS
    std::cout << "step 2 - vectorOfPossibleCharsInScene.Count = " << vectorOfPossibleCharsInScene.size() << std::endl;//ͼ���п������ַ�����������

    imgContours = cv::Mat(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);
    std::vector<std::vector<cv::Point> > contours;

    for (auto &possibleChar : vectorOfPossibleCharsInScene) {
        contours.push_back(possibleChar.contour);
    }
    cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);
    //cv::imshow("2b", imgContours);
    cv::imwrite("2b.jpg", imgContours);
#endif	// SHOW_STEPS

    // �������ҵ����ַ���������
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInScene);//����һ�������Ƶ�����

#ifdef SHOW_STEPS
    std::cout << "step 3 - vectorOfVectorsOfMatchingCharsInScene.size() = " << vectorOfVectorsOfMatchingCharsInScene.size() << std::endl;

    imgContours = cv::Mat(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);

    for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {
        int intRandomBlue = rng.uniform(0, 256);
        int intRandomGreen = rng.uniform(0, 256);
        int intRandomRed = rng.uniform(0, 256);

        std::vector<std::vector<cv::Point> > contours;

        for (auto &matchingChar : vectorOfMatchingChars) {
            contours.push_back(matchingChar.contour);
        }
        cv::drawContours(imgContours, contours, -1, cv::Scalar((double)intRandomBlue, (double)intRandomGreen, (double)intRandomRed));
    }
    //cv::imshow("3", imgContours);
    cv::imwrite("3.jpg", imgContours);
#endif	// SHOW_STEPS

    for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {                     // ��ÿһ���ַ�����
        PossiblePlate possiblePlate = extractPlate(imgOriginalScene, vectorOfMatchingChars);        // ��ȡƽ��

        if (possiblePlate.imgPlate.empty() == false) {                                              // ����ҵ���һ��ƽ��
            vectorOfPossiblePlates.push_back(possiblePlate);
        }
    }

    std::cout << std::endl << vectorOfPossiblePlates.size() << " possible plates found" << std::endl;

#ifdef SHOW_STEPS
    std::cout << std::endl;
    //cv::imshow("4a", imgContours);
    cv::imwrite("4a.jpg", imgContours);

    for (unsigned int i = 0; i < vectorOfPossiblePlates.size(); i++) {
        cv::Point2f p2fRectPoints[4];

        vectorOfPossiblePlates[i].rrLocationOfPlateInScene.points(p2fRectPoints);//�õ�������ĸ�������

        for (int j = 0; j < 4; j++) {
            cv::line(imgContours, p2fRectPoints[j], p2fRectPoints[(j + 1) % 4], SCALAR_RED, 2);
        }
        //cv::imshow("4a", imgContours);
        std::string filename;
        filename = filename+ "4a_" + (char)('0' + i) + ".jpg";
        cv::imwrite(filename, imgContours);

        std::cout << "possible plate " << i << ", click on any image and press a key to continue . . ." << std::endl;

        //cv::imshow("4b", vectorOfPossiblePlates[i].imgPlate);
        std::string filename1;
        filename1 = filename1+ "4b_" + (char)('0' + i) + ".jpg";
        cv::imwrite(filename1, vectorOfPossiblePlates[i].imgPlate);
        //cv::waitKey(0);
    }
    std::cout << std::endl << "plate detection complete, click on any image and press a key to begin char recognition . . ." << std::endl << std::endl;
    //cv::waitKey(0);
#endif	// SHOW_STEPS

    return vectorOfPossiblePlates;
}

std::vector<PossibleChar> findPossibleCharsInScene(cv::Mat &imgThresh)
{
    std::vector<PossibleChar> vectorOfPossibleChars;

    cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK); // ����ɫͼƬ

	//cv::imshow("Ч��ͼ1", imgContours);

    int intCountOfPossibleChars = 0;

    cv::Mat imgThreshCopy = imgThresh.clone();

    std::vector<std::vector<cv::Point> > contours;
	//ÿһ��Point�㼯����һ������
    cv::findContours(imgThreshCopy, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);        // find all contours �Ӷ�ֵͼ��������������  CHAIN_APPROX_SIMPLE�� �����������Ĺյ���Ϣ

    for (unsigned int i = 0; i < contours.size(); i++) {                // for each contour
#ifdef SHOW_STEPS
		cv::drawContours(imgContours, contours, i, SCALAR_WHITE);
#endif	// SHOW_STEPS

        PossibleChar possibleChar(contours[i]); //���������򣬼�����ߣ��������߱ȣ��Խ��߳��ȣ���������λ��

        if (checkIfPossibleChar(possibleChar)) {                // ��������Ǹ����ܵ��ַ�
            intCountOfPossibleChars++;                          // ���ӿ��ܵ��ַ���
            vectorOfPossibleChars.push_back(possibleChar);      //���ص������������п��ܵ��ַ�������λ���Լ�������Ϣ
        }
    }
	//cv::imshow("Ч��ͼ2", imgContours);
	//cv::waitKey();

#ifdef SHOW_STEPS
    std::cout << std::endl << "contours.size() = " << contours.size() << std::endl;                         // 2362 with MCLRNF1 image
    std::cout << "step 2 - intCountOfValidPossibleChars = " << intCountOfPossibleChars << std::endl;        // 131 with MCLRNF1 image
    //cv::imshow("2a", imgContours);
    cv::imwrite("2a.jpg", imgContours);
#endif	// SHOW_STEPS

    return(vectorOfPossibleChars);
}

PossiblePlate extractPlate(cv::Mat &imgOriginal, std::vector<PossibleChar> &vectorOfMatchingChars)
{
    PossiblePlate possiblePlate;
    //����һ����ַ��������������ǵ�x�������������
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    //����һ���ַ����������ģ���ƽ�����������
    double dblPlateCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
    double dblPlateCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;
    cv::Point2d p2dPlateCenter(dblPlateCenterX, dblPlateCenterY); //ƽ�����������

    //����ƽ��Ŀ�͸�
    int intPlateWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * PLATE_WIDTH_PADDING_FACTOR);

    double intTotalOfCharHeights = 0;

    for (auto &matchingChar : vectorOfMatchingChars) {
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.boundingRect.height;
    }

    double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();

    int intPlateHeight = (int)(dblAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR);//ƽ���߶ȵ�1.5��

    // ����ƽ��ĽǶ�
    double dblOpposite = vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY - vectorOfMatchingChars[0].intCenterY;
    double dblHypotenuse = distanceBetweenChars(vectorOfMatchingChars[0], vectorOfMatchingChars[vectorOfMatchingChars.size() - 1]);
    double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
    double dblCorrectionAngleInDeg = dblCorrectionAngleInRad * (180.0 / CV_PI);

    // �ô��Ƕȵ�б��������ƽ������
    possiblePlate.rrLocationOfPlateInScene = cv::RotatedRect(p2dPlateCenter, cv::Size2f((float)intPlateWidth, (float)intPlateHeight), (float)dblCorrectionAngleInDeg);

    cv::Mat rotationMatrix;             // ִ����ת
    cv::Mat imgRotated;
    cv::Mat imgCropped;

    rotationMatrix = cv::getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);         // ������ת����

    cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());            // ������ͼ��ת

	//cv::imshow("Ч��ͼ1", imgRotated);
                                                                                            // crop out the actual plate portion of the rotated image
    cv::getRectSubPix(imgRotated, possiblePlate.rrLocationOfPlateInScene.size, possiblePlate.rrLocationOfPlateInScene.center, imgCropped);//����ת֮���ͼ���а���֮ǰ�ķ���ߴ��Լ�����λ�û������򣬵õ�ƽ��

	//cv::imshow("Ч��ͼ2", imgCropped);
	//cv::waitKey();

    possiblePlate.imgPlate = imgCropped;            // ����ȡ������ˮƽ��ƽ�棨���ƣ�ͼ�񸳸���ƽ�����

    return possiblePlate;
}

