#include "DetectPlates.h"

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);      //scalar是将图像设置成单一灰度和颜色，此为黑色
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

//#define SHOW_STEPS

std::vector<PossiblePlate> detectPlatesInScene(cv::Mat &imgOriginalScene)
{
    std::vector<PossiblePlate> vectorOfPossiblePlates;

    cv::Mat imgGrayscaleScene;//灰度化后的图
    cv::Mat imgThreshScene;   //二值化后的图
    cv::Mat imgContours(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);  //纯黑色图像

    cv::RNG rng;

    cv::destroyAllWindows();//删除建立的全部窗口

#ifdef SHOW_STEPS
    //cv::imshow("0", imgOriginalScene);
    cv::imwrite("0.jpg", imgOriginalScene);
#endif	// SHOW_STEPS

    preprocess(imgOriginalScene, imgGrayscaleScene, imgThreshScene);        // 得到灰度图和二值化图

#ifdef SHOW_STEPS
    //cv::imshow("1a", imgGrayscaleScene);
    //cv::imshow("1b", imgThreshScene);
    cv::imwrite("1a.jpg", imgGrayscaleScene);
    cv::imwrite("1b.jpg", imgThreshScene);
#endif	// SHOW_STEPS

    // 找到所有的字符
    // 首先找到所有轮廓，然后只保留含有字符的轮廓
    std::vector<PossibleChar> vectorOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene);//返回所有有可能是字符的一些轮廓  //每一个轮廓就是一组点集

#ifdef SHOW_STEPS
    std::cout << "step 2 - vectorOfPossibleCharsInScene.Count = " << vectorOfPossibleCharsInScene.size() << std::endl;//图像中可能是字符轮廓的数量

    imgContours = cv::Mat(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);
    std::vector<std::vector<cv::Point> > contours;

    for (auto &possibleChar : vectorOfPossibleCharsInScene) {
        contours.push_back(possibleChar.contour);
    }
    cv::drawContours(imgContours, contours, -1, SCALAR_WHITE);
    //cv::imshow("2b", imgContours);
    cv::imwrite("2b.jpg", imgContours);
#endif	// SHOW_STEPS

    // 给所有找到的字符轮廓分组
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInScene);//返回一组组相似的轮廓

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

    for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {                     // 对每一个字符分组
        PossiblePlate possiblePlate = extractPlate(imgOriginalScene, vectorOfMatchingChars);        // 提取平面

        if (possiblePlate.imgPlate.empty() == false) {                                              // 如果找到了一个平面
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

        vectorOfPossiblePlates[i].rrLocationOfPlateInScene.points(p2fRectPoints);//得到方框的四个角坐标

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

    cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK); // 纯黑色图片

	//cv::imshow("效果图1", imgContours);

    int intCountOfPossibleChars = 0;

    cv::Mat imgThreshCopy = imgThresh.clone();

    std::vector<std::vector<cv::Point> > contours;
	//每一组Point点集就是一个轮廓
    cv::findContours(imgThreshCopy, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);        // find all contours 从二值图像中来计算轮廓  CHAIN_APPROX_SIMPLE： 仅保存轮廓的拐点信息

    for (unsigned int i = 0; i < contours.size(); i++) {                // for each contour
#ifdef SHOW_STEPS
		cv::drawContours(imgContours, contours, i, SCALAR_WHITE);
#endif	// SHOW_STEPS

        PossibleChar possibleChar(contours[i]); //给轮廓画框，计算宽，高，面积，宽高比，对角线长度，轮廓中心位置

        if (checkIfPossibleChar(possibleChar)) {                // 如果轮廓是个可能的字符
            intCountOfPossibleChars++;                          // 增加可能的字符数
            vectorOfPossibleChars.push_back(possibleChar);      //返回的是所有轮廓中可能的字符轮廓的位置以及上述信息
        }
    }
	//cv::imshow("效果图2", imgContours);
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
    //对于一组的字符轮廓，按照它们的x坐标从左到右排序
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    //计算一组字符轮廓的中心，即平面的中心坐标
    double dblPlateCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
    double dblPlateCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;
    cv::Point2d p2dPlateCenter(dblPlateCenterX, dblPlateCenterY); //平面的中心坐标

    //计算平面的宽和高
    int intPlateWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * PLATE_WIDTH_PADDING_FACTOR);

    double intTotalOfCharHeights = 0;

    for (auto &matchingChar : vectorOfMatchingChars) {
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.boundingRect.height;
    }

    double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();

    int intPlateHeight = (int)(dblAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR);//平均高度的1.5倍

    // 计算平面的角度
    double dblOpposite = vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY - vectorOfMatchingChars[0].intCenterY;
    double dblHypotenuse = distanceBetweenChars(vectorOfMatchingChars[0], vectorOfMatchingChars[vectorOfMatchingChars.size() - 1]);
    double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
    double dblCorrectionAngleInDeg = dblCorrectionAngleInRad * (180.0 / CV_PI);

    // 用带角度的斜方框把这个平面框出来
    possiblePlate.rrLocationOfPlateInScene = cv::RotatedRect(p2dPlateCenter, cv::Size2f((float)intPlateWidth, (float)intPlateHeight), (float)dblCorrectionAngleInDeg);

    cv::Mat rotationMatrix;             // 执行旋转
    cv::Mat imgRotated;
    cv::Mat imgCropped;

    rotationMatrix = cv::getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);         // 计算旋转矩阵

    cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());            // 将整张图旋转

	//cv::imshow("效果图1", imgRotated);
                                                                                            // crop out the actual plate portion of the rotated image
    cv::getRectSubPix(imgRotated, possiblePlate.rrLocationOfPlateInScene.size, possiblePlate.rrLocationOfPlateInScene.center, imgCropped);//在旋转之后的图像中按照之前的方框尺寸以及中心位置画出方框，得到平面

	//cv::imshow("效果图2", imgCropped);
	//cv::waitKey();

    possiblePlate.imgPlate = imgCropped;            // 将提取出来的水平的平面（车牌）图像赋给该平面对象

    return possiblePlate;
}

