#include "DetectChars.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/types_c.h>

const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);      //scalar是将图像设置成单一灰度和颜色，此为黑色
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// global variables
cv::Ptr<cv::ml::KNearest> kNearest = cv::ml::KNearest::create();

//#define SHOW_STEPS

bool loadKNNDataAndTrainKNN(void)
{
    //读取类别
    cv::Mat matClassificationInts;
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);
    if (fsClassifications.isOpened() == false) {
        std::cout << "error, unable to open training classifications file, exiting program\n\n";
        return false;
    }
    fsClassifications["classifications"] >> matClassificationInts;
    fsClassifications.release();

    //读取用来训练的图像矩阵
    cv::Mat matTrainingImagesAsFlattenedFloats;
    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);
    if (fsTrainingImages.isOpened() == false) {
        std::cout << "error, unable to open training images file, exiting program\n\n";
        return false;
    }
    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;
    fsTrainingImages.release();

    //训练
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
    
    // 此时至少有一个平面
    for (auto &possiblePlate : vectorOfPossiblePlates)
    {
        preprocess(possiblePlate.imgPlate, possiblePlate.imgGrayscale, possiblePlate.imgThresh);        // 预处理得到灰度图和二值化图

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

        // 放大1.6倍，为了更好的识别
        cv::resize(possiblePlate.imgThresh, possiblePlate.imgThresh, cv::Size(), 1.6, 1.6);

		//cv::imshow("6c", possiblePlate.imgThresh);

        // 再次二值化，消除灰度判断模糊区域
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

        // 寻找在平面上所有可能的字符
        //首先寻找所有轮廓，然后保留可能是字符的轮廓
        std::vector<PossibleChar> vectorOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh);

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK); //创建全黑图
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

        // 对所有的字符轮廓分组
        std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInPlate = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInPlate);

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK); //创建全黑图

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

        if (vectorOfVectorsOfMatchingCharsInPlate.size() == 0) {                // 如果在平面上没有字符轮廓的组
#ifdef SHOW_STEPS
            std::cout << "chars found in plate number " << intPlateCounter << " = (none), click on any image and press a key to continue . . ." << std::endl;
            intPlateCounter++;
            //cv::destroyWindow("8");
            //cv::destroyWindow("9");
            //cv::destroyWindow("10");
            //cv::waitKey(0);
#endif	// SHOW_STEPS
            possiblePlate.strChars = "";            // 将该平面表示的字符设置为空
            continue;                               // 返回，对下一个平面再操作
        }

        for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInPlate) {                                         // 对每一个字符轮廓组
            std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);      // 按照从左到右位置排序
            vectorOfMatchingChars = removeInnerOverlappingChars(vectorOfMatchingChars);                                     // 消除重叠的字符
        }

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK); //创建全黑图

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

        // 在一个平面中，认为包含最多字符轮廓的字符轮廓组是实际的字符轮廓组
        unsigned int intLenOfLongestVectorOfChars = 0;
        unsigned int intIndexOfLongestVectorOfChars = 0;
        // 对所有的字符轮廓组遍历，寻找拥有最多字符轮廓数量的字符轮廓组索引号
        for (unsigned int i = 0; i < vectorOfVectorsOfMatchingCharsInPlate.size(); i++) {
            if (vectorOfVectorsOfMatchingCharsInPlate[i].size() > intLenOfLongestVectorOfChars) {
                intLenOfLongestVectorOfChars = vectorOfVectorsOfMatchingCharsInPlate[i].size();
                intIndexOfLongestVectorOfChars = i;
            }
        }
        // 在一个平面中，认为包含最多字符轮廓的字符轮廓组是实际的字符轮廓组
        std::vector<PossibleChar> longestVectorOfMatchingCharsInPlate = vectorOfVectorsOfMatchingCharsInPlate[intIndexOfLongestVectorOfChars];

#ifdef SHOW_STEPS
        imgContours = cv::Mat(possiblePlate.imgThresh.size(), CV_8UC3, SCALAR_BLACK); //创建全黑图

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

        // 在这个字符轮廓组里进行字符识别
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

        if (checkIfPossibleChar(possibleChar)) {                // 粗检测，是否可能是字符
            vectorOfPossibleChars.push_back(possibleChar);      // 放入容器中
        }
    }

    return vectorOfPossibleChars;
}

bool checkIfPossibleChar(PossibleChar &possibleChar) {
    // 用于第一次粗寻找，判断该轮廓是否可能是个字符，没有进行分类
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
    // 将之前检测到的可能是字符的轮廓进行分组，未被分组的字符舍弃
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingChars;

    for (auto &possibleChar : vectorOfPossibleChars)
    {
        //for循环找与当前字符轮廓最接近的char,接近指的是大小角度接近的
        std::vector<PossibleChar> vectorOfMatchingChars = findVectorOfMatchingChars(possibleChar, vectorOfPossibleChars);

        vectorOfMatchingChars.push_back(possibleChar);          // 把自己也分进这个组

        if (vectorOfMatchingChars.size() < MIN_NUMBER_OF_MATCHING_CHARS) {  //如果当前分组少于3个字符，舍弃
            continue;
        }
        // 将同一个组的字符轮廓放进容器中，作为一个组，vectorOfMatchingChars为一组与possibleChar接近的char（包括自己）
        vectorOfVectorsOfMatchingChars.push_back(vectorOfMatchingChars);
        
        //将已经分过组的字符轮廓从所有字符轮廓中删除，这样就不会二次分组了
        std::vector<PossibleChar> vectorOfPossibleCharsWithCurrentMatchesRemoved;  

        for (auto &possChar : vectorOfPossibleChars) {    //从big vector里面去除了vectorOfMatchingChars
            if (std::find(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), possChar) == vectorOfMatchingChars.end()) {
                vectorOfPossibleCharsWithCurrentMatchesRemoved.push_back(possChar);
            }
        }
        // 申明一个递归
        std::vector<std::vector<PossibleChar> > recursiveVectorOfVectorsOfMatchingChars;

        // 将筛选之后剩下的所有字符轮廓再去分组
        recursiveVectorOfVectorsOfMatchingChars = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsWithCurrentMatchesRemoved);//进行递归

        for (auto &recursiveVectorOfMatchingChars : recursiveVectorOfVectorsOfMatchingChars) {
            vectorOfVectorsOfMatchingChars.push_back(recursiveVectorOfMatchingChars);               // 将递归中分出的字符组也放进分组容器中
        }

        break;
    }

    return vectorOfVectorsOfMatchingChars;//返回一组一组的相似字符
}

std::vector<PossibleChar> findVectorOfMatchingChars(const PossibleChar &possibleChar, const std::vector<PossibleChar> &vectorOfChars)
{
    //为当前字符找到一组最接近的字符
    std::vector<PossibleChar> vectorOfMatchingChars;

    for (auto &possibleMatchingChar : vectorOfChars) {
        if (possibleMatchingChar == possibleChar) {         //把自己舍弃
            continue;
        }
        // 计算它们之间的参数，判断分组
        double dblDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar);  //两个char之间的距离
        double dblAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar);      //char之间的角度
        double dblChangeInArea = (double)abs(possibleMatchingChar.boundingRect.area() - possibleChar.boundingRect.area()) / (double)possibleChar.boundingRect.area();//相对于当前字符轮廓的相对面积差
        double dblChangeInWidth = (double)abs(possibleMatchingChar.boundingRect.width - possibleChar.boundingRect.width) / (double)possibleChar.boundingRect.width;//相对于当前字符轮廓的相对宽度差
        double dblChangeInHeight = (double)abs(possibleMatchingChar.boundingRect.height - possibleChar.boundingRect.height) / (double)possibleChar.boundingRect.height;//相对于当前字符轮廓的相对高度差

        // 进行判断
        if (dblDistanceBetweenChars < (possibleChar.dblDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) && //小于对角线长的5倍
            dblAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS &&   //12
            dblChangeInArea < MAX_CHANGE_IN_AREA && //小于字符面积的1.5倍
            dblChangeInWidth < MAX_CHANGE_IN_WIDTH &&  //小于字符宽度的1.8倍
            dblChangeInHeight < MAX_CHANGE_IN_HEIGHT) {  //小于字符高度的1.2倍
            vectorOfMatchingChars.push_back(possibleMatchingChar);      // if the chars are a match, add the current char to vector of matching chars
        }
    }

    return(vectorOfMatchingChars);          // return result
}

// 用勾股定理计算距离
double distanceBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar)
{
    int intX = abs(firstChar.intCenterX - secondChar.intCenterX);
    int intY = abs(firstChar.intCenterY - secondChar.intCenterY);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

// 用arctan计算两个字符轮廓中心之间角度
double angleBetweenChars(const PossibleChar &firstChar, const PossibleChar &secondChar)
{
    double dblAdj = abs(firstChar.intCenterX - secondChar.intCenterX);
    double dblOpp = abs(firstChar.intCenterY - secondChar.intCenterY);

    double dblAngleInRad = atan(dblOpp / dblAdj);

    double dblAngleInDeg = dblAngleInRad * (180.0 / CV_PI);

    return(dblAngleInDeg);
}

// 如果有两个字符重叠或彼此太近，不可能是单独的字符，删除较小的字符，
// 为了防止同一个字符检测出两个轮廓，
// 例如，对于字母“O”，内环和外环都可以作为轮廓找到，但我们应该只包括字符一次。
std::vector<PossibleChar> removeInnerOverlappingChars(std::vector<PossibleChar> &vectorOfMatchingChars)
{
    std::vector<PossibleChar> vectorOfMatchingCharsWithInnerCharRemoved(vectorOfMatchingChars);

    for (auto &currentChar : vectorOfMatchingChars) {
        for (auto &otherChar : vectorOfMatchingChars) {
            if (currentChar != otherChar) {
                // 如果当前字符轮廓和其他字符轮廓的中心点接近同一个位置，距离小于对角线的0.3倍
                if (distanceBetweenChars(currentChar, otherChar) < (currentChar.dblDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY)) {
                    // 在这里我们找到了重叠的字符，然后移除更小的字符轮廓
                    // 如果当前字符轮廓比其他字符轮廓小
                    if (currentChar.boundingRect.area() < otherChar.boundingRect.area()) {
                        // 在容器中寻找当前字符轮廓
                        std::vector<PossibleChar>::iterator currentCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), currentChar);
                        // 如果迭代器没有到最后，那么这个字符轮廓在原来的容器中找到了
                        if (currentCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(currentCharIterator);       // 移除当前字符
                        }
                    }
                    else {        // 如果是其他字符比当前字符小
                                  // 在容器中寻找这个其他字符轮廓
                        std::vector<PossibleChar>::iterator otherCharIterator = std::find(vectorOfMatchingCharsWithInnerCharRemoved.begin(), vectorOfMatchingCharsWithInnerCharRemoved.end(), otherChar);
                        // 如果迭代器没有到最后，那么这个字符轮廓在原来的容器中找到了
                        if (otherCharIterator != vectorOfMatchingCharsWithInnerCharRemoved.end()) {
                            vectorOfMatchingCharsWithInnerCharRemoved.erase(otherCharIterator);         // 移除这个其他字符
                        }
                    }
                }
            }
        }
    }

    return vectorOfMatchingCharsWithInnerCharRemoved;
}

// 在字符轮廓组中进行字符识别
std::string recognizeCharsInPlate(cv::Mat &imgThresh, std::vector<PossibleChar> &vectorOfMatchingChars, int &intPlateCounter)
{
    std::string strChars;               // 最后返回的结果，在平面上的字符串

    cv::Mat imgThreshColor;

    // 把字符轮廓组里的字符轮廓按照坐标从左到右排序
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    cv::cvtColor(imgThresh, imgThreshColor, CV_GRAY2BGR);

	//cv::imshow("a", imgThresh);
	//v::imshow("b", imgThreshColor);

    for (auto &currentChar : vectorOfMatchingChars) {           // 对实际字符轮廓组里的每个字符
        cv::rectangle(imgThreshColor, currentChar.boundingRect, SCALAR_GREEN, 2);       // 给字符画上绿框
        //cv::imshow("c", imgThreshColor);
        
        cv::Mat imgROItoBeCloned = imgThresh(currentChar.boundingRect);                 // 把该字符框区域从二值化平面上提取出来
        
        cv::Mat imgROI = imgROItoBeCloned.clone();      // clone ROI image so we don't change original when we resize
        
        cv::Mat imgROIResized;
        // 把该字符图片尺寸调整成与训练图片一致，20x30，用于识别
        cv::resize(imgROI, imgROIResized, cv::Size(RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT));
        
        cv::Mat matROIFloat;

        imgROIResized.convertTo(matROIFloat, CV_32FC1);         // 转为浮点类型，因为KNN识别的需要

        cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // 转成一行像素

        cv::Mat matCurrentChar(0, 0, CV_32F);                   // 声明一个矩阵用于获取识别的字符结果

        kNearest->findNearest(matROIFlattenedFloat, 3, matCurrentChar);     // KNN识别

        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);       // 将当前字符从矩阵转为浮点类型

        strChars = strChars + char(int(fltCurrentChar));        // 转为char类型加在总的字符串后面
    }

#ifdef SHOW_STEPS
    //cv::imshow("10", imgThreshColor);
    std::string filename9;
    filename9 = filename9+ "10_" + (char)('0' + intPlateCounter) + ".jpg";
    cv::imwrite(filename9, imgThreshColor);
#endif	// SHOW_STEPS

    return strChars;               // 返回识别的字符串
}

