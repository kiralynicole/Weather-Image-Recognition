#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <filesystem>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

const int CLASSES = 11;
const int PROPERTIES = 262;
const int RADIUS = 108;

enum tag
{
    DEW,
    FOGSMOG,
    FROST,
    GLAZE,
    HAIL,
    LIGHTNING,
    RAIN,
    RAINBOW,
    RIME,
    SANDSTORM,
    SNOW
} tagEnum;

const char* tagList[] =
{
    "dew",
    "fogsmog",
    "frost",
    "glaze",
    "hail",
    "lightning",
    "rain",
    "rainbow",
    "rime",
    "sandstorm",
    "snow"
};

map<string, int> tagToInt = {
    {"dew", DEW},
    {"fogsmog", FOGSMOG},
    {"frost", FROST},
    {"glaze", GLAZE},
    {"hail", HAIL},
    {"lightning", LIGHTNING},
    {"rain", RAIN},
    {"rainbow", RAINBOW},
    {"rime", RIME},
    {"sandstorm", SANDSTORM},
    {"snow", SNOW}
};

char folderPath[] = ".\\dataset";

struct HSVThreshold {
    vector<double> mean;
    int count;
    HSVThreshold() : mean(0), count(0) {}
};

vector<string> testSet;
vector<string> trainSet;
vector<string> imgSet;
vector<int> tagTestSet;
vector<int> tagTrainSet;

vector<HSVThreshold> classThresholds(CLASSES);

int nrImg;
int nrTag;
int accMat[CLASSES][CLASSES];

bool thresholdIsEmpty = true;

void wait() {
    int c;
    cout << endl << "Press enter to continue.";
    while ((c = getchar()) != '\n' && c != EOF);
    getchar();
}

void createTagSet() {
    tagTrainSet.clear();
    tagTestSet.clear();

    for (const auto& img : trainSet) {
        stringstream str(img);
        string tagName, lastWord, word;
        while (getline(str, word, '\\')) {
            tagName = lastWord;
            lastWord = word;
        }
        tagTrainSet.push_back(tagToInt[tagName]);
    }
    for (const auto& img : testSet) {
        stringstream str(img);
        string tagName, lastWord, word;
        while (getline(str, word, '\\')) {
            tagName = lastWord;
            lastWord = word;
        }
        tagTestSet.push_back(tagToInt[tagName]);
    }
}

void openImages()
{
    char file[MAX_PATH];
    char subfolder[MAX_PATH];

    nrImg = 0;
    trainSet.clear();
    testSet.clear();

    for (int i = 0; i < sizeof(tagList) / sizeof(char*); i++) {
        strcpy(subfolder, folderPath);
        strcat(subfolder, "\\");
        strcat(subfolder, tagList[i]);

        imgSet.clear();

        FileGetter fg(subfolder, "jpg");
        while (fg.getNextAbsFile(file))
        {
            nrImg++;
            imgSet.push_back(file);
        }
        sort(imgSet.begin(), imgSet.end());

        int index = 0;
        for (const auto& img : imgSet) {
            if (index++ < imgSet.size() / 2)
                trainSet.push_back(img);
            else
                testSet.push_back(img);
        }
    }
    sort(trainSet.begin(), trainSet.end());
    sort(testSet.begin(), testSet.end());

    createTagSet();
}

void saveImage(const string& img, int actual, int predicted) {
    Mat image = imread(img);
    if (image.empty()) {
        printf("Could not read the image: %s\n", img.c_str());
        return;
    }

    string imgName = img.substr(img.find_last_of("\\/") + 1); // Extract filename from path
    string folderPath = ".\\wrongPredictions";
    string path = folderPath + "\\" + to_string(actual) + "_" + to_string(predicted) + "_" + imgName;

    // Save the image
    if (!imwrite(path, image)) {
        printf("Could not save the image at path: %s\n", path.c_str());
    }
}



void verifyNbOfImages() {
    openImages();
    if (nrImg == 6862 && nrTag == 6862) {
        printf("\nTest verify nb of images/tags successfully\n");
    }
    else {
        printf("\nYou made a mistake in counting the files\nnrImg = %d; nrTag = %d.", nrImg, nrTag);
    }
}

int randomTag() {
    return  rand() % CLASSES;
}

void accuracy(vector<int> newTagTestSet) {
    double ok = 0;

    for (size_t i = 0; i < newTagTestSet.size(); i++) {
        if (newTagTestSet[i] == tagTestSet[i]) {
            ok++;
        }
    }

    double acc = (ok / newTagTestSet.size()) * 100.0;
    printf("Accuracy: %f\n", acc);

}

void resetAccMat() {
    for (int i = 0; i < CLASSES; i++) {
        for (int j = 0; j < CLASSES; j++) {
            accMat[i][j] = 0;
        }
    }
}

void printLines(int index) {
    printf("\n___________|");
    for (int i = 0; i < CLASSES; i++) {
        if (i == index) {
            printf("|______||");
        }
        else {
            printf("________|");
        }
    }
    printf("\n");
}

void printAccuracyMat() {
    printf("           |");
    for (int i = 0; i < CLASSES; i++) {
        printf("%8.7s|", tagList[i]);
    }
    for (int i = 0; i < CLASSES; i++) {
        printLines(i - 1);
        printf("%10s |", tagList[i]);
        for (int j = 0; j < CLASSES; j++) {
            if (i == j) {
                printf("|%6d||", accMat[i][j]);
            }
            else {
                printf("%7d |", accMat[i][j]);
            }
        }
    }
    printLines(10);
}

bool isDew(const string& imagePath) {
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "Failed to load image: " << imagePath << std::endl;
        return false;
    }

    Mat hsvImage;
    cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    Scalar lowerGreen(40, 40, 40);
    Scalar upperGreen(80, 255, 255);

    Mat mask;
    inRange(hsvImage, lowerGreen, upperGreen, mask);

    double greenPercentage = cv::countNonZero(mask) / (double)(image.rows * image.cols);

    return (greenPercentage > 0.5);
}

vector<double> convertToFrequencyAndCountGrays(const string& imagePath, int radius) {
    Mat image = imread(imagePath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Failed to load image: " << imagePath << std::endl;
        return {};
    }

    // Din cate tin minte de la profu, imaginea trebuie sa fie rescalata, asa ca inainte sa convertim imaginea trebuie sa ii dam un resize
    // pentru ca altfel poate sa aiba un comportament ciudat sau sa nu mearga pur si simplu, idk :) 
    Mat padded;
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols);
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    // dft
    dft(complexI, complexI);

    split(complexI, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];

    magI += Scalar::all(1);
    log(magI, magI);

    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // originea trebuie sa fie la centrul imaginii, aici o rearanjam sa sa corespunda 
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));
    Mat q1(magI, Rect(cx, 0, cx, cy));
    Mat q2(magI, Rect(0, cy, cx, cy));
    Mat q3(magI, Rect(cx, cy, cx, cy));

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, NORM_MINMAX);

    // Aici am creat cercul, raza o dai in antetul functiei, depinde cat de mare vrem noi sa fie
    Mat circleMask = Mat::zeros(magI.size(), CV_8U);
    Point center = Point(cx, cy);
    circle(circleMask, center, radius, Scalar(255), 1);

    Mat circlePixels;
    magI.convertTo(magI, CV_8U, 255.0);
    bitwise_and(magI, circleMask, circlePixels);

    // Aici se numara nuantele de gri de pe cerc
    vector<double> grayShades(256, 0);
    for (int y = 0; y < circlePixels.rows; ++y) {
        for (int x = 0; x < circlePixels.cols; ++x) {
            if (circleMask.at<uchar>(y, x) > 0) {
                int grayValue = circlePixels.at<uchar>(y, x);
                grayShades[grayValue]++;
            }
        }
    }

    return grayShades;
}


void calculateMeanHSVMeanRGBForTags() {
    if (thresholdIsEmpty) {
        thresholdIsEmpty = false;

        if (trainSet.empty()) {
            openImages();
        }

        for (auto& threshold : classThresholds) {
            threshold.mean.clear();
            for (int i = 0; i < PROPERTIES; i++) {
                threshold.mean.push_back(0);
            }
            threshold.count = 0;
        }

        for (size_t i = 0; i < trainSet.size(); i++) {
            string img = trainSet[i];
            Mat image = imread(img);
            if (image.empty()) {
                printf("Could not read the img\n");
                continue;
            }

            vector<double> frequency = convertToFrequencyAndCountGrays(img, RADIUS);
            int tagIndex = tagTrainSet[i];

            for (int j = 0; j < 256; j++) {
                classThresholds[tagIndex].mean.at(j) += frequency.at(j);
            }

            Mat hsvImage;
            cvtColor(image, hsvImage, COLOR_BGR2HSV);

            Scalar meanHSV = mean(hsvImage);
            Scalar meanRGB = mean(image);

            for (int j = 0; j < 3; j++) {
                classThresholds[tagIndex].mean.at(j + 256) += meanHSV(j);
                classThresholds[tagIndex].mean.at(j + 259) += meanRGB(j);
            }
            classThresholds[tagIndex].count++;


        }

        for (auto& threshold : classThresholds) {
            if (threshold.count > 0) {
                for (int i = 0; i < PROPERTIES; i++) {
                    threshold.mean.at(i) /= threshold.count;
                }
            }
        }
    }
}

void printMeanHSVMeanRGBValues() {
    calculateMeanHSVMeanRGBForTags();
    for (int i = 0; i < CLASSES; i++) {
        printf("_________________\n");
        printf("Mean HSV for %10s: %7.2f, %7.2f, %7.2f\n",
            tagList[i],
            classThresholds[i].mean.at(0),
            classThresholds[i].mean.at(1),
            classThresholds[i].mean.at(2));
        printf("Mean RGB for %10s: %7.2f, %7.2f, %7.2f\n",
            tagList[i],
            classThresholds[i].mean.at(3),
            classThresholds[i].mean.at(4),
            classThresholds[i].mean.at(5));
    }
}

double euclideanDistance(const vector<double>& v1, const vector<double>& v2) {
    double sum = 0;
    for (int i = 0; i < PROPERTIES; i++) {
        double dif = v1.at(i) - v2.at(i);
        sum += dif * dif;
    }
    return std::sqrt(sum);
}

int closestHSVandRGB(const vector<double>& imageProp) {
    int index = 0;
    double minDist = euclideanDistance(imageProp, classThresholds[0].mean);
    for (int i = 1; i < CLASSES; i++) {
        double nextDist = euclideanDistance(imageProp, classThresholds[i].mean);
        if (nextDist < minDist) {
            minDist = nextDist;
            index = i;
        }
    }
    return index;
}

void predictAndUpdateAccMat() {
    if (testSet.empty()) {
        openImages();
    }

    for (size_t i = 0; i < testSet.size(); ++i) {
        int actualTag = tagTestSet[i];
        string img = testSet[i];
        Mat image = imread(img);
        if (image.empty()) {
            printf("Could not read the img\n");
            continue;
        }

        Mat hsvImage;
        cvtColor(image, hsvImage, COLOR_BGR2HSV);
        Scalar meanHSV = mean(hsvImage);
        Scalar meanRGB = mean(image);

        vector<double> mean = convertToFrequencyAndCountGrays(img, RADIUS);
        for (int j = 0; j < 3; j++) {
            mean.push_back(meanHSV[j]);
        }
        for (int j = 0; j < 3; j++) {
            mean.push_back(meanRGB[j]);
        }

        int predictedTag = closestHSVandRGB(mean);

        accMat[predictedTag][actualTag]++;

        if (actualTag != predictedTag) {
            saveImage(img, actualTag, predictedTag);
        }
    }
}

void testAccuracy() {
    calculateMeanHSVMeanRGBForTags();
    vector<int> newTagTestSet;

    if (testSet.size() == 0) {
        openImages();
    }

    for (const auto& img : testSet) {
        int predictedTag = 0;
        Mat image = imread(img);

        if (image.empty()) {
            printf("Could not read the img\n");
        }
        else {
            Mat hsvImage;
            cvtColor(image, hsvImage, COLOR_BGR2HSV);
            Scalar meanHSV = mean(hsvImage);
            Scalar meanRGB = mean(image);

            vector<double> mean = convertToFrequencyAndCountGrays(img, RADIUS);
            for (int j = 0; j < 3; j++) {
                mean.push_back(meanHSV[j]);
            }
            for (int j = 0; j < 3; j++) {
                mean.push_back(meanRGB[j]);
            }

            predictedTag = closestHSVandRGB(mean);
        }

        newTagTestSet.push_back(predictedTag);
    }
    accuracy(newTagTestSet);
}

void testAccuracyPerClass() {
    resetAccMat();
    calculateMeanHSVMeanRGBForTags();
    predictAndUpdateAccMat();
    printAccuracyMat();
}

int main()
{
    srand(time(NULL));
    int op;
    do
    {
        system("cls");
        printf("Menu:\n");
        printf(" 1 - Main program\n");
        printf(" 2 - Testing number of images\n");
        printf(" 3 - Calculate accuracy\n");
        printf(" 4 - Calculate accuracy per class\n");
        printf(" 5 - Calculate thresholds per class\n");
        printf(" 6 - Convert image to frequency space and count gray shades\n");
        printf("Option: ");
        scanf("%d", &op);

        switch (op)
        {
        case 1:
            openImages();
            break;
        case 2:
            verifyNbOfImages();
            break;
        case 3:
            testAccuracy();
            break;
        case 4:
            testAccuracyPerClass();
            break;
        case 5:
            printMeanHSVMeanRGBValues();
            break;
        case 6: {
            string imagePath = ".\\dataset\\dew\\2208.jpg";
            int radius;
            cout << "Enter radius: ";
            cin >> radius;
            vector<double> grayShades = convertToFrequencyAndCountGrays(imagePath, radius);
            for (size_t i = 0; i < grayShades.size(); ++i) {
                if (grayShades[i] > 0) {
                    printf("Gray shade %zu: %f pixels\n", i, grayShades[i]);
                }
            }
            break;
        }
        }
        wait();

    } while (op != 0);

    return 0;
}