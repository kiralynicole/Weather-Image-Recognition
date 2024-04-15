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

using namespace std;
using namespace cv;

const int CLASSES = 11;
const int PROPERTIES = 6;

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

void saveImage(string img, int actual, int predicted) {
    stringstream str(img);
    string tagName, imgName, word;
    while (getline(str, word, '\\')) {
        tagName = imgName;
        imgName = word;
    }
    string path = ".\\wrongPredictions\\" + to_string(actual) + "_" + to_string(predicted) + "_" + tagName + "_" + imgName;
    if (!imwrite(path, imread(img))) {
        printf("Could not save the img\n");
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
            Mat image = imread(trainSet[i]);
            if (image.empty()) {
                printf("Could not read the img\n");
                continue;
            }

            Mat hsvImage;
            cvtColor(image, hsvImage, COLOR_BGR2HSV);

            Scalar meanHSV = mean(hsvImage);
            Scalar meanRGB = mean(image);

            int tagIndex = tagTrainSet[i];
            for (int j = 0; j < 3; j++) {
                classThresholds[tagIndex].mean.at(j) += meanHSV(j);
                classThresholds[tagIndex].mean.at(j + 3) += meanRGB(j);
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
        printf("___________________________________________________\n");
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
        
        vector<double> mean;
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

            vector<double> mean;
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
        }
        wait();

    } while (op != 0);

    return 0;
}