#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

const int CLASSES = 11;

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
    Scalar meanHSV;
    int count;
    HSVThreshold() : meanHSV(0, 0, 0), count(0) {}
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
    nrTag = 0;
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
        nrTag++;
    }
    for (const auto& img : testSet) {
        stringstream str(img);
        string tagName, lastWord, word;
        while (getline(str, word, '\\')) {
            tagName = lastWord;
            lastWord = word;
        }
        tagTestSet.push_back(tagToInt[tagName]);
        nrTag++;
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

//void predictAndUpdateAccMat() {
//    if (tagTestSet.empty() || tagTrainSet.empty()) {
//        openImages();
//    }
//    for (size_t i = 0; i < trainSet.size(); ++i) {
//        string imagePath = trainSet[i];
//        int actualTag = tagTrainSet[i];
//
//        bool isGreen = isDew(imagePath);
//
//        int predictedTag;
//        if (isGreen) {
//            predictedTag = DEW;
//        }
//        else {
//            predictedTag = randomTag();
//        }
//        accMat[predictedTag][actualTag]++;
//    }
//}

void calculateMeanHSVForTags() {
    if (thresholdIsEmpty) {
        thresholdIsEmpty = false;

        if (trainSet.empty()) {
            openImages();
        }

        for (auto& threshold : classThresholds) {
            threshold.meanHSV = Scalar(0, 0, 0);
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

            int tagIndex = tagTrainSet[i];
            classThresholds[tagIndex].meanHSV += meanHSV;
            classThresholds[tagIndex].count++;

        }

        for (auto& threshold : classThresholds) {
            if (threshold.count > 0) {
                threshold.meanHSV[0] /= threshold.count;
                threshold.meanHSV[1] /= threshold.count;
                threshold.meanHSV[2] /= threshold.count;
            }
        }
    }
}

void printMeanHSVValues() {
    calculateMeanHSVForTags();
    for (int i = 0; i < CLASSES; i++) {
        printf("Mean HSV for %10s: %10f, %10f, %10f\n", 
            tagList[i],
            classThresholds[i].meanHSV[0],
            classThresholds[i].meanHSV[1],
            classThresholds[i].meanHSV[2]);
    }
}

double euclideanDistance(const Scalar& s1, const Scalar& s2) {
    double dx = s1[0] - s2[0];
    double dy = s1[1] - s2[1];
    double dz = s1[2] - s2[2];

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

int closestHSV(const Scalar& hsvImage) {
    int index = 0;
    double minDist = euclideanDistance(hsvImage, classThresholds[0].meanHSV);
    for (int i = 1; i < CLASSES; i++) {
        double nextDist = euclideanDistance(hsvImage, classThresholds[i].meanHSV);
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
        Mat image = imread(testSet[i]);
        if (image.empty()) {
            printf("Could not read the img\n");
            continue;
        }

        Mat hsvImage;
        cvtColor(image, hsvImage, COLOR_BGR2HSV);
        Scalar meanHSV = mean(hsvImage);

        int predictedTag = closestHSV(meanHSV);

        accMat[predictedTag][actualTag]++;
    }
}

void testAccuracy() {
    calculateMeanHSVForTags();
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
            predictedTag = closestHSV(meanHSV);
        }

        newTagTestSet.push_back(predictedTag);
    }
    accuracy(newTagTestSet);
}

void testAccuracyPerClass() {
    resetAccMat();
    calculateMeanHSVForTags();
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
            printMeanHSVValues();
        }
        wait();

    } while (op != 0);

    return 0;
}