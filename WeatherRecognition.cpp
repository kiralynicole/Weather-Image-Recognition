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

vector<string> testSet;
vector<string> trainSet;
vector<string> imgSet;
vector<int> tagTestSet;
vector<int> tagTrainSet;

int nrImg;
int nrTag;
int accMat[11][11] = { 0 }; // Initialize accMat with zeros

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
    return  rand() % 11;
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

void testAccuracy() {
    vector<int> newTagTestSet;

    if (tagTestSet.size() == 0) {
        openImages();
    }

    for (const auto& img : imgSet) {
        int newTag = randomTag();
        newTagTestSet.push_back(newTag);
    }
    accuracy(newTagTestSet);
}

void printLines(int index) {
    printf("\n___________|");
    for (int i = 0; i < 11; i++) {
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
    for (int i = 0; i < 11; i++) {
        printf("%8.7s|", tagList[i]);
    }
    for (int i = 0; i < 11; i++) {
        printLines(i - 1);
        printf("%10s |", tagList[i]);
        for (int j = 0; j < 11; j++) {
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



bool isDew(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return false;
    }


    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);


    cv::Scalar lowerGreen(40, 40, 40);
    cv::Scalar upperGreen(80, 255, 255);


    cv::Mat mask;
    cv::inRange(hsvImage, lowerGreen, upperGreen, mask);


    double greenPercentage = cv::countNonZero(mask) / (double)(image.rows * image.cols);


    return (greenPercentage > 0.5);
}

void predictAndUpdateAccMat() {
    if (tagTestSet.empty() || tagTrainSet.empty()) {
        openImages();
    }
    for (size_t i = 0; i < trainSet.size(); ++i) {
        std::string imagePath = trainSet[i];
        int actualTag = tagTrainSet[i];

        bool isGreen = isDew(imagePath);

        int predictedTag;
        if (isGreen) {
            predictedTag = DEW;
        }
        else {
            predictedTag = randomTag();
        }
        accMat[predictedTag][actualTag]++;

    }
}

void testAccuracyPerClass() {
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
        }
        wait();

    } while (op != 0);

    return 0;
}
