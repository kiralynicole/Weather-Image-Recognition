#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

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
}tagEnum;

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

int randomNumber() {


	return  rand() % 11;
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
		}
		wait();

	} while (op != 0);

	openImages();

	//cout << "Train" << endl;
	//auto imgName = trainSet.begin();
	//auto tagName = tagTrainSet.begin();
	//for (; imgName != trainSet.end(), tagName != tagTrainSet.end(); ++imgName, ++tagName) {
	//	cout << *imgName << " -> " << *tagName << endl;
	//}
	//cout << endl << "Test" << endl;
	//imgName = testSet.begin();
	//tagName = tagTestSet.begin();
	//for (; imgName != testSet.end(), tagName != tagTestSet.end(); ++imgName, ++tagName) {
	//	cout << *imgName << " -> " << *tagName << endl;
	//}

	return 0;
}