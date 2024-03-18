#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include <set>
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
	SANDSTONE,
	SNOW
};

const char* tagName[] =
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

char folderPath[] = ".\\dataset";

set<string> testSet;
set<string> trainSet;
set<string> imgSet;

int nrImg;

void wait() {
	int c;
	cout << endl << "Press enter to continue.";
	while ((c = getchar()) != '\n' && c != EOF);
	getchar();
}

void openImages()
{
	char file[MAX_PATH];
	char subfolder[MAX_PATH];

	nrImg = 0;

	for (int i = 0; i < sizeof(tagName) / sizeof(char*); i++) {
		strcpy(subfolder, folderPath);
		strcat(subfolder, "\\");
		strcat(subfolder, tagName[i]);

		imgSet.clear();

		FileGetter fg(subfolder, "jpg");
		while (fg.getNextAbsFile(file))
		{
			nrImg++;
			imgSet.insert(file);
		}

		size_t setSize = imgSet.size();
		int index = 0;
		for (const auto& img : imgSet) {
			if (index++ < setSize / 2)
				trainSet.insert(img);
			else
				testSet.insert(img);
		}
	}
}

void verifyNbOfImages() {
	openImages();
	if (nrImg == 6862) {
		printf("\nTest verify nb of images successfully\n");
	}
	else {
		printf("\nYou made a mistake in counting the files\n");
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

	return 0;
}