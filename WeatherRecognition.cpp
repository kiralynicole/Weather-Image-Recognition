#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <string>
#include <set>

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

void openImages()
{
	char file[MAX_PATH];
	char subfolder[MAX_PATH];

	for (int i = 0; i < sizeof(tagName) / sizeof(char*); i++) {
		strcpy(subfolder, folderPath);
		strcat(subfolder, "\\");
		strcat(subfolder, tagName[i]);
		
		imgSet.clear();

		FileGetter fg(subfolder, "jpg");
		while (fg.getNextAbsFile(file))
		{
			imgSet.insert(file);
		}

		size_t setSize = imgSet.size();
		int index = 0;
		for (const auto& img : imgSet) {
			if (index++ < setSize/2)
				trainSet.insert(img);
			else
				testSet.insert(img);
		}
	}
	cout << "Train set size: " << trainSet.size() << endl;
	cout << "Test set size: " << testSet.size() << endl;
}

int main() 
{
	openImages();

	return 0;
}