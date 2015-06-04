#pragma once

#include <vector>
#include <fstream>
#include <algorithm>

#include "ImagePixel.h"
#include "ImageHeaderBMP.h"

class ImageFileBMP
{
public:
	ImageFileBMP(const char* relativePath);
	~ImageFileBMP();

	void ReadFileToMemory();
	ImagePixel* GetPixelsInOrder(PixelOrder order);

	unsigned int Width = 0;
	unsigned int Height = 0;
	unsigned int PixelsCount = 0;
private:
	bool IsBitmapFormatValid();

	const char* FilePath;
	std::vector<char> FileBuffer;

	BitmapFileHeader FileHeader;
	BitmapInfoHeader InfoHeader;
};