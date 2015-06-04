#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "ImagePixel.h"
#include "ImageHeaderBMP.h"

class ImageFileBMP
{
public:
	ImageFileBMP(const char* relativePath);
	~ImageFileBMP();

	bool ReadFileToMemory();
	ImagePixel* GetPixelsInOrder(PixelOrder order);

	unsigned int Width;
	unsigned int Height;
	unsigned int PixelsCount;
	
private:
	bool IsBitmapFormatValid();

	const char* FilePath;
	std::vector<char> FileBuffer;

	BitmapFileHeader FileHeader;
	BitmapInfoHeader InfoHeader;
};
