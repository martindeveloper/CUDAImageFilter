#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdint>

#include "ImagePixel.h"
#include "ImageHeaderBMP.h"

class ImageFileBMP
{
public:
	ImageFileBMP(const char* relativePath);
	~ImageFileBMP();

	bool ReadFileToMemory();
	ImagePixel* GetPixelsInOrder(PixelOrder order);
	void SetPixelsInOrder(ImagePixel* pixels, PixelOrder order);
	bool SaveChangesToFile(const char* PathToFile);

	uint32_t Width;
	uint32_t Height;
	uint32_t PixelsCount;
	
private:
	bool IsBitmapFormatValid();
	char* GetPointerToPixels();

	const char* FilePath;
	std::vector<char> FileBuffer;

	BitmapFileHeader FileHeader;
	BitmapInfoHeader InfoHeader;
};
