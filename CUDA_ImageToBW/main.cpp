#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include "ImageFileBMP.h"

void CreateBWImage(ImagePixel* inputPixels, ImagePixel* outputPixels, int pixelsCount);

int main()
{
	ImageFileBMP* bitmap = new ImageFileBMP("test.bmp");
	bitmap->ReadFileToMemory();

	ImagePixel* pixels = bitmap->GetPixelsInOrder(PixelOrder::RGB);
	ImagePixel* bwPixels = new ImagePixel[bitmap->PixelsCount];
	memset(bwPixels, 0, bitmap->PixelsCount * sizeof(ImagePixel));

	CreateBWImage(pixels, bwPixels, bitmap->PixelsCount);

	delete bitmap;
	delete[] pixels;
	delete[] bwPixels;

	return 0;
}