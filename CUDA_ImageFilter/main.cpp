#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include "ImageFileBMP.h"
#include "ImageFilter.h"

int main(int argc, char* argv[])
{
	ImageFileBMP* bitmap = new ImageFileBMP("Input.bmp");
	bitmap->ReadFileToMemory();

	ImagePixel* pixels = bitmap->GetPixelsInOrder(PixelOrder::BGR);

	// Allocate memory for modified pixels
	ImagePixel* modifiedPixels = new ImagePixel[bitmap->PixelsCount];
	memset(modifiedPixels, 0, bitmap->PixelsCount * sizeof(ImagePixel));

	ImageFilter filterHelper(pixels, bitmap->Width, bitmap->Height);

	// Apply grayscale filter
	filterHelper.ApplyGrayscaleFilter(modifiedPixels);

	// Override original pixels with new one
	bitmap->SetPixelsInOrder(modifiedPixels, PixelOrder::BGR);

	// Write changes to file
	bitmap->SaveChangesToFile("Output_grayscale.bmp");

	// Apply sepia filter
	memset(modifiedPixels, 0, bitmap->PixelsCount * sizeof(ImagePixel));
	filterHelper.ApplySepiaFilter(modifiedPixels);

	bitmap->SetPixelsInOrder(modifiedPixels, PixelOrder::BGR);
	bitmap->SaveChangesToFile("Output_sepia.bmp");

	// Cleanup
	delete bitmap;
	delete[] pixels;
	delete[] modifiedPixels;

	return 0;
}