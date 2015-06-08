#pragma once

#include <cstdint>

#include "ImagePixel.h"

// These functions will be linked from CUDA files

void CreateGrayscaleImageOnGPU(ImagePixel* inputPixels, ImagePixel* outputPixels, uint32_t width, uint32_t height);
void CreateSepiaImageOnGPU(ImagePixel* inputPixels, ImagePixel* outputPixels, uint32_t width, uint32_t height);

class ImageFilter
{
public:
	ImageFilter(ImagePixel* inputPixels, uint32_t width, uint32_t height);

	void ApplyGrayscaleFilter(ImagePixel* outputPixels);
	void ApplySepiaFilter(ImagePixel* outputPixels);

private:
	uint32_t Width;
	uint32_t Height;
	ImagePixel* Pixels;
};
