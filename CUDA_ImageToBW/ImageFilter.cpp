#include "ImageFilter.h"

ImageFilter::ImageFilter(ImagePixel* inputPixels, uint32_t width, uint32_t height) : Width(width), Height(height), Pixels(inputPixels)
{
}

void ImageFilter::ApplyGrayscaleFilter(ImagePixel* outputPixels)
{
	CreateGrayscaleImageOnGPU(Pixels, outputPixels, Width, Height);
}

void ImageFilter::ApplySepiaFilter(ImagePixel* outputPixels)
{
	CreateSepiaImageOnGPU(Pixels, outputPixels, Width, Height);
}