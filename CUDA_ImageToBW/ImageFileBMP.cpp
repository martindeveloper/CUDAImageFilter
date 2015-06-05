#include "ImageFileBMP.h"

ImageFileBMP::ImageFileBMP(const char* relativePath) : FilePath(relativePath), Width(0), Height(0), PixelsCount(0)
{
}

ImageFileBMP::~ImageFileBMP()
{
	FileBuffer.clear();
}

bool ImageFileBMP::ReadFileToMemory()
{
	std::ifstream file(FilePath, std::ifstream::in | std::ios::binary);

	file.unsetf(std::ios::skipws);

	if (file.is_open())
	{
		// Measure file size and resize buffer
		file.seekg(0, std::ios::end);
		std::streampos length = file.tellg();
		file.seekg(0, std::ios::beg);

		FileBuffer.resize((uint32_t)length);

		// Read file
		file.read(&FileBuffer[0], length);

		// First is file header and after that, there is bitmap info header
		FileHeader = *(BitmapFileHeader*)(&FileBuffer[0]);
		InfoHeader = *(BitmapInfoHeader*)(&FileBuffer[0] + sizeof(BitmapFileHeader));

		file.close();

		if (IsBitmapFormatValid())
		{
			Width = InfoHeader.biWidth;
			Height = InfoHeader.biHeight;

			PixelsCount = Width * Height;
		}
		else
		{
			std::cerr << "Error: File needs to be bitmap, uncompressed and 24bit" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Error: Can not read file" << std::endl;
		return false;
	}

	return true;
}

bool ImageFileBMP::IsBitmapFormatValid()
{
	bool isBitmap = FileHeader.bfType == 0x4D42;
	bool isUncompressed = InfoHeader.biCompression == 0L;
	bool is24bit = InfoHeader.biBitCount == 24;

	return isBitmap && isUncompressed && is24bit;
}

bool ImageFileBMP::SaveChangesToFile(const char* PathToFile)
{
	std::ofstream outfile(PathToFile, std::ofstream::binary);

	outfile.write(&FileBuffer[0], FileBuffer.size());

	outfile.close();

	return true;
}

char* ImageFileBMP::GetPointerToPixels()
{
	char* rawBuffer = &FileBuffer[0];
	rawBuffer += FileHeader.bfOffBits; // Move pointer after header

	return rawBuffer;
}

void ImageFileBMP::SetPixelsInOrder(ImagePixel* pixels, PixelOrder order)
{
	char* rawBuffer = GetPointerToPixels();

	switch (order)
	{
	default:
	case BGR:
		break;

	case RGB:
		for (uint32_t i = 0; i < PixelsCount; i++) {
			ImagePixel* pixel = (ImagePixel*)&rawBuffer[i * 3];

			std::swap(pixel->B, pixel->R);
		}
		break;
	}

	// Override pixels in memory
	std::memcpy(rawBuffer, pixels, Width * Height * sizeof(ImagePixel));
}

ImagePixel* ImageFileBMP::GetPixelsInOrder(PixelOrder order)
{
	char* rawBuffer = GetPointerToPixels();

	ImagePixel* pixels = new ImagePixel[PixelsCount];

	for (uint32_t pixelIndex = 0; pixelIndex < PixelsCount; pixelIndex++) {
		// 24 bits = 3 bytes
		// B G R
		ImagePixel* bitmapPixel = (ImagePixel*)&rawBuffer[pixelIndex * 3];

		switch (order)
		{
		default:
		case BGR:
			// In Windows is BGR default order
			// Do nothing
			break;

		case RGB:
			std::swap(bitmapPixel->B, bitmapPixel->R);
			break;
		}

		pixels[pixelIndex] = *bitmapPixel;
	}

	return pixels;
}
