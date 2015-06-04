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

		FileBuffer.resize((unsigned int)length);

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
		}
		else
		{
			std::cout << "Error: File needs to be bitmap, uncompressed and 24bit" << std::endl;
			return false;
		}
	}
	else
	{
		std::cout << "Error: Can not read file" << std::endl;
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

ImagePixel* ImageFileBMP::GetPixelsInOrder(PixelOrder order)
{
	char* rawBuffer = &FileBuffer[0];
	rawBuffer += FileHeader.bfOffBits; // Move pointer after header

	ImagePixel* pixels = new ImagePixel[Width * Height];

	int pixelIndex = 0;

	for (unsigned int y = 0; y < Height; y++)
	{
		for (unsigned int x = 0; x < Width; x++)
		{
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

			pixelIndex++;
		}
	}

	PixelsCount = pixelIndex;

	return pixels;
}
