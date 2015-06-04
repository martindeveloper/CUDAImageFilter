#pragma once
#pragma pack(push, 1)

struct BitmapFileHeader
{
	unsigned short bfType;
	unsigned int   bfSize;
	unsigned short bfReserved1;
	unsigned short bfReserved2;
	unsigned int   bfOffBits;
};

struct BitmapInfoHeader
{
	unsigned int biSize;
	unsigned int biWidth;
	unsigned int biHeight;
	unsigned short biPlanes;
	unsigned short biBitCount;
	unsigned int biCompression;
	unsigned int biSizeImage;
	unsigned int biXpelsPerMeter;
	unsigned int biYpelsPerMeter;
	unsigned int biClrUses;
	unsigned int biClrImportant;
}; 

#pragma pack(pop)