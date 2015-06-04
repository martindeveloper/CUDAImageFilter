#pragma once

struct ImagePixel
{
public:
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

enum PixelOrder
{
	BGR = 1,
	RGB = 2
};