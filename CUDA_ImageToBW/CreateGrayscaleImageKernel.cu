#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>

#include "ImagePixel.h"

__global__ void CreateGrayscaleImageKernel(struct ImagePixel* inputPixels, struct ImagePixel* outputPixels, unsigned int width, unsigned int height)
{
	unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int index = width * y + x;

	if (x > width - 1 || y > height - 1) {
		// We are out of image size
		return;
	}

	ImagePixel* inputPixel = &inputPixels[index];
	ImagePixel* outputPixel = &outputPixels[index];

	unsigned char gray = inputPixel->B * 0.299f + inputPixel->G * 0.587f + inputPixel->R * 0.114f;

	// BGR order is used
	outputPixel->B = gray;
	outputPixel->G = gray;
	outputPixel->R = gray;
}

void CreateGrayscaleImageOnGPU(ImagePixel* inputPixels, ImagePixel* outputPixels, unsigned int width, unsigned int height)
{
	int pixelsCount = width * height;
	int pixelsBytes = sizeof(ImagePixel) * pixelsCount;

	// Output pixels on device - malloc
	ImagePixel* outputPixelsDevice;
	assert(cudaMalloc((void **)&outputPixelsDevice, pixelsBytes) == cudaSuccess);
	assert(cudaMemcpy(outputPixelsDevice, outputPixels, pixelsBytes, cudaMemcpyHostToDevice) == cudaSuccess);
	
	// Input pixels on device - malloc and copy to VRAM
	ImagePixel* inputPixelsDevice;
	assert(cudaMalloc((void **)&inputPixelsDevice, pixelsBytes) == cudaSuccess);
	assert(cudaMemcpy(inputPixelsDevice, inputPixels, pixelsBytes, cudaMemcpyHostToDevice) == cudaSuccess);

	dim3 threads(32, 32);
	dim3 blocks(width / threads.x, height / threads.y);

	CreateGrayscaleImageKernel<<<blocks, threads>>>(inputPixelsDevice, outputPixelsDevice, width, height);
	assert(cudaPeekAtLastError() == cudaSuccess);

	cudaDeviceSynchronize();

	// Copy from device to CPU memory pointer
	assert(cudaMemcpy(outputPixels, outputPixelsDevice, pixelsBytes, cudaMemcpyDeviceToHost) == cudaSuccess);

	// Free device memory
	cudaFree(outputPixelsDevice);
	cudaFree(inputPixelsDevice);

	assert(cudaPeekAtLastError() == cudaSuccess);

	cudaDeviceReset();
}