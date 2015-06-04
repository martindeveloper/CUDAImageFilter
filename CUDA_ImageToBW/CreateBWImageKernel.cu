#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>
#include "ImagePixel.h"

__global__ void CreateBWImageKernel(ImagePixel* inputPixels, ImagePixel* outputPixels)
{
	int pixelIndex = /*blockIdx.x * */threadIdx.x;

	ImagePixel* inputPixel = &inputPixels[pixelIndex];
	ImagePixel* outputPixel = &outputPixels[pixelIndex];

	// Data loss of fractional part
	outputPixel->R = inputPixel->R * 0.299f;
	outputPixel->G = inputPixel->G * 0.587f;
	outputPixel->B = inputPixel->B * 0.114f;
}

void CreateBWImage(ImagePixel* inputPixels, ImagePixel* outputPixels, int pixelsCount)
{
	int pixelsBytes = sizeof(ImagePixel) * pixelsCount;

	// Output pixels on device - malloc
	ImagePixel* outputPixelsDevice;
	assert(cudaMalloc((void **)&outputPixelsDevice, pixelsBytes) == cudaSuccess);

	// Input pixels on device - malloc and copy to VRAM
	ImagePixel* inputPixelsDevice;
	assert(cudaMalloc((void **)&inputPixelsDevice, pixelsBytes) == cudaSuccess);
	assert(cudaMemcpy(inputPixelsDevice, inputPixels, pixelsBytes, cudaMemcpyHostToDevice) == cudaSuccess);

	// One block and pixelsCount threads
	dim3 blocks(0, 0, 0);
	dim3 threads(0, 0, 0);

	CreateBWImageKernel<<<1, 1>>>(inputPixelsDevice, outputPixelsDevice);
	cudaDeviceSynchronize();

	// Copy from device to CPU memory pointer
	assert(cudaMemcpy(outputPixels, outputPixelsDevice, pixelsBytes, cudaMemcpyDeviceToHost) == cudaSuccess);

	// Free device memory
	cudaFree(outputPixelsDevice);
	cudaFree(inputPixelsDevice);

	const char* error = cudaGetErrorString(cudaGetLastError());

	cudaDeviceReset();
}