#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>

#include "ImagePixel.h"

#define min(a, b) (a > b)? b: a
#define max(a, b) (a > b)? a: b 

__global__ void CreateSepiaImageKernel(struct ImagePixel* inputPixels, struct ImagePixel* outputPixels, unsigned int width, unsigned int height)
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

	// BGR order is used
	outputPixel->B = min((unsigned int)((inputPixel->B * 0.393f) + (inputPixel->G * 0.769f) + (inputPixel->R * 0.189f)), 255U);
	outputPixel->G = min((unsigned int)((inputPixel->B * 0.349f) + (inputPixel->G * 0.686f) + (inputPixel->R * 0.168f)), 255U);
	outputPixel->R = min((unsigned int)((inputPixel->B * 0.272f) + (inputPixel->G * 0.534f) + (inputPixel->R * 0.131f)), 255U);
}

void MallocAndCopyPixelsToDevice(ImagePixel* pixelsHost, ImagePixel** pixelsDevice, unsigned int pixelsBytes)
{
	assert(cudaMalloc((void **)pixelsDevice, pixelsBytes) == cudaSuccess);
	assert(cudaMemcpy(*pixelsDevice, pixelsHost, pixelsBytes, cudaMemcpyHostToDevice) == cudaSuccess);
}

void CreateSepiaImageOnGPU(ImagePixel* inputPixels, ImagePixel* outputPixels, unsigned int width, unsigned int height)
{
	int pixelsCount = width * height;
	int pixelsBytes = sizeof(ImagePixel) * pixelsCount;

	// Output pixels on device - malloc
	ImagePixel* outputPixelsDevice;
	MallocAndCopyPixelsToDevice(outputPixels, &outputPixelsDevice, pixelsBytes);

	// Input pixels on device - malloc and copy to VRAM
	ImagePixel* inputPixelsDevice;
	MallocAndCopyPixelsToDevice(inputPixels, &inputPixelsDevice, pixelsBytes);

	dim3 threads(32, 32);
	dim3 blocks(width / threads.x, height / threads.y);

	CreateSepiaImageKernel<<<blocks, threads>>>(inputPixelsDevice, outputPixelsDevice, width, height);
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

#undef min
#undef max