#pragma once

#include "GeneralDefinitions.h"
#include <vector>


using namespace std;


#ifndef LINUX
#include "aligned_allocator.h"
#include "windows.h"
// Mapping file to memory
#define CLOSE_WIN_HANDLE(handle) if (handle) CloseHandle(handle);
#else
#define CLOSE_WIN_HANDLE(handle)
#endif

bool MapFile(const char *fileName, HANDLE *pHandle,char** ppMappedFile);

// Get file size in bytes
int GetFileSize(const char *fileName);

// Various utilities for STL data structures
void VectorOfVectorsToMatrix(vector<vector<real, CUSTOM_ALLOCATOR(real)> > &vectorOfVectors, vector<real, CUSTOM_ALLOCATOR(real)>& matrix);




void LogSoftMax(real *pIn, real *pOut, unsigned int numElements);
void SoftMax(real *pIn, real *pOut, unsigned int numElements);

unsigned int SVHNGetBestGuess(vector<real,CUSTOM_ALLOCATOR(real)> &output);

float dotproduct_autovectorized(float *pSrc, float *pKernel, unsigned int len);

template <unsigned int kernelX, unsigned int kernelY, unsigned int stride>
void ConvolveImageImpl(real * __restrict inputFeatureMap, real * __restrict outputFeatureMap, real * __restrict kernel, unsigned int nInputFeatureMapWidth, unsigned int nInputFeatureMapHeight)
{
	unsigned int count = 0;

	for (unsigned int hItr = 0; hItr < (nInputFeatureMapHeight - kernelY) + 1; hItr+=stride)
	{
		for (unsigned int wItr = 0; wItr < (nInputFeatureMapWidth - kernelX) +1 ; wItr+=stride)
		{
			real sum = (real)0.0f;

			for (unsigned int kH = 0; kH < kernelY; kH++)
			{
				for (unsigned int kW = 0; kW < kernelX; kW++)
				{
					// Those castings are a must for Intel compiler vectorizer. 
					// I could't figure out why, but apparently, without these, the vectorized code would convert everything to double,
					// do the calculations and then convert back to single precision, which is slow as hell.
					sum += (real)((real)kernel[kernelY*kH+kW]*(real)inputFeatureMap[(nInputFeatureMapHeight*hItr+wItr) + kH*nInputFeatureMapWidth+ kW]);
				}
			}
			outputFeatureMap[count++]+=sum;
		}
	}
}



void MaxPooling( 
	real * __restrict inputFeatureMaps, 
	unsigned int numFeatureMaps, 
	unsigned int ifmX, 
	unsigned int ifmY, 
	real * __restrict outputFeatureMaps, 
	unsigned int kernelX, 
	unsigned int kernelY,
	unsigned int stride
	);

void gen_random(char *s, const int len);
