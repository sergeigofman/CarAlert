#include "stdafx.h"
#include "Utils.h"

#include <fstream>
#include <iostream>
#include <string>
using namespace std;
#include "GeneralDefinitions.h"
#include <assert.h>

#ifdef LINUX
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#endif

#ifndef LINUX
#include "windows.h"


bool MapFile(const char *fileName, HANDLE *pHandle,char** ppMappedFile)
{
	HANDLE hFile = ::CreateFileA(
		fileName,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);

	if (!hFile)
	{
		return false;
	}


	*pHandle = ::CreateFileMapping(
		hFile,
		NULL,
		PAGE_READONLY,
		0,
		0,
		NULL);
	if (!*pHandle)
	{
		return false;
	}

	*ppMappedFile = (char *)::MapViewOfFile(*pHandle,FILE_MAP_READ,0,0,0);
	if (!*ppMappedFile)
	{
		return false;
	}

	::CloseHandle(hFile);
	return true;

}

int GetFileSize(const char *fileName)
{
    BOOL                        fOk;
    WIN32_FILE_ATTRIBUTE_DATA   fileInfo;

    if (NULL == fileName)
        return -1;

    fOk = GetFileAttributesExA(fileName, GetFileExInfoStandard, (void*)&fileInfo);
    if (!fOk)
        return -1;
    ASSERT(0 == fileInfo.nFileSizeHigh);
    return (int)fileInfo.nFileSizeLow;
}
#else

static void
check (int test, const char * message, ...)
{
    if (test) {
        va_list args;
        va_start (args, message);
        vfprintf (stderr, message, args);
        va_end (args);
        fprintf (stderr, "\n");
        exit (EXIT_FAILURE);
    }
}
// Implementig Linux versions here
bool MapFile(const char *fileName, HANDLE *pHandle,char** ppMappedFile)
{

	int size = GetFileSize(fileName);

	if (size < 0)
	{
		return false;
	}


    int fd = open(fileName, O_RDONLY);
    if (fd < 0)
    {
    	return false;
    }
	void *mapped = mmap (0, size, PROT_READ, MAP_PRIVATE, fd, 0);
    check (mapped == MAP_FAILED, "mmap %s failed: %s",
    		fileName, strerror (errno));
	if (mapped == MAP_FAILED)
	{
		return false;
	}

	*ppMappedFile = (char *)mapped;
	return true;
}

int GetFileSize(const char *fileName)
{
    if (NULL == fileName)
        return -1;

    int fd = open(fileName, O_RDONLY);
    if (fd < 0)
    {
    	return fd;
    }
    struct stat s;
    int status = fstat (fd, & s);
    if (status < 0)
    {
    	return status;
    }
    close(fd);
    return s.st_size;
}

#endif

unsigned int SVHNGetBestGuess(vector<real,CUSTOM_ALLOCATOR(real)> &output)
{
	real max = -FLT_MAX;
	unsigned int maxIndex = 0;
	for (unsigned int itr = 0; itr < output.size(); itr++)
	{
		if (output[itr] > max)
		{
			maxIndex = itr;
			max = output[itr];
		}
	}
	
	return (maxIndex+1)%10;
}



#ifndef max
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#endif

/* Credits to Leon Bottou */
real THExpMinusApprox(real x)
{
#define EXACT_EXPONENTIAL 0
#if EXACT_EXPONENTIAL
  return exp(-x);
#else
  /* fast approximation of exp(-x) for x positive */
# define A0   (1.0f)
# define A1   (0.125f)
# define A2   (0.0078125f)
# define A3   (0.00032552083f)
# define A4   (1.0172526e-5f)
  if (x < 13.0f)
  {
/*    assert(x>=0); */
    real y;
    y = A0+x*(A1+x*(A2+x*(A3+x*A4)));
    y *= y;
    y *= y;
    y *= y;
    y = 1/y;
    return y;
  }
  return 0;
# undef A0
# undef A1
# undef A2
# undef A3
# undef A4
#endif
}


void LogSoftMax(real *pIn, real *pOut, unsigned int numElements)
{
	real sumExp = 0;
	for (unsigned int itr = 0; itr < numElements; itr++)
	{
		sumExp+=exp(pIn[itr]);
	}

	for (unsigned int itr = 0; itr < numElements; itr++)
	{
		pOut[itr] = log(exp(pIn[itr])/sumExp);
	}
}

void SoftMax(real *pIn, real *pOut, unsigned int numElements)
{
	real sumExp = 0;
	for (unsigned int itr = 0; itr < numElements; itr++)
	{
		sumExp+=exp(pIn[itr]);
	}

	for (unsigned int itr = 0; itr < numElements; itr++)
	{
		pOut[itr] = exp(pIn[itr])/sumExp;
	}
}

void VectorOfVectorsToMatrix(vector<vector<real, CUSTOM_ALLOCATOR(real)> > &vectorOfVectors, vector<real, CUSTOM_ALLOCATOR(real)>& matrix)
{
	matrix.clear();

	unsigned int lineLength = vectorOfVectors[0].size();
	unsigned int matrixLength = vectorOfVectors.size() * lineLength ;

	matrix.resize(matrixLength + RESERVED_SPACE);
	for (unsigned int itr = 0; itr < vectorOfVectors.size(); itr++)
	{
		vector<real, CUSTOM_ALLOCATOR(real)> &curLine = vectorOfVectors[itr];
		memcpy(&matrix[itr*lineLength],&curLine[0],lineLength *sizeof (real));
	}
}

float dotproduct_autovectorized(float * __restrict pSrc, float * __restrict pKernel, unsigned int len)
{
	float res = 0;
	 
	for (unsigned int itr = 0; itr < len; itr++)
	{
		res+=pSrc[itr]*pKernel[itr];
	}
	return res;
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
	)
{
	unsigned int ofmX = (ifmX - kernelX)/stride +1 ;
	unsigned int ofmY = (ifmY - kernelY)/stride +1 ;

	for (unsigned int fmItr = 0; fmItr < numFeatureMaps; fmItr++)
	{ // Go over all input feature maps
		for (unsigned int hItr = 0; hItr < (ifmY - kernelY) + 1; hItr+=stride)
		{ // For each input feature map, go over all locations where the kernel-sized stencil would fit - in both dimensions, y...
			for (unsigned int wItr = 0; wItr < (ifmX - kernelX) +1 ; wItr+=stride)
			{ // and x...
				real maxValue = -REAL_MAX;
				for (unsigned int kH = 0; kH < kernelY; kH++)
				{ 
					for (unsigned int kW = 0; kW < kernelX; kW++)
					{ // For each stencil placement, compute 2D convolution at the placement

						real ifm_pixel = inputFeatureMaps[(fmItr * ifmX * ifmY + (ifmX*hItr+wItr) + kH*ifmX + kW)];
						maxValue = max(ifm_pixel,maxValue);
					}
				}
				outputFeatureMaps[(fmItr * ofmX * ofmY + (hItr/stride) * ofmX + (wItr/stride))] = maxValue;							
			}
		}
	}
}


void gen_random(char *s, const int len) 
{
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    s[len] = 0;
}
