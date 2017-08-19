#include "stdafx.h"
#include "ImageBinarizer.h"


void ThresholdImageBinarizer::BinarizeImage(const Mat &inputImage, Mat &outputImage)
{
	outputImage.create(inputImage.size(), inputImage.type());
	threshold(inputImage,outputImage,0, 1, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
}


void AdaptiveThresholdImageBinarizer::BinarizeImage(const Mat &inputImage, Mat &outputImage)
{
	outputImage.create(inputImage.size(), inputImage.type());
	adaptiveThreshold(inputImage,outputImage, 1, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV,m_nBlockSize,m_C);
}




