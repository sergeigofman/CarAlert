#pragma once

#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

class ImageBinarizer
{

public:
	ImageBinarizer(void) {}
	virtual ~ImageBinarizer(void) {}

	virtual void BinarizeImage(const Mat &inputImage, Mat &outputImage) = 0;


};

class ThresholdImageBinarizer : public ImageBinarizer
{
public:
	ThresholdImageBinarizer(void) : ImageBinarizer() {}
	virtual ~ThresholdImageBinarizer(void){}

	void BinarizeImage(const Mat &inputImage, Mat &outputImage);
};

class AdaptiveThresholdImageBinarizer : public ImageBinarizer
{
private:
	int m_nBlockSize;
	double m_C;
public:
	AdaptiveThresholdImageBinarizer(int blockSize, double C) : m_nBlockSize(blockSize), m_C (C), ImageBinarizer() {}
	virtual ~AdaptiveThresholdImageBinarizer (void){}

	void BinarizeImage(const Mat &inputImage, Mat &outputImage);
};

