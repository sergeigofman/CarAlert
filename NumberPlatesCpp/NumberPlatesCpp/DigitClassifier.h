#pragma once

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#include "NeuralNetwork.h"


class DigitClassifier
{
public:
	DigitClassifier(bool bSaveImageOfDigitWithLabel, float paddingRatio = 0.1f);
	~DigitClassifier(void);
	unsigned int ClassifyDigit(const Mat &img, const Rect &bbox);

private:
	NeuralNetwork<NNLayerBase> nnSVHN;
	bool m_bSaveImageOfDigitWithLabel;
	float m_fPaddingRatio;
};

