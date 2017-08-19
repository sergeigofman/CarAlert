#include "stdafx.h"
#include "DigitClassifier.h"
#include "Utils.h"

DigitClassifier::DigitClassifier(bool bSaveImageOfDigitWithLabel, float paddingRatio) : 
	m_bSaveImageOfDigitWithLabel(bSaveImageOfDigitWithLabel), m_fPaddingRatio(paddingRatio) 
{
	cout << "Building Neural Net" << endl;

	bool bSucceded = BuildSVHNNetwork(nnSVHN,"");
	if (!bSucceded)
	{
		exit(-1);
	}

}


DigitClassifier::~DigitClassifier(void)
{
}


unsigned int DigitClassifier::ClassifyDigit(const Mat &img, const Rect &bbox)
{
	Mat src,digit;// = img(rect);
	int diff = abs(bbox.height - bbox.width);
	int padding;
	if (bbox.height > bbox.width)
	{
		padding = (int)(m_fPaddingRatio*bbox.height);
		copyMakeBorder(img(bbox),digit,padding,padding,diff/2+padding,diff/2+padding,BORDER_CONSTANT);
	}
	else
	{
		// Not expecting to be here...
		padding = (int)(m_fPaddingRatio*bbox.width);
		copyMakeBorder(img(bbox),digit,diff/2+padding,diff/2+padding,padding,padding,BORDER_CONSTANT);
	}

	resize(digit,digit,Size(32,32));
	cvtColor(digit, digit, CV_BGR2YCrCb);

	for (int rowItr = 0; rowItr < 32; rowItr++)
	{
		for (int colItr = 0; colItr < 32; colItr++)
		{
			nnSVHN.m_Layers[0]->m_NeuronOutputs[0][rowItr*32+colItr] = (float)(255 - digit.at<Vec3b>(rowItr,colItr)[0])/128.0f - 1.0f;
			nnSVHN.m_Layers[0]->m_NeuronOutputs[0][32*32+rowItr*32+colItr] = (float)(255 - digit.at<Vec3b>(rowItr,colItr)[1])/128.0f - 1.0f;
			nnSVHN.m_Layers[0]->m_NeuronOutputs[0][2*32*32+rowItr*32+colItr] = (float)(255 - digit.at<Vec3b>(rowItr,colItr)[2])/128.0f - 1.0f;
		}
	}

	nnSVHN.ForwardPropagate(0);

	unsigned int bestGuess = SVHNGetBestGuess(nnSVHN.m_Layers[6]->m_NeuronOutputs[0]);

	if (m_bSaveImageOfDigitWithLabel)
	{
#ifndef LINUX
		char label[10];
		_itoa_s(bestGuess,label,10,10);
		char fileName[20];
		gen_random(fileName,6);
		strcat_s(fileName,20,".");
		strcat_s(fileName,20,label);
		strcat_s(fileName,20,".jpg");
		imwrite(fileName,digit);
#endif
	}

	return bestGuess;
}
