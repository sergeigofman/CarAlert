#pragma once


#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

class BlobDetector
{
public:
	BlobDetector(
		float heightRatioLowerBound, // relative to the entire image's height 
		float heightRatioUpperBound, // relative to the entire image's height 
		float widthRatioUpperBound, // relative to the entire image's height 
		float bboxWidthToHeightRatioUpperBound,
		bool drawBboxesOnInputImage // Debug feature
		) : 
	m_fHeightRatioLowerBound(heightRatioLowerBound),
	m_fHeightRatioUpperBound(heightRatioUpperBound),
	m_fWidthRatioUpperBound(widthRatioUpperBound),
	m_fBboxWidthToHeightRatioUpperBound(bboxWidthToHeightRatioUpperBound),
	m_bDrawBboxesOnInputImage(drawBboxesOnInputImage) {}


	virtual ~BlobDetector(void);

	int DetectBlobs(Mat &binary, vector<Rect> &boundingBoxes);

private:
	float m_fHeightRatioLowerBound; // relative to the entire image's height 
	float m_fHeightRatioUpperBound; // relative to the entire image's height 
	float m_fWidthRatioUpperBound; // relative to the entire image's height 
	float m_fBboxWidthToHeightRatioUpperBound;
	bool  m_bDrawBboxesOnInputImage; // Debug feature

};

