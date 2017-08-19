#include "stdafx.h"
#include "BlobDetector.h"


BlobDetector::~BlobDetector(void)
{
}


int BlobDetector::DetectBlobs(Mat &binary, vector<Rect> &boundingBoxes)
{
 
    // Using labels from 2+ for each blob
    Mat label_image;
    binary.convertTo(label_image, CV_32FC1);
 
    int label_count = 2; // starts at 2 because 0,1 are used already
 
    for(int y=0; y < binary.rows; y++) 
	{
        for(int x=0; x < binary.cols; x++) 
		{
            if((int)label_image.at<float>(y,x) != 1) 
			{
                continue;
            }
 
            Rect rect;
            floodFill(label_image, Point(x,y), Scalar(label_count), &rect, Scalar(0), Scalar(0), 4);
			if ( rect.height > binary.rows * m_fHeightRatioLowerBound && 
				 rect.height < binary.rows * m_fHeightRatioUpperBound && 
				 rect.width < binary.cols * m_fWidthRatioUpperBound)
			{
				if ((float)rect.width / (float)rect.height < m_fBboxWidthToHeightRatioUpperBound)
				{
					boundingBoxes.push_back(rect);
					if (m_bDrawBboxesOnInputImage)
					{
						rectangle(binary, rect, Scalar(1),1);
					}
				}
			}
            label_count++;
        }
    }
	return label_count;
}