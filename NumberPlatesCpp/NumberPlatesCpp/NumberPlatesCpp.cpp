// NumberPlatesCpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

#include "ImageBinarizer.h"
#include "BlobDetector.h"
#include "NumberPlateRectangleGrouper.h"
#include "DigitClassifier.h"


void getSobelKernels( OutputArray _kx, OutputArray _ky,
                             int dx, int dy, int _ksize, bool normalize, int ktype )
{
    int i, j, ksizeX = _ksize, ksizeY = _ksize;
    if( ksizeX == 1 && dx > 0 )
        ksizeX = 3;
    if( ksizeY == 1 && dy > 0 )
        ksizeY = 3;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );

    _kx.create(ksizeX, 1, ktype, -1, true);
    _ky.create(ksizeY, 1, ktype, -1, true);
    Mat kx = _kx.getMat();
    Mat ky = _ky.getMat();

    if( _ksize % 2 == 0 || _ksize > 31 )
        CV_Error( CV_StsOutOfRange, "The kernel size must be odd and not larger than 31" );
    std::vector<int> kerI(std::max(ksizeX, ksizeY) + 1);

    CV_Assert( dx >= 0 && dy >= 0 && dx+dy > 0 );

    for( int k = 0; k < 2; k++ )
    {
        Mat* kernel = k == 0 ? &kx : &ky;
        int order = k == 0 ? dx : dy;
        int ksize = k == 0 ? ksizeX : ksizeY;

        CV_Assert( ksize > order );

        if( ksize == 1 )
            kerI[0] = 1;
        else if( ksize == 3 )
        {
            if( order == 0 )
                kerI[0] = 1, kerI[1] = 2, kerI[2] = 1;
            else if( order == 1 )
                kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;
            else
                kerI[0] = 1, kerI[1] = -2, kerI[2] = 1;
        }
        else
        {
            int oldval, newval;
            kerI[0] = 1;
            for( i = 0; i < ksize; i++ )
                kerI[i+1] = 0;

            for( i = 0; i < ksize - order - 1; i++ )
            {
                oldval = kerI[0];
                for( j = 1; j <= ksize; j++ )
                {
                    newval = kerI[j]+kerI[j-1];
                    kerI[j-1] = oldval;
                    oldval = newval;
                }
            }

            for( i = 0; i < order; i++ )
            {
                oldval = -kerI[0];
                for( j = 1; j <= ksize; j++ )
                {
                    newval = kerI[j-1] - kerI[j];
                    kerI[j-1] = oldval;
                    oldval = newval;
                }
            }
        }

        Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
        double scale = !normalize ? 1. : 1./(1 << (ksize-order-1));
        temp.convertTo(*kernel, ktype, scale);
    }
}

bool GetNumberPlate(Mat &image, vector<NumberPlateRectanglesGroup> &groups, DigitClassifier &digitClassifier, string &numberPlate)
{
	ostringstream result;
	bool bRes = false;
	for (vector<NumberPlateRectanglesGroup>::iterator groupsItr = groups.begin();
		groupsItr != groups.end(); groupsItr++)
	{
		const vector<Rect> &groupRectangles = groupsItr->GetGroupRectangles();
		//cout << "Group count = " << groupRectangles.size() << endl;
		if (groupRectangles.size() >= 7 && groupRectangles.size() <= 9 )
		{
			cout << "Found number plate!" << endl;
			bRes = true;
			for (vector<Rect>::const_iterator numbersItr = groupRectangles.begin(); numbersItr < groupRectangles.end(); numbersItr++)
			{
				unsigned int digit = digitClassifier.ClassifyDigit(image, *numbersItr);
				result << digit;
			}
			result << "\r\n";
		}

	}
	numberPlate = result.str();
	return bRes;
}

int main(int argc, char* argv[])
{
    if( argc != 2)
    {
     cout <<" Usage: NumberPlatesCpp image_to_look_for_number_plates_in" << endl;
     return -1;
    }
	/*
	 Mat kd, ks;
     getSobelKernels( kd, ks, 2, 2, 5, false, CV_32F  );

	 cout << kd << endl << ks << endl;
	 */

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	Rect rect(image.cols/4,image.rows/3,2*image.cols/4,image.rows/3);
	//Rect rect(0,0,image.cols,image.rows);

	Mat croppedImage = image(rect);

	//imwrite("croppedImage.jpg",croppedImage);
	
	//Grayscale matrix
    Mat grayscaleMat (croppedImage.size(), CV_8U);

    //Convert BGR to Gray
    cvtColor( croppedImage, grayscaleMat, CV_BGR2GRAY );
	//GaussianBlur(grayscaleMat, grayscaleMat, Size(3,3),0);

	cout << "Trying binary threshold" << endl;
	Mat binaryMat;
	ThresholdImageBinarizer thresholdBinarizer;
	thresholdBinarizer.BinarizeImage(grayscaleMat, binaryMat);

	BlobDetector blobDetector(0.02f,0.4f,0.2f,0.8f,true);
	vector<Rect> boundingBoxes;

	blobDetector.DetectBlobs(binaryMat,boundingBoxes);

	//imwrite("out_bbs_binary.jpg",255*binaryMat);
	
	NumberPlateRectangleGrouper grouper(0.8f,0.2f,4);

	grouper.Group(boundingBoxes);
	DigitClassifier digitClassifier(false);

	vector<NumberPlateRectanglesGroup> &groups = grouper.GetGroups();
	
	string numberPlate;
	bool bRes = GetNumberPlate(croppedImage,groups,digitClassifier, numberPlate);

	if (bRes)
	{
		cout << numberPlate << endl;
		return 1;
	}

	cout << "Trying adaptive threshold" << endl;
	boundingBoxes.clear();

	AdaptiveThresholdImageBinarizer adaptiveThresholdBinarizer(51,15);
	adaptiveThresholdBinarizer.BinarizeImage(grayscaleMat, binaryMat);

	blobDetector.DetectBlobs(binaryMat,boundingBoxes);

	//imwrite("out_bbs_adaptive.jpg",255*binaryMat);
    
	grouper.Clear();
	grouper.Group(boundingBoxes);

	groups = grouper.GetGroups();
	bRes = GetNumberPlate(croppedImage,groups,digitClassifier, numberPlate);

	if (bRes)
	{
		cout << numberPlate << endl;
		return 1;
	}

	return 0;
}

