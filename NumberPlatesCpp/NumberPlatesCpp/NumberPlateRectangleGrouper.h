#pragma once

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


class RectangleGroup
{
public:
	RectangleGroup(void) 
	{
		m_rectanglesInGroup.clear();
	}
	virtual ~RectangleGroup(void) {}

	void AddToGroup(const Rect &rect);
	virtual bool BelongsToGroup(const Rect &rect) = 0;
	
	vector<Rect> &  GetGroupRectangles() ;

	const Rect& GetAverageRectangle()
	{
		return m_AvgRect;
	}


protected:
	Rect m_AvgRect;

private:
	vector<Rect> m_rectanglesInGroup;
};


class NumberPlateRectanglesGroup : public RectangleGroup
{
public:
	NumberPlateRectanglesGroup(float y_diff_allowance, float height_allowance, int distanceFactorAllowance) : 
		m_fY_diff_allowance(y_diff_allowance), m_fHeight_allowance (height_allowance), m_nDistanceFactorAllowance(distanceFactorAllowance) {}

	virtual ~NumberPlateRectanglesGroup(void) {}

	bool BelongsToGroup(const Rect &rect);

private:
	float m_fY_diff_allowance;
	float m_fHeight_allowance;
	int m_nDistanceFactorAllowance;
};



class NumberPlateRectangleGrouper
{
public:
	NumberPlateRectangleGrouper(float y_diff_allowance, float height_allowance, int distanceFactorAllowance) : 
		m_fY_diff_allowance(y_diff_allowance), m_fHeight_allowance (height_allowance), m_nDistanceFactorAllowance(distanceFactorAllowance) {}
	
	virtual ~NumberPlateRectangleGrouper(void);

	void Group(vector<Rect> &rectanglesToGroup);

	vector<NumberPlateRectanglesGroup> &GetGroups()
	{
		return m_vectorGroups;
	}

	void Clear()
	{
		m_vectorGroups.clear();
	}

private:
	vector<NumberPlateRectanglesGroup> m_vectorGroups;
	float m_fY_diff_allowance;
	float m_fHeight_allowance;
	int m_nDistanceFactorAllowance;
};

