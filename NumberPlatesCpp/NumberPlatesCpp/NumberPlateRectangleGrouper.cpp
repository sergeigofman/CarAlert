#include "stdafx.h"
#include "NumberPlateRectangleGrouper.h"


void RectangleGroup::AddToGroup(const Rect &rect)
{
	m_AvgRect.x = (m_AvgRect.x * m_rectanglesInGroup.size() + rect.x)/(m_rectanglesInGroup.size()+1);
	m_AvgRect.y = (m_AvgRect.y * m_rectanglesInGroup.size() + rect.y)/(m_rectanglesInGroup.size()+1);
	m_AvgRect.width = (m_AvgRect.width * m_rectanglesInGroup.size() + rect.width)/(m_rectanglesInGroup.size()+1);
	m_AvgRect.height = (m_AvgRect.height * m_rectanglesInGroup.size() + rect.height)/(m_rectanglesInGroup.size()+1);
	m_rectanglesInGroup.push_back(rect);
}



NumberPlateRectangleGrouper::~NumberPlateRectangleGrouper(void)
{
}


bool NumberPlateRectanglesGroup::BelongsToGroup(const Rect &rect)
{
	// Criteria
	int y_diff = abs(rect.y - m_AvgRect.y);
	int height_diff = abs(rect.height - m_AvgRect.height);
	int x_diff = abs(rect.x - m_AvgRect.x);
	if (x_diff > m_nDistanceFactorAllowance*rect.height)
	{
		return false;
	}
	return 	(y_diff	< m_fY_diff_allowance * m_AvgRect.height) && (height_diff < m_fHeight_allowance * m_AvgRect.height);
}

bool compareRectsByX(const Rect &a, const Rect &b)
{
    return a.x < b.x;
}

void NumberPlateRectangleGrouper::Group(vector<Rect> &rectanglesToGroup)
{
	sort(rectanglesToGroup.begin(), rectanglesToGroup.end(), compareRectsByX);
	for (vector<Rect>::const_iterator itr = rectanglesToGroup.begin(); itr != rectanglesToGroup.end(); itr++)
	{
		vector<NumberPlateRectanglesGroup>::iterator groupsItr = m_vectorGroups.begin();
		for (; groupsItr < m_vectorGroups.end(); groupsItr++)
		{
			if (groupsItr->BelongsToGroup(*itr))
			{
				groupsItr->AddToGroup(*itr);
				break;
			}
		}

		if(groupsItr == m_vectorGroups.end())
		{
			// Did not find any group to add to - create a new group
			NumberPlateRectanglesGroup newGroup(m_fY_diff_allowance, m_fHeight_allowance, m_nDistanceFactorAllowance);
			newGroup.AddToGroup(*itr);
			m_vectorGroups.push_back(newGroup);
		}
	}
}


vector<Rect>&  RectangleGroup::GetGroupRectangles() 
{
	sort(m_rectanglesInGroup.begin(), m_rectanglesInGroup.end(), compareRectsByX);
	return m_rectanglesInGroup;
}