#include "stdafx.h"
#include "Sigmoids.h"

#include <math.h>


void TanHSigmoid::operator()(vector<real, CUSTOM_ALLOCATOR(real)> &neurons)
{
	unsigned int length = neurons.size();

	for (unsigned int itr = 0; itr < neurons.size(); itr++)
	{
		neurons[itr] = a*tanh(b*neurons[itr]);
	}
}

real TanHSigmoid::operator()(real neuron)
{
	return (a*tanh((float)b*(float)neuron));
}


void ThresholdSigmoid::operator()(vector<real, CUSTOM_ALLOCATOR(real)> &neurons) 
{

	for (unsigned int itr = 0; itr < neurons.size(); itr++)
	{
		neurons[itr] = neurons[itr] > m_ThresholdVal ? neurons[itr] : m_Val;
	}

}


real ThresholdSigmoid::operator()(real neuron)
{
	if (neuron > m_ThresholdVal) 
	{
		return neuron;
	}
	else
	{
		return m_Val;
	}
}


void Logistic::operator()(vector<real, CUSTOM_ALLOCATOR(real)> &neurons)
{
	for (unsigned int itr = 0; itr < neurons.size(); itr++)
	{
		neurons[itr] = (*this)(neurons[itr]);
	}
}

real Logistic::operator()(real neuron)
{
	return (real)1.0f/((real)1.0f+(real)exp(-1.0f*neuron));
}
