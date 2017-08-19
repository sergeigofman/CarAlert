#pragma once
#include "GeneralDefinitions.h"
#include <vector>
using namespace std;

#ifndef LINUX
#include "aligned_allocator.h"
#endif

class SigmoidsBase
{
public:
	virtual ~SigmoidsBase(void) {}

	virtual void operator()(vector<real, CUSTOM_ALLOCATOR(real)> &neurons) = 0;
	virtual real operator()(real neuron) = 0;
};

class TanHSigmoid : public SigmoidsBase
{
public:
	TanHSigmoid() : a(1.7159f), b(0.66666667f) {}
	TanHSigmoid(real factor, real multiplier) : a(factor), b(multiplier) {}

	virtual ~TanHSigmoid(void){}
	void operator()(vector<real, CUSTOM_ALLOCATOR(real)> &neurons);
	real operator()(real neuron);

private:
	real a, b;

};

class EmptySigmoid : public SigmoidsBase
{
	virtual ~EmptySigmoid(void) {}
	void operator()(vector<real, CUSTOM_ALLOCATOR(real)> &neurons) {}
	real operator()(real neuron) {return neuron;}
};

class ThresholdSigmoid: public SigmoidsBase
{
public:
	ThresholdSigmoid() : m_ThresholdVal(0.000001f),m_Val(0.0f) {};
	ThresholdSigmoid(real threshold) : m_ThresholdVal(threshold), m_Val(0.0f)  {};
	ThresholdSigmoid(real threshold, real value) : m_ThresholdVal(threshold), m_Val(value)  {};

	virtual ~ThresholdSigmoid(void) {}
	void operator()(vector<real, CUSTOM_ALLOCATOR(real)> &neurons);
	real operator()(real neuron);

private:
	real m_ThresholdVal;
	real m_Val;
};

class Logistic : public SigmoidsBase
{
	virtual ~Logistic(void) {}
	void operator()(vector<real, CUSTOM_ALLOCATOR(real)> &neurons);
	real operator()(real neuron);
};
