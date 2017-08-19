// NeuralNetwork.h: interface for the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_)
#define AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_

#pragma once

#include "GeneralDefinitions.h"

#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
using namespace std;

#define SIGMOID(x) (1.7159f*tanh((float)0.66666667f*(float)x))
#define DSIGMOID(S) (0.66666667f/1.7159f*(1.7159f+(S))*(1.7159f-(S)))  // derivative of the sigmoid as a function of the sigmoid's output

#ifndef LINUX
#include "aligned_allocator.h"
#endif

#include "Sigmoids.h"

// forward declarations

class NNLayerBase;

template <class Layer>
class NeuralNetwork  
{
public:

	void ForwardPropagate(real* inputVector, unsigned int count);
	void ForwardPropagate(unsigned int nThreadIndex);

	void Save(string fileName);
	void Restore(string fileName);
	void Save(ofstream &file);
	void Restore (ifstream &file);

	NeuralNetwork<Layer>();
	virtual ~NeuralNetwork<Layer>();
	void Initialize();

	vector <Layer *> m_Layers;


};

///////////////////////////////////////////////////////////////////////
//
//  NeuralNetwork class definition

template <class Layer>
NeuralNetwork<Layer>::NeuralNetwork() 
{
	Initialize();
}

template <class Layer>
void NeuralNetwork<Layer>::Initialize()
{
	// delete all layers
	
	typename vector<Layer *>::iterator it;
	
	for( it=m_Layers.begin(); it<m_Layers.end(); it++ )
	{
		delete *it;
	}
	
	m_Layers.clear();
	
}

template <class Layer>
NeuralNetwork<Layer>::~NeuralNetwork()
{
	Initialize();
}

template <class Layer>
void NeuralNetwork<Layer>::ForwardPropagate(real* inputVector, unsigned int iCount )
{
	typename vector<Layer *>::iterator lit = m_Layers.begin();
	
	// first layer is input layer: directly set outputs of all of its neurons
	// to the input vector
	
	if ( lit<m_Layers.end() )  
	{
		ASSERT((*lit)->m_NeuronOutputs[0].size() == iCount);
		memcpy(&((*lit)->m_NeuronOutputs[0][0]),inputVector, iCount * sizeof((*lit)->m_NeuronOutputs[0][0]));
	}	
	
	ForwardPropagate(0);	
}

template <class Layer>
void NeuralNetwork<Layer>::ForwardPropagate(unsigned int nThreadIndex)
{
	typename vector<Layer*>::iterator lit = m_Layers.begin();
	int count = 1;

	for( lit; lit<m_Layers.end(); lit++ )
	{
#if COUNT_NUM_CALCULATIONS
		cout << "Layer " << count << endl;
#endif
		(*lit)->ForwardPropagate(nThreadIndex);
		count++;
	}
}


template <class Layer>
void NeuralNetwork<Layer>::Save(ofstream &file)
{
	// TODO
}

template <class Layer>
void NeuralNetwork<Layer>::Restore (ifstream &file)
{
	// TODO
}

template <class Layer>
void NeuralNetwork<Layer>::Save(string fileName)
{
	ofstream file(fileName);
	if (file.good())
	{
		Save(file);
	}
	else
	{
		// SG-TODO:throw exception here...
	}
}


template <class Layer>
void NeuralNetwork<Layer>::Restore(string fileName)
{
	ifstream file(fileName);
	if (file.good())
	{
		Restore(file);
	}
	else
	{
		// SG-TODO:throw exception here...
	}
}




class NNLayerBase
{
public:
	// Constructor/destructor
	NNLayerBase() : m_strLabel(""), m_pPrevLayer(NULL), m_pSigmoid(NULL)
	{
	};
	NNLayerBase( const char *label, NNLayerBase* pPrev = NULL, SigmoidsBase *pSigmoid = NULL ): m_strLabel(label), m_pPrevLayer(pPrev), m_pSigmoid(pSigmoid) {};
	virtual ~NNLayerBase() 
	{
		if (m_pSigmoid)
		{
			delete m_pSigmoid;
		}
	};
	// Methods
	virtual void ForwardPropagate(unsigned int nThreadIndex) = 0;

	virtual void Save(ofstream &file) = 0;
	virtual void Restore (ifstream &file) = 0;


	// State
	vector<real, CUSTOM_ALLOCATOR(real)> m_NeuronOutputs[MAX_NUM_THREADS * BATCH_SIZE];

protected:
	string m_strLabel;
	NNLayerBase* m_pPrevLayer;

	SigmoidsBase *m_pSigmoid;
};

// Input layer - only the m_NeuronOutputs matters
class NNInputLayer : public NNLayerBase
{
public:
	NNInputLayer():NNLayerBase() {};
	virtual ~NNInputLayer() {};

	// Methods
	virtual void ForwardPropagate(unsigned int nThreadIndex);

	virtual void Save(ofstream &file) {};
	virtual void Restore (ifstream &file) {};

};


// Dummy class to hold gradientOutput of the network
class NNGradientHolderLayer : public NNLayerBase
{
public:
	NNGradientHolderLayer ():NNLayerBase() {};
	virtual ~NNGradientHolderLayer () {};

	// Methods
	virtual void ForwardPropagate(unsigned int nThreadIndex){};

	virtual void Save(ofstream &file) {};
	virtual void Restore (ifstream &file) {};
};


class FullyConnectedNNLayer: public NNLayerBase
{
public:
	
	FullyConnectedNNLayer(		
		NNLayerBase *prevLayer, 
		SigmoidsBase *pSigmoid,
		unsigned int nOutputNeurons,
		float *parameters,
		unsigned int pctOfWeightsToKeep = 100);

	void ForwardPropagate(unsigned int nThreadIndex);
	virtual ~FullyConnectedNNLayer(){};
	void Save(ofstream &file);
	void Restore (ifstream &file);


	// Temporary, needs to go back to private
public:
	vector<real, CUSTOM_ALLOCATOR(real)> m_WeightsMatrix;
	vector<real, CUSTOM_ALLOCATOR(real)> m_ErrorGradientWrtWeights[MAX_NUM_THREADS];

	vector<real, CUSTOM_ALLOCATOR(real)> m_Biases;
	vector<real, CUSTOM_ALLOCATOR(real)> m_ErrorGradientWrtBias[MAX_NUM_THREADS];

	vector<vector<real, CUSTOM_ALLOCATOR(real)> > m_VectorWeights;

};


class MaxPoolingLayer : public NNLayerBase
{
public:
	
	MaxPoolingLayer(
		NNLayerBase *prevLayer, 
		SigmoidsBase *pSigmoid,
		unsigned int nInputFeatureMaps,
		unsigned int nInputFeatureMapWidth,
		unsigned int nInputFeatureMapHeight,
		unsigned int nStride
		);
	virtual ~MaxPoolingLayer(){};

	void ForwardPropagate(unsigned int nThreadIndex);
	void Save(ofstream &file) {};
	void Restore (ifstream &file) {};

private:

	// internal state
	unsigned int m_nInputFeatureMaps;
	unsigned int m_nInputFeatureMapWidth;
	unsigned int m_nInputFeatureMapHeight;
	unsigned int m_nStride;
	unsigned int m_nOutputFeatureMapWidth;
	unsigned int m_nOutputFeatureMapHeight;
};

class ConvolutionalLayer: public NNLayerBase
{
public:
	ConvolutionalLayer(
		NNLayerBase *prevLayer, 
		SigmoidsBase *pSigmoid,
		unsigned int nInputFeatureMaps,
		unsigned int nInputFeatureMapWidth,
		unsigned int nInputFeatureMapHeight,
		unsigned int nStride,
		unsigned int nKernelWidth,
		unsigned int nKernelHeight,
		unsigned int nOutputFeatureMaps,
		float *parameters
		);

	virtual ~ConvolutionalLayer(){};

	void ForwardPropagate(unsigned int nThreadIndex);
	void Save(ofstream &file);
	void Restore (ifstream &file);

	// For each Input Feature Map, there is a kernel + bias.
	// For each Output Feature Map, there is a set of nInputFeatureMaps kernels + biases.

	vector<vector<vector<real, CUSTOM_ALLOCATOR(real)> > > m_Kernels;

	// Flattened kernels in the form of OFM x IFM x KH x KW
	vector<real, CUSTOM_ALLOCATOR(real)> m_WeightsMatrix;
	// kernels[i] is a vector of nInputFeatureMap elements, each element j of which is a matrix to convolve Input Feature Map j with
	// We lay out the biases separately to prepare for vectorized addition.
	vector<real, CUSTOM_ALLOCATOR(real)> m_Biases;


	// internal state
	unsigned int m_nInputFeatureMaps;
	unsigned int m_nInputFeatureMapWidth;
	unsigned int m_nInputFeatureMapHeight;
	unsigned int m_nStride;
	unsigned int m_nKernelWidth;
	unsigned int m_nKernelHeight;
	unsigned int m_nOutputFeatureMaps;
	unsigned int m_nOutputFeatureMapWidth;
	unsigned int m_nOutputFeatureMapHeight;

private:


	// Internal pointers to the input feature maps

	vector<real *> m_InputFeatureMaps[MAX_NUM_THREADS * BATCH_SIZE];
	vector<real *> m_OutputFeatureMaps[MAX_NUM_THREADS * BATCH_SIZE];



	void ExtractWeightsAndBiases(float *params);
	void InitializeWeightsAndBiases();

	void Convolve(real *inputFeatureMap, real *outputFeatureMap, real *kernel);
	void NaiveConvolve(real *inputFeatureMap, real *outputFeatureMap, real *kernel);
};





// Building networks
bool BuildMNISTNetwork(NeuralNetwork<NNLayerBase> &nn, char *path);
bool BuildSVHNNetwork(NeuralNetwork<NNLayerBase> &nn, char *path);

#endif // !defined(AFX_NEURALNETWORK_H__186C10A1_9662_4C1C_B5CB_21F2F361D268__INCLUDED_)

