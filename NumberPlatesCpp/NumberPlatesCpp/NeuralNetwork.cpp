// NeuralNetwork.cpp: implementation of the NeuralNetwork class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "GeneralDefinitions.h"
#include "Utils.h"
#include "NeuralNetwork.h"
#include <malloc.h>  // for the _alloca function
#include <assert.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <algorithm>
using namespace std;

#define COUNT_NUM_CALCULATIONS 0

FullyConnectedNNLayer::	FullyConnectedNNLayer(		
		NNLayerBase *prevLayer, 
		SigmoidsBase *pSigmoid,
		unsigned int nOutputNeurons,
		float *parameters,
		unsigned int pctOfWeightsToKeep /*= 100 */) : NNLayerBase("FullyConnectedLayer",prevLayer,pSigmoid)
{
	m_Biases.resize(nOutputNeurons);
	m_VectorWeights.resize(nOutputNeurons);

	unsigned int nInputNeurons = prevLayer->m_NeuronOutputs[0].size();

	for (int itr = 0; itr < BATCH_SIZE*MAX_NUM_THREADS; itr++)
	{
		m_NeuronOutputs[itr].resize(nOutputNeurons);
	}


	real *paramsPtr = parameters;

	// If a pointer to trainable params is valid, go and load them
	for (unsigned int itr = 0; itr < nOutputNeurons; itr++)
	{
		vector<real, CUSTOM_ALLOCATOR(real)> &weightsForCurrentOutputNeuron = m_VectorWeights[itr];
		weightsForCurrentOutputNeuron.resize(nInputNeurons);
		for (unsigned int jtr = 0; jtr < nInputNeurons; jtr++)
		{ 
			// We can do memcpy but then it won't work if the real is double. 
			weightsForCurrentOutputNeuron[jtr] = *paramsPtr++;
		}
	}
	// Initialize biases
	for (unsigned int itr = 0; itr < nOutputNeurons; itr++)
	{
		// We can do memcpy but then it won't work if the real is double. 
		m_Biases[itr] = *paramsPtr++;
	}

	VectorOfVectorsToMatrix(m_VectorWeights,m_WeightsMatrix);

}

void FullyConnectedNNLayer::ForwardPropagate(unsigned int nThreadIndex)
{

  //m_NeuronOutputs[i] = SIGMOID(biases[i]+dotproduct(m_NeuronOutputs_from_prev_layer, vectorWeights[i]));
	for (unsigned int itr = 0; itr < m_NeuronOutputs[0].size(); itr++)
	{
		for (unsigned int jtr = nThreadIndex*BATCH_SIZE; jtr < (nThreadIndex+1)*BATCH_SIZE; jtr++)
		{
			real dSum = dotproduct_autovectorized(&m_pPrevLayer->m_NeuronOutputs[jtr][0],&m_WeightsMatrix[itr*m_pPrevLayer->m_NeuronOutputs[0].size()],m_pPrevLayer->m_NeuronOutputs[0].size());

			dSum+=m_Biases[itr];

			m_NeuronOutputs[jtr][itr] = dSum;
		}
	}

	// Sigmoid
	if (m_pSigmoid)
	{
		for (unsigned int jtr = nThreadIndex*BATCH_SIZE; jtr < (nThreadIndex+1)*BATCH_SIZE; jtr++)
		{
			(*m_pSigmoid)(m_NeuronOutputs[jtr]);
		}
	}

}


void FullyConnectedNNLayer::Save(ofstream &file)
{
	// TODO
}

void FullyConnectedNNLayer::Restore (ifstream &file)
{
	// TODO
}




MaxPoolingLayer::MaxPoolingLayer(
	NNLayerBase *prevLayer, 
	SigmoidsBase *pSigmoid,
	unsigned int nInputFeatureMaps,
	unsigned int nInputFeatureMapWidth,
	unsigned int nInputFeatureMapHeight,
	unsigned int nStride
	) :	NNLayerBase( "MaxPooling", prevLayer, pSigmoid),
	m_nInputFeatureMaps(nInputFeatureMaps),
	m_nInputFeatureMapWidth(nInputFeatureMapWidth),
	m_nInputFeatureMapHeight(nInputFeatureMapHeight),
	m_nStride(nStride) 
{
	// Only supporting full strides
	ASSERT(m_nInputFeatureMapWidth % m_nStride == 0);
	ASSERT(m_nInputFeatureMapHeight % m_nStride == 0);

	m_nOutputFeatureMapWidth = m_nInputFeatureMapWidth / m_nStride;
	m_nOutputFeatureMapHeight = m_nInputFeatureMapHeight / m_nStride;

	for (int itr = 0; itr < BATCH_SIZE*MAX_NUM_THREADS; itr++)
	{
		m_NeuronOutputs[itr].reserve(m_nInputFeatureMaps*m_nOutputFeatureMapWidth*m_nOutputFeatureMapHeight + RESERVED_SPACE);
		m_NeuronOutputs[itr].resize(m_nInputFeatureMaps*m_nOutputFeatureMapWidth*m_nOutputFeatureMapHeight); // for efficient pooling with stride to work correctly
	}
}

void MaxPoolingLayer::ForwardPropagate(unsigned int nThreadIndex)
{
	for (unsigned int jtr = nThreadIndex*BATCH_SIZE; jtr < (nThreadIndex+1)*BATCH_SIZE; jtr++)
	{
		MaxPooling(
			&m_pPrevLayer->m_NeuronOutputs[jtr][0],
			m_nInputFeatureMaps,
			m_nInputFeatureMapWidth,
			m_nInputFeatureMapHeight,
			&m_NeuronOutputs[jtr][0],
			m_nStride,
			m_nStride,
			m_nStride);
	}
}


ConvolutionalLayer::ConvolutionalLayer(
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
	):
	m_nInputFeatureMaps(nInputFeatureMaps),
	m_nInputFeatureMapWidth(nInputFeatureMapWidth),
	m_nInputFeatureMapHeight(nInputFeatureMapHeight),
	m_nStride(nStride),
	m_nOutputFeatureMaps(nOutputFeatureMaps),
	m_nKernelWidth(nKernelWidth),
	m_nKernelHeight(nKernelHeight)
{

	m_pSigmoid = pSigmoid;
	m_pPrevLayer = prevLayer;

	m_nOutputFeatureMapWidth = (m_nInputFeatureMapWidth - m_nKernelWidth) / m_nStride + 1;
	m_nOutputFeatureMapHeight = (m_nInputFeatureMapHeight - m_nKernelHeight ) / m_nStride + 1;

	unsigned int numOutputNeurons = m_nOutputFeatureMaps*m_nOutputFeatureMapWidth*m_nOutputFeatureMapHeight;
	unsigned int numInputNeurons = m_nInputFeatureMaps * m_nInputFeatureMapHeight * m_nInputFeatureMapWidth;

	for(int itr = 0; itr < BATCH_SIZE * MAX_NUM_THREADS; itr++)
	{
		m_NeuronOutputs[itr].reserve(numOutputNeurons + RESERVED_SPACE);
		m_NeuronOutputs[itr].resize(numOutputNeurons); 
		
	}

	m_Biases.resize(m_nOutputFeatureMaps);
	m_WeightsMatrix.resize(m_nOutputFeatureMaps*m_nInputFeatureMaps*m_nKernelWidth*m_nKernelHeight);

	// Create a vector of pointers to the input feature maps

	ASSERT(m_nInputFeatureMaps == m_pPrevLayer->m_NeuronOutputs[0].size()/(m_nInputFeatureMapWidth*nInputFeatureMapHeight));
	for (unsigned int itr = 0; itr < m_pPrevLayer->m_NeuronOutputs[0].size(); itr += m_nInputFeatureMapWidth*nInputFeatureMapHeight)
	{
		for (unsigned int jtr = 0; jtr < BATCH_SIZE * MAX_NUM_THREADS; jtr++)
		{
			m_InputFeatureMaps[jtr].push_back(&m_pPrevLayer->m_NeuronOutputs[jtr][itr]);
		}
	}

	// Create a vector of pointers to the output feature maps

	ASSERT(m_nOutputFeatureMaps == m_NeuronOutputs[0].size() / ( m_nOutputFeatureMapWidth * m_nOutputFeatureMapHeight));
	for (unsigned int itr = 0; itr < m_NeuronOutputs[0].size(); itr += m_nOutputFeatureMapWidth*m_nOutputFeatureMapHeight)
	{
		for (unsigned int jtr = 0; jtr < BATCH_SIZE * MAX_NUM_THREADS; jtr++)
		{
			m_OutputFeatureMaps[jtr].push_back(&m_NeuronOutputs[jtr][itr]);
		}
	}

	// Initialize weights and bias here 
	if (parameters)
	{
		ExtractWeightsAndBiases(parameters);
	}

}


void ConvolutionalLayer::ExtractWeightsAndBiases(float *params)
{

	// Flattened kernels
	
	m_WeightsMatrix.resize(m_nOutputFeatureMaps*m_nInputFeatureMaps*m_nKernelWidth*m_nKernelHeight);
	memcpy(&m_WeightsMatrix[0],params,sizeof(m_WeightsMatrix[0])*m_WeightsMatrix.size());

	m_Biases.resize(m_nOutputFeatureMaps);
	
	real *pWeights = params + m_WeightsMatrix.size();

	memcpy(&m_Biases[0], (params + m_WeightsMatrix.size()), m_nOutputFeatureMaps * sizeof(m_Biases[0]));
}


inline void ConvolutionalLayer::NaiveConvolve(real *inputFeatureMap, real *outputFeatureMap, real *kernel)
{
	unsigned int count = 0;

	for (unsigned int hItr = 0; hItr < (m_nInputFeatureMapHeight - m_nKernelHeight) + 1; hItr+=m_nStride)
	{
		for (unsigned int wItr = 0; wItr < (m_nInputFeatureMapWidth - m_nKernelWidth) +1 ; wItr+=m_nStride)
		{
			real sum = (real)0.0f;
			// Dot product here
			for (unsigned int kH = 0; kH < m_nKernelHeight; kH++)
			{
				for (unsigned int kW = 0; kW < m_nKernelWidth; kW++)
				{
					// Those castings are a must for Intel compiler vectorizer. 
					// I could't figure out why, but apparently, without these, the vectorized code would convert everything to double,
					// do the calculations and then convert back to single precision, which is slow as hell.
					sum += (real)((real)kernel[m_nKernelHeight*kH+kW]*(real)inputFeatureMap[(m_nInputFeatureMapHeight*hItr+wItr) + kH*m_nInputFeatureMapWidth+ kW]);
					//sum += (kernel[m_nKernelHeight*kH+kW]*inputFeatureMap[(m_nInputFeatureMapHeight*hItr+wItr) + kH*m_nInputFeatureMapWidth+ kW]);
				}
			}
			outputFeatureMap[count++]+=sum;
		}
	}
}

inline void ConvolutionalLayer::Convolve(real *inputFeatureMap, real *outputFeatureMap, real *kernel)
{
	// The condition below may be replaced with checking the condition once, and setting a function pointer to be called.
	// Empiric check shows almost no impact to performance, leavig for now.
	if ( (m_nKernelWidth == 5) && (m_nKernelHeight == 5) && (m_nStride == 2) && (m_nInputFeatureMapWidth >= 5) && (m_nInputFeatureMapHeight >= 5) )
	{
		ConvolveImageImpl<5,5,2>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);
	}
	else if ( (m_nKernelWidth == 5) && (m_nKernelHeight == 5) && (m_nStride == 1) && (m_nInputFeatureMapWidth >= 5) && (m_nInputFeatureMapHeight >= 5) )
	{
		ConvolveImageImpl<5,5,1>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);

	}
	else if((m_nKernelWidth == 16) && (m_nKernelHeight == 16) && (m_nStride == 1))
	{
		ConvolveImageImpl<16,16,1>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);
	}
	else if((m_nKernelWidth == 8) && (m_nKernelHeight == 8) && (m_nStride == 1))
	{
		ConvolveImageImpl<8,8,1>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);
	}
	else if((m_nKernelWidth == 6) && (m_nKernelHeight == 6) && (m_nStride == 1))
	{
		ConvolveImageImpl<6,6,1>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);
	}
	else if((m_nKernelWidth == 3) && (m_nKernelHeight == 3) && (m_nStride == 1))
	{
		ConvolveImageImpl<3,3,1>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);
	}
	else if((m_nKernelWidth == 11) && (m_nKernelHeight == 11) && (m_nStride == 1))
	{
		ConvolveImageImpl<11,11,1>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);
	}
	else if((m_nKernelWidth == 11) && (m_nKernelHeight == 11) && (m_nStride == 4))
	{
		ConvolveImageImpl<11,11,4>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);
	}
    else if((m_nKernelWidth == 7) && (m_nKernelHeight == 7) && (m_nStride == 1))
	{
		ConvolveImageImpl<7,7,1>(inputFeatureMap,outputFeatureMap,kernel,m_nInputFeatureMapWidth,m_nInputFeatureMapHeight);
	}
	else
	{
		NaiveConvolve(inputFeatureMap, outputFeatureMap, kernel);
	}
}



void ConvolutionalLayer::ForwardPropagate(unsigned int nThreadIndex)
{
	for (unsigned int itr = nThreadIndex*BATCH_SIZE; itr < (nThreadIndex+1)*BATCH_SIZE; itr++)
	{
		// Add convolutions
		for (unsigned int outputFeatureMapsItr = 0; outputFeatureMapsItr < m_nOutputFeatureMaps; outputFeatureMapsItr++)
		{
			real *outputFeatureMap = m_OutputFeatureMaps[itr][outputFeatureMapsItr];
	
			// Init output feature map with bias.
			// This is far from being ideal from the performance standpoint, as it causes extra BW, but should be OK for correctness.
			//ippsSet_32f(m_Biases[outputFeatureMapsItr],outputFeatureMap,m_nOutputFeatureMapHeight * m_nOutputFeatureMapWidth);
			for (unsigned int sItr = 0; sItr < m_nOutputFeatureMapHeight * m_nOutputFeatureMapWidth; sItr++)
			{
				outputFeatureMap[sItr] = m_Biases[outputFeatureMapsItr];
			}

			for (unsigned int inputFeatureMapsItr = 0; inputFeatureMapsItr < m_nInputFeatureMaps; inputFeatureMapsItr++)
			{
				real *inputFeatureMap = m_InputFeatureMaps[itr][inputFeatureMapsItr];

				// Convolve each input feature map with corresponding kernel, accummulate the result
				// m_FlattenedKernel[OFM][IFM][KH][KW]
				real *kernel = &m_WeightsMatrix[outputFeatureMapsItr * (m_nInputFeatureMaps * m_nKernelHeight * m_nKernelWidth) + 
												   inputFeatureMapsItr * (m_nKernelHeight * m_nKernelWidth)];

				Convolve(inputFeatureMap,outputFeatureMap,&kernel[0]);
			}
		}

		if (m_pSigmoid)
		{
			(*m_pSigmoid)(m_NeuronOutputs[itr]);
		}
	}
}

void ConvolutionalLayer::Save(ofstream &file)
{
	// TODO
}

void ConvolutionalLayer::Restore (ifstream &file)
{
	// TODO
}






bool BuildSVHNNetwork(NeuralNetwork<NNLayerBase> &nn, char *path)
{

	string strPath = path;
	
	/*
	string fileName = strPath + "weights";
	WriteRawFileFloat(fileName);
	*/

	// Get weights
	
	string parametersFileName = strPath + "weights.raw";

	int fileSize = GetFileSize(parametersFileName.c_str());

	unsigned int nParameters = fileSize / (sizeof(float));

	if (nParameters != 1703370)
	{
		cout << "Couldn't find weights file" << endl;
		return false;
	}

	HANDLE hParametersFile = NULL;
	char *pParameters = NULL;
	if (!MapFile(parametersFileName.c_str(),&hParametersFile,&pParameters))
	{
		return false;
	}

	float *pFloatParams = (float *)pParameters;

	// Get rid of all the existing contents
	nn.Initialize();

	// Start building the network

	// (0) Input - 3x32x32
	// (1) Convolution (5x5, stride=1) - 32x28x28
	//		(1a) Threshold (t=1e-6,val=0)
	// (2) MaxPooling (2x2) - 32x14x14
	// (3) Convolution (5x5, stride=1) - 64x10x10
	//		(3a) Threshold (t=1e-6,val=0)
	// (4) MaxPooling (2x2) - 64x5x5
	// (4') Reshape - 1600   - dropped, no meaning
	// (5) FullyConnected - 1024
	//		(5a) Threshold (t=1e-6,val=0)
	// (6) FullyConnected - 10
	// (--) LogSoftMax - 10  - dropped, does not change the result
	
	NNInputLayer *pInputLayer = new NNInputLayer();
	for (int itr = 0; itr < BATCH_SIZE*MAX_NUM_THREADS; itr++)
	{
		pInputLayer->m_NeuronOutputs[itr].resize(3*32*32);
	}
	nn.m_Layers.push_back(pInputLayer);

	ConvolutionalLayer *pConvLayer1 = new ConvolutionalLayer(
		pInputLayer,
		new ThresholdSigmoid(),
		3,
		32,
		32,
		1,
		5,
		5,
		32,
		pFloatParams);


	nn.m_Layers.push_back(pConvLayer1);

	MaxPoolingLayer *pMaxPoolingLayer1 = new MaxPoolingLayer(
		pConvLayer1,
		NULL,
		32,
		28,
		28,
		2);
	nn.m_Layers.push_back(pMaxPoolingLayer1);


	ConvolutionalLayer *pConvLayer2 = new ConvolutionalLayer(
		pMaxPoolingLayer1,
		new ThresholdSigmoid(),
		32,
		14,
		14,
		1,
		5,
		5,
		64,
		pFloatParams+2432);

	nn.m_Layers.push_back(pConvLayer2);
	MaxPoolingLayer *pMaxPoolingLayer2 = new MaxPoolingLayer(
		pConvLayer2,
		NULL,
		64,
		10,
		10,
		2);
	nn.m_Layers.push_back(pMaxPoolingLayer2);

	// TODO: instead of specifying constant values, implement a meothod in NeuralNetwork and in NNLayerBase derivatives to return the count of parameters
	FullyConnectedNNLayer *pFullyConnectedLayer1 = new FullyConnectedNNLayer(
		pMaxPoolingLayer2,
		new ThresholdSigmoid(),
		1024,
		pFloatParams+2432+51264);
	nn.m_Layers.push_back(pFullyConnectedLayer1);

	FullyConnectedNNLayer *pFullyConnectedLayer2 = new FullyConnectedNNLayer(
		pFullyConnectedLayer1,
		new EmptySigmoid(),
		10,
		pFloatParams+2432+51264+1639424);
	nn.m_Layers.push_back(pFullyConnectedLayer2);


	CLOSE_WIN_HANDLE(hParametersFile);

	return true;
}


bool BuildMNISTNetwork(NeuralNetwork<NNLayerBase> &nn, char *path)
{
	string strPath = path;
	
	// Get weights
	
	string parametersFileName = strPath + "weights.raw";

	int fileSize = GetFileSize(parametersFileName.c_str());

	unsigned int nParameters = fileSize / (sizeof(float));

	ASSERT(nParameters == 133816);

	HANDLE hParametersFile = NULL;
	char *pParameters = NULL;
	if (!MapFile(parametersFileName.c_str(),&hParametersFile,&pParameters))
	{
		return false;
	}

	float *pFloatParams = (float *)pParameters;

	// Get rid of all the existing contents
	nn.Initialize();

	// Start building the network
	NNInputLayer *pInputLayer = new NNInputLayer();
	for (int itr = 0; itr < BATCH_SIZE*MAX_NUM_THREADS; itr++)
	{
		pInputLayer->m_NeuronOutputs[itr].resize(29*29);
	}
	nn.m_Layers.push_back(pInputLayer);

	ConvolutionalLayer *pConvLayer1 = new ConvolutionalLayer(
		pInputLayer,
		new TanHSigmoid(),
		1,
		29,
		29,
		2,
		5,
		5,
		6,
		pFloatParams);

	nn.m_Layers.push_back(pConvLayer1);


	ConvolutionalLayer *pConvLayer2 = new ConvolutionalLayer(
		pConvLayer1,
		new TanHSigmoid(),
		6,
		13,
		13,
		2,
		5,
		5,
		50,
		pFloatParams+156);
	nn.m_Layers.push_back(pConvLayer2);

	FullyConnectedNNLayer *pFullyConnectedLayer1 = new FullyConnectedNNLayer(
		pConvLayer2,
		new TanHSigmoid(),
		100,
		pFloatParams+156+7550);
	nn.m_Layers.push_back(pFullyConnectedLayer1);


	FullyConnectedNNLayer *pFullyConnectedLayer2 = new FullyConnectedNNLayer(
		pFullyConnectedLayer1,
		new TanHSigmoid(),
		10,
		pFloatParams+156+7550+125100);
	nn.m_Layers.push_back(pFullyConnectedLayer2);

	CLOSE_WIN_HANDLE(hParametersFile);

	return true;

}



void NNInputLayer::ForwardPropagate(unsigned int nThreadIndex) 
{
}

