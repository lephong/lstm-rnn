/*
 * Default.h
 *
 *  Created on: Dec 24, 2014
 *      Author: Phong
 */

#include <random>

#ifndef DEFAULT_H_
#define DEFAULT_H_

//#define __GRADIENT_CHECKING__

#ifdef __GRADIENT_CHECKING__
#define __DOUBLE__
class Matrix;
extern Matrix* innerDropout;
extern Matrix* leafDropout;
#endif

#include <string>
#include <unordered_set>
using namespace std;

#ifdef __DOUBLE__
	typedef double real;
#else
	typedef float real;
#endif

#define ONLY_ROOT false

// for standardization (Dictionary)
#define TEMPLATE_NONE 0
#define TEMPLATE_COLLOBERT 1
#define TEMPLATE_GLOVE 2
#define TEMPLATE_GLOVE_BIG 3

#define COMPOSE_NORMAL 0
#define COMPOSE_LSTM 1
#define COMPOSE_COMBINE 2

#define FUNC_SOFTSIGN 0
#define FUNC_SIGMOID 1
#define FUNC_TANH 2
#define FUNC_RLU 3

#define N_CLASS 5
//#define N_CLASS 2

extern int dim;

// for training
extern real lambda;
extern real lambdaL;
extern real lambdaC;
extern real lambdaR;
extern real dropoutRate;

extern real paramLearningRate;
extern real learningDecayRate;
extern real wembLearningRate;
extern real classLearningRate;
extern real repLearningRate;

extern int maxNEpoch;
extern int batchSize;
extern int evalDevStep;

extern int nThreads;
extern real normGradThresh;
extern int funcType;

// paths
extern string dataDir;
extern string dicDir;
extern string modelPath;

extern std::mt19937 randGen;

#endif /* DEFAULT_H_ */
