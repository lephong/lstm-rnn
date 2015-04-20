/*
 * AbstractNN.h
 *
 *  Created on: Jan 22, 2015
 *      Author: phong
 */

#ifndef ABSTRACTNN_H_
#define ABSTRACTNN_H_

#include <string>
#include "Dictionary.h"
#include "Matrix.h"
#include "AbstractParam.h"
#include "SenBinTree.h"
#include "Treebank.h"
#include <vector>

using namespace std;

class AbstractNN {
public:
	Dictionary *vocaDic;
	int dim;
	AbstractParam *params;

	int funcType, composeType;

	AbstractNN(int composeType = COMPOSE_NORMAL, int funcType = FUNC_TANH);
	virtual ~AbstractNN();

	virtual void save(string fname) = 0;

	virtual AbstractParam* createGrad() = 0;

	Matrix* func(Matrix* X, int type = -1);

	Matrix* funcPrime(Matrix* F, int type = -1);

	virtual void** forward(SenBinTree* tree, bool predict = false) = 0;

	virtual void backprop(SenBinTree* tree, AbstractParam* grad, bool predict = false) = 0;

	virtual void predict(SenBinTree* tree) = 0;

    virtual void** computeCostAndGrad(Treebank* tb, int startId, int endId, AbstractParam** grad) = 0;

    virtual void predict(vector<SenBinTree*>& trees) = 0;

	// make sure gradients are computed correctly
	void checkGradient(Treebank* tb);
};

#endif /* ABSTRACTNN_H_ */
