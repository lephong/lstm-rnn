/*
 * AbstractNN.cpp
 *
 *  Created on: Jan 22, 2015
 *      Author: phong
 */

#include "AbstractNN.h"
#include <iostream>
#include <cmath>
#include "Utils.h"
using namespace std;

AbstractNN::AbstractNN(int composeType, int funcType) {
	vocaDic = NULL;
	params = NULL;
	dim = 0;

	this->composeType = composeType;
	this->funcType = funcType;
}

AbstractNN::~AbstractNN() {
	delete vocaDic;
	delete params;
}

Matrix* AbstractNN::func(Matrix* X, int type) {
	Matrix* F = new Matrix(X->rows, X->cols);
	if (type == -1) type = funcType;

	switch (type) {
	case FUNC_TANH:
		for (int i = 0; i < F->length; i++)
			F->data[i] = tanh(X->data[i]);
		break;
	case FUNC_SIGMOID:
		for (int i = 0; i < F->length; i++)
			F->data[i] = 1 / (1 + exp(-X->data[i]));
		break;
	case FUNC_SOFTSIGN:
		for (int i = 0; i < F->length; i++)
			F->data[i] = X->data[i] / (1 + abs(X->data[i]));
		break;
	case FUNC_RLU:
		for (int i = 0; i < F->length; i++) 
			F->data[i] = max((real)0, X->data[i]);
		break;
	default:
		Utils::error("undefined");
	}

	return F;
}

Matrix* AbstractNN::funcPrime(Matrix* F, int type) {
	Matrix* Fp = new Matrix(F->rows, F->cols);
	if (type == -1) type = funcType;

	switch (type) {
	case FUNC_TANH:
		for (int i = 0; i < F->length; i++)
			Fp->data[i] = 1 - F->data[i] * F->data[i];
		break;
	case FUNC_SIGMOID:
		for (int i = 0; i < F->length; i++)
			Fp->data[i] = F->data[i] * (1 - F->data[i]);
		break;
	case FUNC_SOFTSIGN:
		for (int i = 0; i < F->length; i++) {
			Fp->data[i] = 1 - abs(F->data[i]);
			Fp->data[i] = Fp->data[i] * Fp->data[i];
		}
		break;
	case FUNC_RLU:
		for (int i = 0; i < F->length; i++)
			Fp->data[i] = F->data[i] == 0 ? 0 : 1;
		break;
	default:
		Utils::error("undefined");
	}

	return Fp;
}

void AbstractNN::checkGradient(Treebank* tb) {
	real epsilon = 1e-4;
    AbstractParam** grad = new AbstractParam*[nThreads];
    for (int i = 0; i < nThreads; i++)
        grad[i] = createGrad();
	void** output = computeCostAndGrad(tb, 0, tb->size()-1, grad);
	delete[] output;

	vector<real> gradV;
	vector<real> numGradV;
    AbstractParam** otherGrad = new AbstractParam*[nThreads];
    for (int i = 0; i < nThreads; i++)
        otherGrad[i] = createGrad();

	for (int i = 0; i < params->weights.size(); i++) {
		Matrix* W = params->weights[i];
		for (int r = 0; r < W->rows; r++)
			for (int c = 0; c < W->cols; c++) {
				gradV.push_back(grad[0]->weights[i]->get(r,c));

				// plus
				W->put(r, c, W->get(r,c)+epsilon);
				output = computeCostAndGrad(tb, 0, tb->size()-1, otherGrad);
				real pluscost = *(real*)output[0];
				delete[] output;

				// minus
				W->put(r, c, W->get(r,c)-2*epsilon);
				output = computeCostAndGrad(tb, 0, tb->size()-1, otherGrad);
				real minuscost = *(real*)output[0];
				delete[] output;

				W->put(r, c, W->get(r,c)+epsilon);

				numGradV.push_back((pluscost - minuscost) / (2*epsilon));
				real diff = abs(numGradV.back() - gradV.back());
				if (diff > 0 && abs(gradV.back()) < 1e-7)
					cout << diff << " " << i << " " << r << " " << c << " : " << numGradV.back() << " " << gradV.back() << endl;
				if (diff > 5e-9)
					cout << diff << " " << i << " " << r << " " << c << " : " << numGradV.back() << " " << gradV.back() << endl;
			}
	}

	Matrix* X = new Matrix((int)gradV.size(), &gradV[0]);
	Matrix* Y = new Matrix((int)numGradV.size(), &numGradV[0]);
	real diff = Matrix::nrm2(X->add(-1, Y)) / Matrix::nrm2(X->add(Y));
	cout << diff << endl;
    cout << "should be < 1e-9" << endl;

    for (int i = 0; i < nThreads; i++) {
        delete grad[i];
        delete otherGrad[i];
    }
    delete[] grad;
    delete[] otherGrad;
    delete X;
    delete Y;
}


