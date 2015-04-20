//
//  IORNNTrainer.cpp
//  iornn-parser-c
//
//  Created by Phong Le on 26/12/14.
//  Copyright (c) 2014 Phong Le. All rights reserved.
//

#include "RNNTrainer.h"
#include "RNN.h"
#include "RNNParam.h"

void RNNTrainer::adagrad(AbstractNN* absNN, AbstractParam* absGrad, unordered_set<int>& tWords) {
	RNN* net = (RNN*)absNN;
	RNNParam* grad = (RNNParam*)absGrad;

	if (paramVarience == NULL) {
		paramVarience = (RNNParam*) net->createGrad();
		nEvals.clear();
		for (int i = 0; i < paramVarience->weights.size(); i++)
			nEvals.push_back(0);
	}

	real eps = 1e-4;

	// for weights # Wword, L
	for (int i = 0; i < grad->nMatricesWoWE; i++) {
		Matrix* W = grad->weights[i];
		Matrix* temp = W->mul(W);
		((RNNParam*)paramVarience)->weights[i]->addi(temp); delete temp;

		nEvals[i]++;
		real rate = paramLearningRate / (1 + nEvals[i] * learningDecayRate);
		if (W == grad->Wc || W == grad->Wwc || W == grad->lstm.Wc || W == grad->bc || W == grad->bwc)
			rate = classLearningRate / (1 + nEvals[i] * learningDecayRate);

		if (nEvals[i] < 2) {
			net->params->weights[i]->addi(-rate, grad->weights[i]);
		}
		else {
			Matrix* std = ((RNNParam*)paramVarience)->weights[i]->sqrt()->addi(eps);
			net->params->weights[i]->addi(grad->weights[i]->divi(std)->muli(-rate));
			delete std;
		}
	}

	// for L
	for (unordered_set<int>::iterator it = tWords.begin(); it != tWords.end(); it++) {
		int word = *it;
		Matrix* temp = grad->L[word]->mul(grad->L[word]);
		((RNNParam*)paramVarience)->L[word]->addi(temp); delete temp;

		int i = grad->nMatricesWoWE + word;
		nEvals[i]++;
		real rate = wembLearningRate / (1 + nEvals[i] * learningDecayRate);

		if (nEvals[i] < 2) {
			((RNNParam*)net->params)->L[word]->addi(-rate, grad->L[word]);
		}
		else {
			Matrix* std = ((RNNParam*)paramVarience)->L[word]->sqrt()->addi(eps);
			((RNNParam*)net->params)->L[word]->addi(grad->L[word]->divi(std)->muli(-rate));
			delete std;
		}
	}
}

void RNNTrainer::adadelta(AbstractNN* absNN, AbstractParam* absGrad, unordered_set<int>& tWords) {
	RNN* net = (RNN*)absNN;
	RNNParam* grad = (RNNParam*)absGrad;

	if (paramVarience == NULL)
		paramVarience = (RNNParam*) net->createGrad();
	if (updateVarience == NULL)
		updateVarience = (RNNParam*) net->createGrad();

	real eps = 1e-6;

	// for weights # Wword, L
	for (int i = 0; i < grad->nMatricesWoWE; i++) {
		Matrix* W = grad->weights[i];
		Matrix* temp = W->mul(W);
		((RNNParam*)paramVarience)->weights[i]->muli(learningDecayRate)->addi(1-learningDecayRate, temp); delete temp;

		Matrix* paramStd = ((RNNParam*)paramVarience)->weights[i]->addi(eps)->sqrt();
		Matrix* updateStd = ((RNNParam*)updateVarience)->weights[i]->addi(eps)->sqrt();
		Matrix* delta = grad->weights[i]->mul(updateStd)->divi(paramStd)->muli(-1);

		temp = delta->mul(delta);
		((RNNParam*)updateVarience)->weights[i]->muli(learningDecayRate)->addi(1-learningDecayRate, temp); delete temp;

		net->params->weights[i]->addi(delta);
		delete paramStd;
		delete updateStd;
		delete delta;
	}

	// for L
	for (unordered_set<int>::iterator it = tWords.begin(); it != tWords.end(); it++) {
		int word = *it;
		Matrix* temp = grad->L[word]->mul(grad->L[word]);
		((RNNParam*)paramVarience)->L[word]->muli(learningDecayRate)->addi(1-learningDecayRate, temp); delete temp;

		Matrix* paramStd = ((RNNParam*)paramVarience)->L[word]->addi(eps)->sqrt();
		Matrix* updateStd = ((RNNParam*)updateVarience)->L[word]->addi(eps)->sqrt();
		Matrix* delta = grad->L[word]->mul(updateStd)->divi(paramStd)->muli(-1);

		temp = delta->mul(delta);
		((RNNParam*)updateVarience)->L[word]->muli(learningDecayRate)->addi(1-learningDecayRate, temp); delete temp;

		((RNNParam*)net->params)->L[word]->addi(delta);
		delete paramStd;
		delete updateStd;
		delete delta;
	}
}

void RNNTrainer::sgd(AbstractNN* absNN, AbstractParam* absGrad, unordered_set<int>& tWords) {
	RNN* net = (RNN*)absNN;
	RNNParam* grad = (RNNParam*)absGrad;

	// for weights # Wword, L
	for (int i = 0; i < grad->nMatricesWoWE; i++) {
		net->params->weights[i]->addi(-paramLearningRate, grad->weights[i]);
	}

	// for L
	for (unordered_set<int>::iterator it = tWords.begin(); it != tWords.end(); it++) {
		int word = *it;
		((RNNParam*)net->params)->L[word]->addi(-wembLearningRate, grad->L[word]);
	}
}

