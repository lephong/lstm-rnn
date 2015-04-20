/*
 * IORNN.cpp
 *
 *  Created on: Dec 17, 2014
 *      Author: Phong
 */

#include "Utils.h"
#include <fstream>
#include "RNN.h"

using namespace std;

RNN::RNN(int dim, Dictionary* vocaDic, vector<Matrix*>& L, int compType, int funcType) : AbstractNN(compType, funcType) {
	this->vocaDic = vocaDic;

	this->dim = dim;
	this->wdim = L.empty() ? dim : L[0]->length;
	this->composeType = compType;

	params = new RNNParam(dim, wdim, vocaDic->size(), N_CLASS);
	if (L.empty()) return;
	for (int i = 0; i < ((RNNParam*)params)->L.size(); i++)
		((RNNParam*)params)->L[i]->copy(L[i]);
}

void RNN::save(string fname) {
	cout << "save model to " << fname << endl;
	ofstream f(fname);

	f << dim << " " << wdim << endl;
	f << composeType << " " << funcType << endl;

	vocaDic->save(f);
	((RNNParam*)params)->save(f);

	f << endl;
	f.close();
}

RNN* RNN::load(string fname) {
	cout << "load model in " << fname << endl;

	ifstream f(fname);
	RNN* net = new RNN();

	f >> net->dim;
	f >> net->wdim;
	f >> net->composeType;
	f >> net->funcType;

	net->vocaDic = Dictionary::load(f);
	net->params = RNNParam::load(f);

	f.close();
	return net;
}

AbstractParam* RNN::createGrad() {
	RNNParam* p = new RNNParam(dim, wdim, vocaDic->size(), N_CLASS);
	p->fill(0);
	return p;
}


void** RNN::forward(SenBinTree* tree, bool predict) {
	real *cost = new real; *cost = 0;
	int *count = new int; *count = 0;

	if (tree->dropout != NULL) delete tree->dropout;
#ifdef __GRADIENT_CHECKING__
	tree->dropout = tree->isLeaf() ? leafDropout->dup() : innerDropout->dup();
#else
	tree->dropout = Matrix::bernoulli(tree->isLeaf() ? wdim : dim, 1-dropoutRate);
#endif

	if (predict) tree->dropout->fill(1-dropoutRate);

	for (int dir = 0; dir < 2; dir++) {
		SenBinTree* child = tree->children[dir];
		if (child != NULL) {
			void** ret = forward(child, predict);
			*cost += *(real*)ret[0];
			*count += *(int*)ret[1];
			delete (real*)ret[0]; delete (int*)ret[1]; delete[] ret;
		}
	}

	Matrix* score = Matrix::zeros(N_CLASS);

	if (tree->isLeaf()) {
		tree->rep = ((RNNParam*)params)->L[tree->word]->dup();

		Matrix *temp = tree->rep->mul(tree->dropout);
		Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->Wwc, temp, 1, score);
		score->addi(((RNNParam*)params)->bwc);
		delete temp;
	}
	else {
		if (composeType == COMPOSE_NORMAL || composeType == COMPOSE_COMBINE) {
			Matrix* input = Matrix::zeros(dim);
			for (int dir = 0; dir < 2; dir++) {
				SenBinTree* child = tree->children[dir];
				if (child->isLeaf())
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->Ww[dir], child->rep, 1, input);
				else
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->W[dir], child->rep, 1, input);
			}

			input->addi(((RNNParam*)params)->b);
			tree->rep = func(input);
			delete input;
		}

		// for LSTM
		if (composeType == COMPOSE_LSTM || composeType == COMPOSE_COMBINE) {
			Matrix *inputIg[2] = { ((RNNParam*)params)->lstm.igb->dup(), ((RNNParam*)params)->lstm.igb->dup() };
			Matrix *inputOg = ((RNNParam*)params)->lstm.ogb->dup();
			Matrix *inputFg[2] = { ((RNNParam*)params)->lstm.fgb->dup(), ((RNNParam*)params)->lstm.fgb->dup() };

			for (int dir = 0; dir < 2; dir++) {
				SenBinTree* child = tree->children[dir];
				if (child->isLeaf()) {
					Matrix *rep = child->rep;
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.igWw, rep, 1, inputIg[dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.igWwSis, rep, 1, inputIg[1-dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.ogWw[dir], rep, 1, inputOg);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.fgWw, rep, 1, inputFg[dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.fgWwSis, rep, 1, inputFg[1-dir]);
				}
				else {
					Matrix *rep = child->lstm.rep;
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.igW, rep, 1, inputIg[dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.igWSis, rep, 1, inputIg[1-dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.igWc, child->lstm.cRep, 1, inputIg[dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.igWcSis, child->lstm.cRep, 1, inputIg[1-dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.ogW[dir], rep, 1, inputOg);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.fgW, rep, 1, inputFg[dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.fgWSis, rep, 1, inputFg[1-dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.fgWc, child->lstm.cRep, 1, inputFg[dir]);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.fgWcSis, child->lstm.cRep, 1, inputFg[1-dir]);
				}
			}

			Matrix *input = ((RNNParam*)params)->lstm.b->dup();
			tree->lstm.cRep = Matrix::zeros(dim);
			for (int dir = 0; dir < 2; dir++) {
				tree->lstm.igRep[dir] = func(inputIg[dir], FUNC_SIGMOID);
				tree->lstm.fgRep[dir] = func(inputFg[dir], FUNC_SIGMOID);

				if (tree->children[dir]->isLeaf()) {
					Matrix *temp = Matrix::zeros(dim);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.Ww[dir], tree->children[dir]->rep, 1, temp);
					temp->muli(tree->lstm.igRep[dir]);
					input->addi(temp); delete temp;
				}
				else {
					Matrix *temp = Matrix::zeros(dim);
					Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.W[dir], tree->children[dir]->lstm.rep, 1, temp);
					temp->muli(tree->lstm.igRep[dir]);
					input->addi(temp); delete temp;

					temp = tree->lstm.fgRep[dir]->mul(tree->children[dir]->lstm.cRep);
					tree->lstm.cRep->addi(temp); delete temp;
				}
			}

			tree->lstm.pRep = func(input);
			tree->lstm.cRep->addi(tree->lstm.pRep);

			Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.ogWc, tree->lstm.cRep, 1, inputOg);
			tree->lstm.ogRep = func(inputOg, FUNC_SIGMOID);

			tree->lstm.rep = func(tree->lstm.cRep)->muli(tree->lstm.ogRep);

			delete input; delete inputOg; delete inputIg[0]; delete inputIg[1]; delete inputFg[0]; delete inputFg[1];
		}

		if (tree->lstm.rep != NULL) {
			Matrix *temp = tree->lstm.rep->mul(tree->dropout);
			Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.Wc, temp, 1, score);
			delete temp;
		}
		if (tree->rep != NULL) {
			Matrix *temp = tree->rep->mul(tree->dropout);
			Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->Wc, temp, 1, score);
			delete temp;
		}
		score->addi(((RNNParam*)params)->bc);
	}

	if (tree->score < N_CLASS && (predict || (tree->conflict == false && (!ONLY_ROOT || tree->isRoot())))) {
		tree->prob = new Matrix(N_CLASS);
		Utils::safelyComputeSoftmax(tree->prob->data, score->data, score->length);
		*cost += -log(tree->prob->data[tree->score]);
		(*count)++;

		if (predict) {
			int c = Utils::maxPos(tree->prob->data, N_CLASS);
			tree->predictedScore = c;

			if (N_CLASS > 2) {
				real negScore = tree->prob->data[0] + tree->prob->data[1];
				real posScore = tree->prob->data[3] + tree->prob->data[4];
				if (negScore > posScore) tree->binPredScore = -1;
				else tree->binPredScore = 1;
			}
		}
	}

	delete score;
	return new void*[2] { cost, count };
}

void RNN::backprop(SenBinTree* tree, AbstractParam* absGrad, bool predict) {
	RNNParam* grad = (RNNParam*)absGrad;
	SenBinTree* parent = tree->parent;

	// for leaf
	if (tree->isLeaf()) {
		tree->grad = Matrix::zeros(wdim);
		if (parent != NULL) {
			if (composeType == COMPOSE_NORMAL || composeType == COMPOSE_COMBINE) {
				for (int dir = 0; dir < 2; dir++) {
					if (parent->children[dir] != tree) continue;
					Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->Ww[dir], parent->grad, 1, tree->grad);
					Matrix::ger(1, parent->grad, tree->rep , grad->Ww[dir]);
				}
			}

			// LSTM
			if (composeType == COMPOSE_LSTM || composeType == COMPOSE_COMBINE) {
				for (int dir = 0; dir < 2; dir++) {
					if (parent->children[dir] != tree) continue;
					Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.ogWw[dir], parent->lstm.ogGrad, 1, tree->grad);
					Matrix::ger(1, parent->lstm.ogGrad, tree->rep, grad->lstm.ogWw[dir]);

					Matrix *temp = parent->lstm.pGrad->mul(parent->lstm.igRep[dir]);
					Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.Ww[dir], temp, 1, tree->grad);
					Matrix::ger(1, temp, tree->rep, grad->lstm.Ww[dir]); delete temp;

					Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.fgWw, parent->lstm.fgGrad[dir], 1, tree->grad);
					Matrix::ger(1, parent->lstm.fgGrad[dir], tree->rep, grad->lstm.fgWw);
					Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.fgWwSis, parent->lstm.fgGrad[1-dir], 1, tree->grad);
					Matrix::ger(1, parent->lstm.fgGrad[1-dir], tree->rep, grad->lstm.fgWwSis);

					Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.igWw, parent->lstm.igGrad[dir], 1, tree->grad);
					Matrix::ger(1, parent->lstm.igGrad[dir], tree->rep, grad->lstm.igWw);
					Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.igWwSis, parent->lstm.igGrad[1-dir], 1, tree->grad);
					Matrix::ger(1, parent->lstm.igGrad[1-dir], tree->rep, grad->lstm.igWwSis);
				}
			}
		}

		if (tree->score < N_CLASS && tree->conflict == false && (!ONLY_ROOT || tree->isRoot())) {
			Matrix* zProb = tree->prob->dup();
			zProb->data[tree->score]--;

			Matrix *temp = tree->rep->mul(tree->dropout);
			Matrix::ger(1, zProb, temp, grad->Wwc);
			delete temp;

			grad->bwc->addi(zProb);

			temp = new Matrix(wdim);
			Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->Wwc, zProb, 0, temp);
			temp->muli(tree->dropout);
			tree->grad->addi(temp);
			delete zProb;
			delete temp;
		}

		grad->L[tree->word]->addi(tree->grad);

		return;
	}

	// for internal node
	if (composeType == COMPOSE_NORMAL || composeType == COMPOSE_COMBINE)
		tree->grad = Matrix::zeros(dim);
	if (composeType == COMPOSE_LSTM || composeType == COMPOSE_COMBINE)
		tree->lstm.grad = Matrix::zeros(dim);

	if (tree->score < N_CLASS && tree->conflict == false && (!ONLY_ROOT || tree->isRoot())) {
		Matrix* zProb = tree->prob->dup();
		zProb->data[tree->score]--;

		if (composeType == COMPOSE_NORMAL || composeType == COMPOSE_COMBINE) {
			Matrix *temp = tree->rep->mul(tree->dropout);
			Matrix::ger(1, zProb, temp, grad->Wc);
			delete temp;

			temp = new Matrix(dim);
			Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->Wc, zProb, 0, temp);
			temp->muli(tree->dropout);
			tree->grad->addi(temp);
			delete temp;
		}
		if (composeType == COMPOSE_LSTM || composeType == COMPOSE_COMBINE) {
			Matrix *temp = tree->lstm.rep->mul(tree->dropout);
			Matrix::ger(1, zProb, temp, grad->lstm.Wc);
			delete temp;

			temp = new Matrix(dim);
			Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.Wc, zProb, 0, temp);
			temp->muli(tree->dropout);
			tree->lstm.grad->addi(temp);
			delete temp;
		}
		grad->bc->addi(zProb);
		delete zProb;
	}

	if (composeType == COMPOSE_NORMAL || composeType == COMPOSE_COMBINE) {
		if (parent != NULL) {
			for (int dir = 0; dir < 2; dir++) {
				if (parent->children[dir] != tree) continue;
				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->W[dir], parent->grad, 1, tree->grad);
				Matrix::ger(1, parent->grad, tree->rep , grad->W[dir]);
			}
		}


		Matrix* temp = funcPrime(tree->rep);
		tree->grad->muli(temp);
		delete temp;

		grad->b->addi(tree->grad);
	}

	// LSTM
	if (composeType == COMPOSE_LSTM || composeType == COMPOSE_COMBINE) {
		tree->lstm.cGrad = Matrix::zeros(dim);

		SenBinTree *parent = tree->parent;
		if (parent != NULL) {
			for (int dir = 0; dir < 2; dir++) {
				if (parent->children[dir] != tree) continue;
				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.ogW[dir], parent->lstm.ogGrad, 1, tree->lstm.grad);
				Matrix::ger(1, parent->lstm.ogGrad, tree->lstm.rep, grad->lstm.ogW[dir]);

				Matrix *temp = parent->lstm.pGrad->mul(parent->lstm.igRep[dir]);
				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.W[dir], temp, 1, tree->lstm.grad);
				Matrix::ger(1, temp, tree->lstm.rep, grad->lstm.W[dir]); delete temp;

				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.fgW, parent->lstm.fgGrad[dir], 1, tree->lstm.grad);
				Matrix::ger(1, parent->lstm.fgGrad[dir], tree->lstm.rep, grad->lstm.fgW);
				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.fgWSis, parent->lstm.fgGrad[1-dir], 1, tree->lstm.grad);
				Matrix::ger(1, parent->lstm.fgGrad[1-dir], tree->lstm.rep, grad->lstm.fgWSis);

				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.igW, parent->lstm.igGrad[dir], 1, tree->lstm.grad);
				Matrix::ger(1, parent->lstm.igGrad[dir], tree->lstm.rep, grad->lstm.igW);
				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.igWSis, parent->lstm.igGrad[1-dir], 1, tree->lstm.grad);
				Matrix::ger(1, parent->lstm.igGrad[1-dir], tree->lstm.rep, grad->lstm.igWSis);

				temp = parent->lstm.cGrad->mul(parent->lstm.fgRep[dir]);
				tree->lstm.cGrad->addi(temp); delete temp;

				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.igWc, parent->lstm.igGrad[dir], 1, tree->lstm.cGrad);
				Matrix::ger(1, parent->lstm.igGrad[dir], tree->lstm.cRep, grad->lstm.igWc);
				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.igWcSis, parent->lstm.igGrad[1-dir], 1, tree->lstm.cGrad);
				Matrix::ger(1, parent->lstm.igGrad[1-dir], tree->lstm.cRep, grad->lstm.igWcSis);

				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.fgWc, parent->lstm.fgGrad[dir], 1, tree->lstm.cGrad);
				Matrix::ger(1, parent->lstm.fgGrad[dir], tree->lstm.cRep, grad->lstm.fgWc);
				Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.fgWcSis, parent->lstm.fgGrad[1-dir], 1, tree->lstm.cGrad);
				Matrix::ger(1, parent->lstm.fgGrad[1-dir], tree->lstm.cRep, grad->lstm.fgWcSis);
			}
		}

		Matrix *tanhC = func(tree->lstm.cRep);
		Matrix *temp = funcPrime(tanhC)->muli(tree->lstm.ogRep)->muli(tree->lstm.grad);
		tree->lstm.cGrad->addi(temp); delete temp;
		tree->lstm.ogGrad = funcPrime(tree->lstm.ogRep, FUNC_SIGMOID)->muli(tanhC)->muli(tree->lstm.grad);
		delete tanhC;

		Matrix::gemv(1, CblasTrans, ((RNNParam*)params)->lstm.ogWc, tree->lstm.ogGrad, 1, tree->lstm.cGrad);
		Matrix::ger(1, tree->lstm.ogGrad, tree->lstm.cRep, grad->lstm.ogWc);
		grad->lstm.ogb->addi(tree->lstm.ogGrad);

		tree->lstm.pGrad = funcPrime(tree->lstm.pRep)->muli(tree->lstm.cGrad);
		grad->lstm.b->addi(tree->lstm.pGrad);

		for (int dir = 0; dir < 2; dir++) {
			if (tree->children[dir]->isLeaf()) {
				tree->lstm.fgGrad[dir] = Matrix::zeros(dim);

				Matrix *temp = Matrix::zeros(dim);
				Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.Ww[dir], tree->children[dir]->rep, 1, temp);
				tree->lstm.igGrad[dir] = funcPrime(tree->lstm.igRep[dir], FUNC_SIGMOID)->muli(temp)->muli(tree->lstm.pGrad);
				delete temp;
			}
			else {
				tree->lstm.fgGrad[dir] = funcPrime(tree->lstm.fgRep[dir], FUNC_SIGMOID)->muli(tree->children[dir]->lstm.cRep)->muli(tree->lstm.cGrad);

				Matrix *temp = Matrix::zeros(dim);
				Matrix::gemv(1, CblasNoTrans, ((RNNParam*)params)->lstm.W[dir], tree->children[dir]->lstm.rep, 1, temp);
				tree->lstm.igGrad[dir] = funcPrime(tree->lstm.igRep[dir], FUNC_SIGMOID)->muli(temp)->muli(tree->lstm.pGrad);
				delete temp;
			}
			grad->lstm.igb->addi(tree->lstm.igGrad[dir]);
			grad->lstm.fgb->addi(tree->lstm.fgGrad[dir]);
		}
	}

	for (int dir = 0; dir < 2; dir++) {
		if (tree->children[dir] != NULL)
			backprop(tree->children[dir], grad);
	}
}

void RNN::predict(SenBinTree* tree) {
	void ** ret = forward(tree, true);
	delete (real*)ret[0]; delete (int*)ret[1]; delete[] ret;
	tree->freeTemp();
}

void RNN::predict(vector<SenBinTree*> &trees) {
	for (SenBinTree* tree : trees) {
		void** ret = forward(tree, true);
		delete (real*)ret[0]; delete (int*)ret[1]; delete[] ret;
		tree->freeTemp();
	}
}

void** RNN::computeCostAndGrad(Treebank* tb, int startId, int endId, AbstractParam** absGradList) {
	RNNParam** gradList = (RNNParam**) absGradList;

	unordered_set<int>* tWords = new unordered_set<int>;

	// prepare data, the first entry is the main
	for (int i = 0; i < nThreads; i++)
		gradList[i]->fill(0);

	real* costList = new real[nThreads]();
	int* countList = new int[nThreads]();

	int step = ceil((endId - startId+1) / (float)nThreads);

#pragma omp parallel for num_threads(nThreads)
	for (int th = 0; th < nThreads; th++) {
		for (int i = startId + th*step; i <= min(endId, startId + (th+1)*step-1); i++) {
			SenBinTree* tree = tb->get(i);
			void** ret = forward(tree);
			costList[th] += *(real*)ret[0];
			countList[th] += *(int*)ret[1];
			delete (real*)ret[0]; delete (int*)ret[1]; delete[] ret;

			backprop(tree, gradList[th]);
			tree->freeTemp();
		}
	}

	for (int i = startId; i <= endId; i++) {
		SenBinTree* tree = tb->get(i);
		tree->getWords(*tWords);
	}

	RNNParam* grad = gradList[0];

	// merge
	for (int i = 1; i < nThreads; i++) {
		for (int j = 0; j < grad->nMatricesWoWE; j++)
			grad->weights[j]->addi(gradList[i]->weights[j]);

		for (unordered_set<int>::iterator it = tWords->begin(); it != tWords->end(); it++) {
			int word = *it;
			grad->L[word]->addi(gradList[i]->L[word]);
		}

		costList[0] += costList[i];
		countList[0] += countList[i];
	}

	// finalize
	real *cost = new real; *cost = costList[0];
	int nSample = endId - startId + 1; //countList[0];

	*cost /= nSample;

	// scaling gradient
	real norm = 0;
	for (int j = 0; j < grad->nMatricesWoWE; j++) {
		real nrm = Matrix::nrm2(grad->weights[j]);
		norm += nrm * nrm;
	}
	for (unordered_set<int>::iterator it = tWords->begin(); it != tWords->end(); it++) {
		int word = *it;
		real nrm = Matrix::nrm2(grad->L[word]);
		norm += nrm * nrm;
	}
	norm = sqrt(norm);
	if (norm > normGradThresh * nSample) {
		for (int j = 0; j < grad->nMatricesWoWE; j++)
			grad->weights[j]->muli(normGradThresh*nSample/norm);
		for (unordered_set<int>::iterator it = tWords->begin(); it != tWords->end(); it++) {
			int word = *it;
			grad->L[word]->muli(normGradThresh*nSample/norm);
		}
	}

	// L2-reg
	for (int i = 0; i < ((RNNParam*)params)->nMatricesWoWE; i++) {
		Matrix* W = ((RNNParam*)params)->weights[i];
		real l = lambda;
		if (W == ((RNNParam*)params)->Wc || W == ((RNNParam*)params)->Wwc || W == ((RNNParam*)params)->lstm.Wc ||
				W == ((RNNParam*)params)->bc || W == ((RNNParam*)params)->bwc)
			l = lambdaC;

		*cost += Utils::sumSqr(W->data, W->length) * l/2;
		grad->weights[i]->divi(nSample)->addi(l, W);
	}

	for (unordered_set<int>::iterator it = tWords->begin(); it != tWords->end(); it++) {
		int word = *it;
		Matrix* wemb = ((RNNParam*)params)->L[word];
		*cost += Utils::sumSqr(wemb->data, wemb->length) * lambdaL/2;
		grad->L[word]->divi(nSample)->addi(lambdaL, wemb);
	}

	delete[] costList;
	delete[] countList;
	return new void*[2] { cost, tWords};
}


