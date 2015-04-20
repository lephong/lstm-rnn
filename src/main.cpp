//
//  main.cpp
//  iornn-parser-c
//
//  Created by Phong Le on 27/12/14.
//  Copyright (c) 2014 Phong Le. All rights reserved.
//

#include <stdio.h>

/*
 * Reranker.cpp
 *
 *  Created on: Dec 24, 2014
 *      Author: Phong
 */

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "Matrix.h"
#include "Default.h"
#include "Dictionary.h"
#include "Treebank.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>


#include "RNN.h"
#include "RNNTrainer.h"
#include "SenBinTree.h"
#include "Classifier.h"

using namespace std;

//----------------------------- default -----------------------------

std::random_device rd;
std::mt19937 randGen(rd());

// for training
real lambda = 1e-4;
real lambdaL = 1e-4;
real lambdaC = 1e-5;
real dropoutRate = 0.;

real learningDecayRate = 0.9;
real paramLearningRate = 0.05;
real wembLearningRate = 0.05;
real classLearningRate = 0.05;
real repLearningRate = 1;
int evalDevStep = 1;

int maxNEpoch = 100;
int batchSize = 5;
int nThreads = 1;

// paths
#ifdef __GRADIENT_CHECKING__
int dim = 10;
Matrix* leafDropout = NULL;
Matrix* innerDropout = NULL;

string dataDir = "./data/toy/trees/";
string dicDir = "./data/toy/dic/glove-100d/";
real normGradThresh = 1000000000000;
#else
int dim = 50;
string dataDir = "../data/trees/";
string dicDir = "../data/dic/glove-50d/";
real normGradThresh = 1000000000000;
#endif
string modelPath = "/tmp/model";

// compose type
int composeType = COMPOSE_LSTM;
int funcType = FUNC_TANH;

// -------------------------------------------------------------------

void loadParamsForTraining(string fname) {
	ifstream f(fname);
	string line;
	while (getline(f, line)) {
		vector<string> comps = Utils::splitString(line, "[\t ]\+");
		if (comps.size() != 2) continue;

		if (comps[0] == "dim")
			dim = stoi(comps[1]);

		// for training
		if (comps[0] == "lambda")
			lambda = (real)stod(comps[1]);
		if (comps[0] == "lambdaL")
			lambdaL = (real)stod(comps[1]);
		if (comps[0] == "lambdaC")
			lambdaC = (real)stod(comps[1]);
		if (comps[0] == "dropoutRate")
			dropoutRate = (real)stod(comps[1]);
		if (comps[0] == "normGradThresh")
			normGradThresh = (real)stod(comps[1]);

		if (comps[0] == "paramLearningRate")
			paramLearningRate = (real)stod(comps[1]);
		if (comps[0] == "learningRateDecay")
			learningDecayRate = (real)stod(comps[1]);
		if (comps[0] == "wembLearningRate")
			wembLearningRate = (real)stod(comps[1]);
		if (comps[0] == "classLearningRate")
			classLearningRate = (real)stod(comps[1]);

		if (comps[0] == "maxNEpoch")
			maxNEpoch = stoi(comps[1]);
		if (comps[0] == "batchSize")
			batchSize = stoi(comps[1]);

		if (comps[0] == "evalDevStep")
			evalDevStep = stoi(comps[1]);

		if (comps[0] == "nThreads")
			nThreads = stoi(comps[1]);

		// paths
		if (comps[0] == "dataDir")
			dataDir = comps[1];
		if (comps[0] == "dicDir")
			dicDir = comps[1];

		// composition type
		if (comps[0] == "compositionType") {
			if (comps[1] == "LSTM") composeType = COMPOSE_LSTM;
			if (comps[1] == "NORMAL") composeType = COMPOSE_NORMAL;
			if (comps[1] == "COMBINE") composeType = COMPOSE_COMBINE;
		}
		if (comps[0] == "functionType") {
			if (comps[1] == "softsign") funcType = FUNC_SOFTSIGN;
			if (comps[1] == "tanh") funcType = FUNC_TANH;
			if (comps[1] == "sigmoid") funcType = FUNC_SIGMOID;
			if (comps[1] == "rlu") funcType = FUNC_RLU;
		}
	}
	f.close();
}

vector<Matrix*> loadWordEmbeddings(string fname, Dictionary* vocaDic) {
	cout << "load wordembeddings at " << fname << endl;
	vector<Matrix*> L;
	fstream f(fname, ios::in);
	int nwords, wdim;
	string word;
	f >> nwords;
	f >> wdim;

	L.resize(nwords + vocaDic->size());
	for (int i = 0; i < L.size(); i++)
		L[i] = NULL;

	Matrix* sum = Matrix::zeros(wdim);
	for (int i = 0; i < nwords; i++) {
		f >> word;
		int id = vocaDic->add(word);

		Matrix* we = new Matrix(wdim);
		for (int j = 0; j < wdim; j++)
			f >> we->data[j];
		L[id] = we;
		sum->addi(we);
	}
	f.close();

	L.resize(vocaDic->size());
	if (nwords > 0) sum->muli(1./nwords);
	for (int i = 0; i < L.size(); i++) {
		if (L[i] == NULL) {
			Matrix* we = Matrix::normal(wdim, 0, 1)->addi(sum);
			L[i] = we;
		}
	}

	delete sum;
	return L;
}

string getComposeType() {
	switch (composeType) {
	case COMPOSE_LSTM: return "LSTM";
	case COMPOSE_NORMAL: return "normal";
	case COMPOSE_COMBINE: return "combined"; 
	default: return "errorrrrrrrrrr";
	}
}

string getDefaultActFunc() {
	switch (funcType) {
	case FUNC_SIGMOID: return "sigmoid";
	case FUNC_TANH: return "tanh";
	case FUNC_SOFTSIGN: return "softsign";
	case FUNC_RLU: return "RLE";
	default: return "errorrrrrrrrrr";
	}
}

int main(int argc, char *argv[]) {
	if (argc == 1) {
		cout << "--train configfile modelfile" << endl;
		cout << "--test testfile modeldir" << endl;
		return 0;
	}

	if (string(argv[1]) == "--train") {
		if (argc != 4)
			Utils::error("--train configfile modelfile");

#ifndef __GRADIENT_CHECKING__
		loadParamsForTraining(string(argv[2]));
		modelPath = string(argv[3]);

		cout << "composeType: " << getComposeType() << endl;
		cout << "DEFAULT_FUNC: " << getDefaultActFunc() << endl;
		cout << "dim: " << dim << endl;
		cout << "lambda: " << lambda << endl;
		cout << "lambdaL: " << lambdaL << endl;
		cout << "lambdaC: " << lambdaC << endl;
		cout << "dropoutRate: "<< dropoutRate << endl;
		cout << "normGradThresh: " << normGradThresh << endl;
		cout << "learningRateDecay: " << learningDecayRate << endl;
		cout << "paramLearningRate: " << paramLearningRate << endl;
		cout << "wembLearningRate: " << wembLearningRate << endl;
		cout << "classLearningRate: " << classLearningRate << endl;
		cout << "maxNEpoch: " << maxNEpoch << endl;
		cout << "batchSize: " << batchSize << endl;

		cout << "modelPath: " << modelPath << endl;

		cout << "---------------------------------------------------------" << endl;
#endif

		openblas_set_num_threads(1);
		omp_set_num_threads(1);

		Dictionary* vocaDic = new Dictionary(TEMPLATE_GLOVE_BIG);
		vocaDic->load(dicDir + "words.lst");

		vector<Matrix*> L = loadWordEmbeddings(dicDir + "wembs.txt", vocaDic);

		Treebank* traintb = Treebank::load(dataDir + "train.txt", vocaDic);
		Treebank* devtb = Treebank::load(dataDir + "dev.txt", vocaDic);

		RNN* net = new RNN(dim, vocaDic, L, composeType, funcType);

#ifdef __GRADIENT_CHECKING__
		cout << dropoutRate << endl;
		leafDropout = Matrix::bernoulli(net->wdim, 1-dropoutRate);
		innerDropout = Matrix::bernoulli(net->dim, 1-dropoutRate);
		net->checkGradient(traintb);
#else
		RNNTrainer trainer;
		trainer.testtb = Treebank::load(dataDir + "test.txt", vocaDic);
		trainer.train(net, traintb, devtb);
		delete net;

		cout << "------------- test ---------------" << endl;
		net = RNN::load(modelPath);
		Classifier classifier;
		std::pair<real,real> tacc = classifier.eval(net, trainer.testtb);
		printf("%.2f\t%.2f\t[%.2fs]\n", tacc.first*100, tacc.second*100, time); fflush(stdout);
#endif

		delete traintb;
		delete devtb;

		return 0;
	}

	if (string(argv[1]) == "--test") {
		if (argc != 4)
		Utils::error("--test testfile model");

		string modelPath = string(argv[3]);
		AbstractNN* net = RNN::load(modelPath);

		Classifier classifier;
		Treebank* testtb = Treebank::load(string(argv[2]), net->vocaDic);

		struct timeval start, finish;
		gettimeofday(&start, NULL);
		std::pair<real,real> tacc = classifier.eval(net, testtb);
		gettimeofday(&finish, NULL);
		double time = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
		printf("%.2f\t%.2f\t[%.2fs]\n", tacc.first*100, tacc.second*100, time); fflush(stdout);

		delete net;
		delete testtb;

		return 0;
	}

	{
		cerr << "--train configfile modelfile" << endl;
		cerr << "--test testfile modeldir" << endl;
		return 0;
	}
}
