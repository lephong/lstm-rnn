/*
 * Classifier.cpp
 *
 *  Created on: Jan 17, 2015
 *      Author: phong
 */

#include "Classifier.h"
#include "Default.h"
#include "AbstractNN.h"
#include <iostream>
#include <fstream>
#include "Utils.h"

using namespace std;

Classifier::Classifier() {
	// TODO Auto-generated constructor stub

}

Classifier::~Classifier() {
	// TODO Auto-generated destructor stub
}

std::pair<real,real> Classifier::eval(AbstractNN* net, Treebank* tb) {
	int nTrees = tb->size();
	net->predict(tb->storage);

	int correct = 0;
	int total = 0;
	int correctBin = 0;
	int totalBin = 0;

	for (int i = 0; i < nTrees; i++) {
		SenBinTree* tree = tb->storage[i];
		if (N_CLASS == 2) {
			if (tree->score < N_CLASS) {
				total++; totalBin++;
				if (tree->predictedScore == tree->score) {
					correct++;
					correctBin++;
				}
			}
		}
		else {
			total++;
			if (tree->predictedScore == tree->score)
				correct++;
			if (tree->score != 2) {
				totalBin++;
				if ((tree->score - 2)*tree->binPredScore > 0)
					correctBin++;
			}
		}
	}

	return std::pair<real,real>(correct / (real)total, correctBin / (real)totalBin);
}

std::pair<real,real> Classifier::eval(vector<AbstractNN*> nets, Treebank* tb) {
	int nTrees = tb->size();

	for (SenBinTree* tree : tb->storage) {
		if (tree->score < N_CLASS) {
			Matrix *score = Matrix::zeros(N_CLASS);

			for (AbstractNN* net : nets) {
				void** ret = net->forward(tree, true);
				delete (real*)ret[0]; delete (int*)ret[1]; delete[] ret;
				score->addi(tree->prob);
				tree->freeTemp();
			}

			tree->predictedScore = Utils::maxPos(score->data, N_CLASS);
			delete score;
		}
	}

	int correct = 0;
	int total = 0;
	int correctBin = 0;
	int totalBin = 0;

	for (int i = 0; i < nTrees; i++) {
		SenBinTree* tree = tb->storage[i];
		if (N_CLASS == 2) {
			if (tree->score < N_CLASS) {
				total++; totalBin++;
				if (tree->predictedScore == tree->score) {
					correct++;
					correctBin++;
				}
			}
		}
		else {
			total++;
			if (tree->predictedScore == tree->score)
				correct++;
			if (tree->score != 2) {
				totalBin++;
				if ((tree->score - 2)*tree->binPredScore > 0)
					correctBin++;
			}
		}
	}

	return std::pair<real,real>(correct / (real)total, correctBin / (real)totalBin);
}
