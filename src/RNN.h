/*
 * IORNN.h
 *
 *  Created on: Dec 17, 2014
 *      Author: Phong
 */

#ifndef IORNN_H_
#define IORNN_H_

#include "cblas.h"
#include <vector>
#include <unordered_set>
#include "Matrix.h"
#include "Default.h"
#include "RNNParam.h"
#include "SenBinTree.h"
#include "Treebank.h"
#include "AbstractNN.h"

using namespace std;

class RNN : public AbstractNN {
public:
	int wdim;

	RNN() : AbstractNN() {wdim = 0;}

	RNN(int dim, Dictionary* vocaDic, vector<Matrix*>& L, int compType, int funcType);

    void save(string fname);

    static RNN* load(string fname);

	AbstractParam* createGrad();

	void** forward(SenBinTree* tree, bool predict = false);

	void backprop(SenBinTree* tree, AbstractParam* grad, bool predict = false);

	void predict(SenBinTree* tree);

	void** computeCostAndGrad(Treebank* tb, int startId, int endId, AbstractParam** grad);

    void predict(vector<SenBinTree*> &trees);

};

#endif /* IORNN_H_ */
