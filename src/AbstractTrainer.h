/*
 * AbstractTrainer.h
 *
 *  Created on: Jan 22, 2015
 *      Author: phong
 */

#ifndef SRC_ABSTRACTTRAINER_H_
#define SRC_ABSTRACTTRAINER_H_

#include "Matrix.h"
#include "Utils.h"
#include <time.h>
#include <sys/time.h>
#include <unordered_set>
#include <iostream>
#include <iomanip>
#include "AbstractNN.h"
#include "AbstractParam.h"
#include "Treebank.h"
#include <vector>

using namespace std;


class AbstractTrainer {
public:

	Treebank* testtb;
    int startId;
    int endId;

    // for adagrad
    AbstractParam *paramVarience, *updateVarience;
	vector<int> nEvals;


	AbstractTrainer() { paramVarience = updateVarience = NULL; }
	virtual ~AbstractTrainer() {
		if (paramVarience != NULL) delete paramVarience;
		if (updateVarience != NULL) delete updateVarience;
	}

    virtual void adagrad(AbstractNN* net, AbstractParam* grad, unordered_set<int>& tWords) = 0;
    virtual void adadelta(AbstractNN* net, AbstractParam* grad, unordered_set<int>& tWords) = 0;
    virtual void sgd(AbstractNN* net, AbstractParam* grad, unordered_set<int>& tWords) = 0;

    void train(AbstractNN* net, Treebank* tb, Treebank* devTb);
};

#endif /* SRC_ABSTRACTTRAINER_H_ */
