/*
 * Classifier.h
 *
 *  Created on: Jan 17, 2015
 *      Author: phong
 */

#ifndef SRC_CLASSIFIER_H_
#define SRC_CLASSIFIER_H_

#include "Default.h"
#include "AbstractNN.h"
#include "Treebank.h"
#include <vector>
using namespace std;

class Classifier {
public:
	Classifier();
	virtual ~Classifier();

	std::pair<real,real> eval(AbstractNN* net, Treebank* tb);
	std::pair<real,real> eval(vector<AbstractNN*> nets, Treebank *tb);
};

#endif /* SRC_CLASSIFIER_H_ */
