/*
 * AbstractParam.h
 *
 *  Created on: Jan 22, 2015
 *      Author: phong
 */

#ifndef SRC_ABSTRACTPARAM_H_
#define SRC_ABSTRACTPARAM_H_

#include <vector>
#include "Matrix.h"
using namespace std;


class AbstractParam {
public:
	vector<Matrix*> weights;
	int nMatricesWoWE;

	AbstractParam();
	virtual ~AbstractParam();
	virtual void fill(real value);
    virtual void save(ofstream& f) = 0;
};

#endif /* SRC_ABSTRACTPARAM_H_ */
