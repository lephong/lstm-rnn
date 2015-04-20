/*
 * AbstractParam.cpp
 *
 *  Created on: Jan 22, 2015
 *      Author: phong
 */

#include "AbstractParam.h"
#include <iostream>
using namespace std;

AbstractParam::AbstractParam() {
	nMatricesWoWE = 0;
}

AbstractParam::~AbstractParam() {
	for (int i = 0; i < weights.size(); i++)
		delete weights[i];
}

void AbstractParam::fill(real value) {
	for (int i = 0; i < weights.size(); i++) {
		weights[i]->fill(value);
	}
}
