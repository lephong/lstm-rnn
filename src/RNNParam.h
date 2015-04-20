/*
 * Param.h
 *
 *  Created on: Dec 17, 2014
 *      Author: Phong
 */

#ifndef PARAM_H_
#define PARAM_H_

#include "Dictionary.h"
#include "Matrix.h"
#include "AbstractParam.h"

#include <vector>
#include <fstream>
using namespace std;

struct LSTMLINK {
	Matrix *Ww[2];
	Matrix *W[2], *b;

	// for input gate
	Matrix *igWw, *igWwSis;
	Matrix *igW, *igWSis, *igb;
	Matrix *igWc, *igWcSis;

	// for outpute gate
	Matrix *ogWw[2];
	Matrix *ogW[2], *ogb;
	Matrix *ogWc;

	// for forget gate
	Matrix *fgWw, *fgWwSis;
	Matrix *fgW, *fgWSis, *fgb;
	Matrix *fgWc, *fgWcSis; // for cell

	// for classification
	Matrix *Wc;

	LSTMLINK() {
		for (int dir = 0; dir < 2; dir++) {
			Ww[dir] = W[dir] = NULL;
			ogWw[dir] = ogW[dir] = NULL;
		}
		igWw = igWwSis = igW = igWSis = igb = igWc = igWcSis = NULL;
		fgWw = fgWwSis = fgW = fgWSis = fgb = fgWc = fgWcSis = NULL; 
		ogWc = ogb = b = Wc = NULL;
	}
};

class RNNParam : public AbstractParam {
public:
	// word embeddings
	vector<Matrix*> L;

	// for ordinary link
	Matrix *Ww[2];
	Matrix *W[2], *b;

	// for classification
	Matrix *Wwc, *bwc, *Wc, *bc;

	// for LSTM link
	LSTMLINK lstm;


    RNNParam() : AbstractParam() {
    	Ww[0] = Ww[1] = W[0] = W[1] = b = Wwc = bwc = Wc = bc = NULL;
    	nMatricesWoWE = 0;
    };

	RNNParam (int dim, int wdim, int nwords, int nclass);
    void save(ofstream& f);
    static RNNParam* load(ifstream &f);
};

#endif /* PARAM_H_ */
