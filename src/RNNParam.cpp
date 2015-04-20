/*
 * Param.cpp
 *
 *  Created on: Dec 17, 2014
 *      Author: Phong
 */

#include <iostream>
#include <math.h>
#include "RNNParam.h"

using namespace std;

RNNParam::RNNParam (int dim, int wdim, int nwords, int nclass) {
	Matrix* I = Matrix::eye(dim)->muli(0.5);

	for (int dir = 0; dir < 2; dir++) {
		Ww[dir] = Matrix::uniform(dim, wdim, -1./sqrt(dim*2+wdim), 1./sqrt(dim*2+wdim));	weights.push_back(Ww[dir]);
		W[dir] = Matrix::uniform(dim, dim, -1./sqrt(dim*3), 1./sqrt(dim*3));				weights.push_back(W[dir]);
	}
	b = Matrix::uniform(dim, 0, 0); weights.push_back(b);

	// for classification
	Wwc = Matrix::uniform(nclass, wdim, -1./sqrt(wdim), 1./sqrt(wdim));	weights.push_back(Wwc);
	bwc = Matrix::uniform(nclass, 0, 0); 								weights.push_back(bwc);
	Wc = Matrix::uniform(nclass, dim, -1./sqrt(dim), 1./sqrt(dim)); 	weights.push_back(Wc);
	bc = Matrix::uniform(nclass, 0, 0); 								weights.push_back(bc);


	// for lstml
	for (int dir = 0; dir < 2; dir++) {
		lstm.Ww[dir] = Matrix::uniform(dim, wdim, -1./sqrt(2*dim+wdim), 1./sqrt(2*dim+wdim));	weights.push_back(lstm.Ww[dir]);
		lstm.W[dir] = Matrix::uniform(dim, dim, -1./sqrt(dim*3), 1./sqrt(dim*3));				weights.push_back(lstm.W[dir]);

		lstm.ogWw[dir] = Matrix::uniform(dim, wdim, -1./sqrt(4*dim+wdim), 1./sqrt(4*dim+wdim));	weights.push_back(lstm.ogWw[dir]);
		lstm.ogW[dir] = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));				weights.push_back(lstm.ogW[dir]);
	}

	lstm.ogWc = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));	weights.push_back(lstm.ogWc);

	lstm.b = Matrix::uniform(dim, 0, 0); 	weights.push_back(lstm.b);
	lstm.ogb = Matrix::uniform(dim, 0, 0); 	weights.push_back(lstm.ogb);

	lstm.igWw = Matrix::uniform(dim, wdim, -1./sqrt(4*dim+wdim), 1./sqrt(4*dim+wdim));		weights.push_back(lstm.igWw);
	lstm.igWwSis = Matrix::uniform(dim, wdim, -1./sqrt(4*dim+wdim), 1./sqrt(4*dim+wdim));	weights.push_back(lstm.igWwSis);
	lstm.igW = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));	 				weights.push_back(lstm.igW);
	lstm.igWSis = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));	 			weights.push_back(lstm.igWSis);
	lstm.igWc = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));		 			weights.push_back(lstm.igWc);
	lstm.igWcSis = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));	 			weights.push_back(lstm.igWcSis);
	lstm.igb = Matrix::uniform(dim, 0, 0); 													weights.push_back(lstm.igb);

	lstm.fgWw = Matrix::uniform(dim, wdim, -1./sqrt(4*dim+wdim), 1./sqrt(4*dim+wdim)); 		weights.push_back(lstm.fgWw);
	lstm.fgWwSis = Matrix::uniform(dim, wdim, -1./sqrt(4*dim+wdim), 1./sqrt(4*dim+wdim)); 	weights.push_back(lstm.fgWwSis);
	lstm.fgW = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));	 				weights.push_back(lstm.fgW);
	lstm.fgWSis = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));	 			weights.push_back(lstm.fgWSis);
	lstm.fgWc = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));		 			weights.push_back(lstm.fgWc);
	lstm.fgWcSis = Matrix::uniform(dim, dim, -1./sqrt(dim*5), 1./sqrt(dim*5));	 			weights.push_back(lstm.fgWcSis);
	lstm.fgb = Matrix::uniform(dim, 0, 0); 													weights.push_back(lstm.fgb);

	lstm.Wc = Matrix::uniform(nclass, dim, -1./sqrt(dim), 1./sqrt(dim)); 					weights.push_back(lstm.Wc);

	nMatricesWoWE = (int)weights.size();

	for (int i = 0; i < nwords; i++) {
		Matrix* w = Matrix::normal(wdim, 0., 1.);
		L.push_back(w);
		weights.push_back(w);
	}

	delete I;
}

void RNNParam::save(ofstream &f) {
	for (int dir = 0; dir < 2; dir++) {
		Ww[dir]->save(f);
		W[dir]->save(f);
	}
	b->save(f);

	// for classification
	Wwc->save(f);
	bwc->save(f);
	Wc->save(f);
	bc->save(f);


	// for lstml
	for (int dir = 0; dir < 2; dir++) {
		lstm.Ww[dir]->save(f);
		lstm.W[dir]->save(f);

		lstm.ogWw[dir]->save(f);
		lstm.ogW[dir]->save(f);
	}

	lstm.ogWc->save(f);

	lstm.b->save(f);
	lstm.ogb->save(f);

	lstm.igWw->save(f);
	lstm.igWwSis->save(f);
	lstm.igW->save(f);
	lstm.igWSis->save(f);
	lstm.igWc->save(f);
	lstm.igWcSis->save(f);
	lstm.igb->save(f);

	lstm.fgWw->save(f);
	lstm.fgWwSis->save(f);
	lstm.fgW->save(f);
	lstm.fgWSis->save(f);
	lstm.fgWc->save(f);
	lstm.fgWcSis->save(f);
	lstm.fgb->save(f);

	lstm.Wc->save(f);

	f << L.size() << endl;
	for (Matrix* w : L)
		w->save(f);

	f << endl;
}

RNNParam* RNNParam::load(ifstream &f) {
	RNNParam * param = new RNNParam();

	for (int dir = 0; dir < 2; dir++) {
		param->Ww[dir] = Matrix::load(f); param->weights.push_back(param->Ww[dir]);
		param->W[dir] = Matrix::load(f); param->weights.push_back(param->W[dir]);
	}
	param->b = Matrix::load(f); param->weights.push_back(param->b);

	// for classification
	param->Wwc = Matrix::load(f); param->weights.push_back(param->Wwc);
	param->bwc = Matrix::load(f); param->weights.push_back(param->bwc);
	param->Wc = Matrix::load(f); param->weights.push_back(param->Wc);
	param->bc = Matrix::load(f); param->weights.push_back(param->bc);


	// for lstml
	for (int dir = 0; dir < 2; dir++) {
		param->lstm.Ww[dir] = Matrix::load(f); param->weights.push_back(param->lstm.Ww[dir]);
		param->lstm.W[dir] = Matrix::load(f); param->weights.push_back(param->lstm.W[dir]);

		param->lstm.ogWw[dir] = Matrix::load(f); param->weights.push_back(param->lstm.ogWw[dir]);
		param->lstm.ogW[dir] = Matrix::load(f); param->weights.push_back(param->lstm.ogW[dir]);
	}

	param->lstm.ogWc = Matrix::load(f); param->weights.push_back(param->lstm.ogWc);

	param->lstm.b = Matrix::load(f); param->weights.push_back(param->lstm.b);
	param->lstm.ogb = Matrix::load(f); param->weights.push_back(param->lstm.ogb);

	param->lstm.igWw = Matrix::load(f); param->weights.push_back(param->lstm.igWw);
	param->lstm.igWwSis = Matrix::load(f); param->weights.push_back(param->lstm.igWwSis);
	param->lstm.igW = Matrix::load(f); param->weights.push_back(param->lstm.igW);
	param->lstm.igWSis = Matrix::load(f); param->weights.push_back(param->lstm.igWSis);
	param->lstm.igWc = Matrix::load(f); param->weights.push_back(param->lstm.igWc);
	param->lstm.igWcSis = Matrix::load(f); param->weights.push_back(param->lstm.igWcSis);
	param->lstm.igb = Matrix::load(f); param->weights.push_back(param->lstm.igb);

	param->lstm.fgWw = Matrix::load(f); param->weights.push_back(param->lstm.fgWw);
	param->lstm.fgWwSis = Matrix::load(f); param->weights.push_back(param->lstm.fgWwSis);
	param->lstm.fgW = Matrix::load(f); param->weights.push_back(param->lstm.fgW);
	param->lstm.fgWSis = Matrix::load(f); param->weights.push_back(param->lstm.fgWSis);
	param->lstm.fgWc = Matrix::load(f); param->weights.push_back(param->lstm.fgWc);
	param->lstm.fgWcSis = Matrix::load(f); param->weights.push_back(param->lstm.fgWcSis);
	param->lstm.fgb = Matrix::load(f); param->weights.push_back(param->lstm.fgb);

	param->lstm.Wc = Matrix::load(f); param->weights.push_back(param->lstm.Wc);

	int nwords;
	f >> nwords;
	for (int i = 0; i < nwords; i++) {
		Matrix *w = Matrix::load(f);
		param->L.push_back(w); 
		param->weights.push_back(w);
	}

	return param;
}
