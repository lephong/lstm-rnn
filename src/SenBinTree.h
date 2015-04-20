/*
 * SenBinTree.h
 *
 *  Created on: Jan 16, 2015
 *      Author: phong
 */

#ifndef SRC_SENBINTREE_H_
#define SRC_SENBINTREE_H_

// sentiment binary tree

#include <string>
#include <unordered_set>
#include "Dictionary.h"
#include "Matrix.h"
#include <vector>
using namespace std;

struct LSTMSTATE {
	Matrix *rep, *grad;
	Matrix *pRep, *pGrad;

	Matrix *igRep[2], *igGrad[2];
	Matrix *ogRep, *ogGrad;
	Matrix *fgRep[2], *fgGrad[2];
	Matrix *cRep,  *cGrad;

	LSTMSTATE() {
		rep = grad = pRep = pGrad = igRep[0] = igRep[1] = igGrad[0] = igGrad[1] = ogRep = ogGrad =
				fgRep[0] = fgGrad[0] = fgRep[1] = fgGrad[1] = cRep = cGrad = NULL;
	}

	void freeTemp() {
		if (rep != NULL) delete rep;		rep = NULL;
		if (grad != NULL) delete grad;		grad = NULL;
		if (pRep != NULL) delete pRep;		pRep = NULL;
		if (pGrad != NULL) delete pGrad;	pGrad = NULL;

		if (ogRep != NULL) delete ogRep;	ogRep = NULL;
		if (ogGrad != NULL) delete ogGrad;	ogGrad = NULL;
		for (int dir=0; dir < 2; dir++) {
			if (igRep[dir] != NULL) delete igRep[dir];		igRep[dir] = NULL;
			if (igGrad[dir] != NULL) delete igGrad[dir];	igGrad[dir] = NULL;

			if (fgRep[dir] != NULL) 	delete fgRep[dir];	fgRep[dir] = NULL;
			if (fgGrad[dir] != NULL)	delete fgGrad[dir];	fgGrad[dir] = NULL;
		}

		if (cRep != NULL) delete cRep;		cRep = NULL;
		if (cGrad != NULL) delete cGrad;	cGrad = NULL;
	}
};

class SenBinTree {
public:
	SenBinTree *children[2];
	SenBinTree *parent;
	int score, word, predictedScore, binPredScore;
	Matrix *rep, *prob, *grad;
	bool isTop, conflict;
	LSTMSTATE lstm;
	Matrix *dropout;

	SenBinTree(int score, int word, SenBinTree* parent);
	SenBinTree(int score, SenBinTree *left, SenBinTree *right, SenBinTree* parent);

	static SenBinTree* create(string input, Dictionary* vocaDic, SenBinTree* parent);
	static SenBinTree* create(string input, Dictionary* vocaDic);
	string toString(Dictionary* vocaDic);

	inline bool isLeaf() {return children[0] == NULL && children[1] == NULL;}
	inline bool isRoot() {return parent == NULL;}
	void getWords(unordered_set<int> &words);

	void freeTemp();
	void extractAllSubTrees(vector<SenBinTree*>& trees);
	SenBinTree* deepCopy();

	virtual ~SenBinTree();
};

#endif /* SRC_SENBINTREE_H_ */
