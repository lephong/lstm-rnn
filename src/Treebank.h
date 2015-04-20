/*
 * Treebank.h
 *
 *  Created on: Dec 18, 2014
 *      Author: Phong
 */

#ifndef TREEBANK_H_
#define TREEBANK_H_

#include <vector>
#include "Dictionary.h"
#include "SenBinTree.h"
using namespace std;


class Treebank {
public:
	vector<SenBinTree*> storage;

    virtual ~Treebank();
	Treebank() {}

	static Treebank* load(string fname, Dictionary* vocaDic);

	void shuffle();
    inline SenBinTree* get(int i) {return storage[i];}
    inline int size() {return (int)storage.size();}

    Treebank* extractAllSubTrees();
    Treebank* bootstrap();
};

#endif /* TREEBANK_H_ */
