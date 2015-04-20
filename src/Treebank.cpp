/*
 * Treebank.cpp
 *
 *  Created on: Dec 18, 2014
 *      Author: Phong
 */

#include "Treebank.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "Utils.h"
#include <random>

using namespace std;

Treebank::~Treebank() {
    for (int i = 0; i < this->size(); i++)
        delete storage[i];
}

Treebank* Treebank::load(string fname, Dictionary* vocaDic) {
    cout << "load " << fname << endl;
    Treebank* tb = new Treebank();
	fstream reader(fname, ios::in);
	vector<string> rows;
	string line;

	while (getline(reader, line)) {
		SenBinTree* sbtree = SenBinTree::create(line, vocaDic);
		sbtree->isTop = true;
		tb->storage.push_back(sbtree);
	}

    cout << tb->size() << " trees" << endl;
	reader.close();
	return tb;
}

void Treebank::shuffle() {
	uniform_int_distribution<int> dis(0, size()-1);
	for (int i = (int)storage.size() - 1; i > 0; i--)
	{
		int index = dis(randGen);
        SenBinTree* temp = storage[index];
        storage[index] = storage[i];
        storage[i] = temp;
	}
}

Treebank* Treebank::extractAllSubTrees() {
	Treebank* tb = new Treebank();
	for (SenBinTree* tree : storage) {
		tree->extractAllSubTrees(tb->storage);
	}
	return tb;
}

Treebank* Treebank::bootstrap() {
	Treebank* tb = new Treebank();
	uniform_int_distribution<int> dis(0, size()-1);

	for (int i = 0; i < size() * 1.5; i++)
		tb->storage.push_back(get(dis(randGen))->deepCopy());

	return tb;
}
