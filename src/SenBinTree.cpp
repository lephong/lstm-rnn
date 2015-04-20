/*
 * SenBinTree.cpp
 *
 *  Created on: Jan 16, 2015
 *      Author: phong
 */

#include "SenBinTree.h"
#include "Utils.h"

SenBinTree::SenBinTree(int score, int word, SenBinTree* parent) {
	if (N_CLASS > 2)
		this->score = score;
	else {
		if (score < 2) this->score = 0;
		else if (score > 2) this->score = 1;
		else this->score = 2;
	}

	this->word = word;
	this->parent = parent;
	children[0] = NULL;
	children[1] = NULL;
	rep = NULL;
	prob = NULL;
	grad = NULL;
	dropout = NULL;
	isTop = false;
}

SenBinTree::SenBinTree(int score, SenBinTree* left, SenBinTree* right, SenBinTree* parent) {
	if (N_CLASS > 2)
		this->score = score;
	else {
		if (score < 2) this->score = 0;
		else if (score > 2) this->score = 1;
		else this->score = 2;
	}

	word = -1;
	rep = NULL;
	prob = NULL;
	grad = NULL;
	dropout = NULL;

	this->children[0] = left; if (children[0] != NULL) children[0]->parent = this;
	this->children[1] = right; if (children[1] != NULL) children[1]->parent = this;
	this->parent = parent;

	isTop = false;
}

SenBinTree::~SenBinTree() {
	freeTemp();
	if (rep != NULL) delete rep;
	if (prob != NULL) delete prob;
	if (grad != NULL) delete grad;
	if (dropout != NULL) delete dropout;
	if (children[0] != NULL) delete children[0];
	if (children[1] != NULL) delete children[1];
}

SenBinTree* SenBinTree::deepCopy() {
	SenBinTree* ret;

	if (isLeaf())
		ret = new SenBinTree(score, word, NULL);
	else {
		SenBinTree* ltree = children[0] == NULL ? NULL : children[0]->deepCopy();
		SenBinTree* rtree = children[1] == NULL ? NULL : children[1]->deepCopy();
		ret = new SenBinTree(score, ltree, rtree, NULL);
	}

	ret->isTop = isTop;
	return ret;
}

void SenBinTree::freeTemp() {
	if (rep != NULL) {
		delete rep;
		rep = NULL;
	}
	if (prob != NULL) {
		delete prob;
		prob = NULL;
	}
	if (grad != NULL) {
		delete grad;
		grad = NULL;
	}
	if (dropout != NULL) {
		delete dropout;
		dropout = NULL;
	}

	lstm.freeTemp();

	if (children[0] != NULL) children[0]->freeTemp();
	if (children[1] != NULL) children[1]->freeTemp();
}

void SenBinTree::getWords(unordered_set<int>& words) {
	if (isLeaf())
		words.insert(word);
	if (children[0] != NULL)
		children[0]->getWords(words);
	if (children[1] != NULL)
		children[1]->getWords(words);
}

SenBinTree* SenBinTree::create(string str, Dictionary* vocaDic) {
	return create(str, vocaDic, NULL);
}

SenBinTree* SenBinTree::create(string str, Dictionary* vocaDic, SenBinTree* parent) {
	string input = str.substr(1, str.length()-2);

	// check if this is a leaf
	if (input.back() != ')') {
		vector<string> comps = Utils::splitStringWoRegex(input, " ");
		return new SenBinTree(stoi(comps[0]), vocaDic->getId(comps[1]), parent);
	}

	// else
	int i1 = 0;

	while (input[i1] != ' ') i1++;
	int score = stoi(input.substr(0, i1));

	i1++; // a '(' should be at i1 now
	int i2 = i1+1;
	int count = 1;
	while (count > 0) {
		if (input[i2] == '(') count++;
		else if (input[i2] == ')') count--;
		i2++;
	}

	SenBinTree* left = create(input.substr(i1, i2-i1), vocaDic, NULL);
	SenBinTree* right = create(input.substr(i2+1, input.length()), vocaDic, NULL);
	SenBinTree* tree = new SenBinTree(score, left, right, parent);


	tree->conflict = false;
/*
	if (left->score >= 2 && right->score >= 2 && tree->score <= 2)
		if (left->score > 2 || right->score > 2)
			//cout << tree->toString(vocaDic) << endl;
			tree->conflict = true;

	if (left->score <= 2 && right->score <= 2 && tree->score >= 2)
		if (left->score < 2 || right->score < 2)
			//cout << tree->toString(vocaDic) << endl;
			tree->conflict = true;
*/
	return tree;
}

string SenBinTree::toString(Dictionary* vocaDic) {
	if (isLeaf())
		return "(" + to_string(score) + " " + vocaDic->id2word[word] + ")";
	else
		return "(" + to_string(score) + /*";" + to_string(Matrix::nrm2(lstm.rep)) + ";" + to_string(Matrix::nrm2(lstm.cRep)) + ";" +*/ " "
		+ /*to_string(Matrix::nrm2(lstm.igRep[0])) + ";" + to_string(Matrix::nrm2(lstm.fgRep[0])) + ";" +*/ children[0]->toString(vocaDic) + " "
		+ /*to_string(Matrix::nrm2(lstm.igRep[1])) + ";" + to_string(Matrix::nrm2(lstm.fgRep[1])) + ";" +*/ children[1]->toString(vocaDic) + ")";
}

void SenBinTree::extractAllSubTrees(vector<SenBinTree*>& trees) {
	if (isLeaf()) return;

	SenBinTree* copy = deepCopy();
	trees.push_back(copy);

	if (children[0] != NULL) children[0]->extractAllSubTrees(trees);
	if (children[1] != NULL) children[1]->extractAllSubTrees(trees);
}

