/*
 * Dictionary.h
 *
 *  Created on: Dec 17, 2014
 *      Author: Phong
 */

#ifndef DICTIONARY_H_
#define DICTIONARY_H_


#include <unordered_map>
#include <string>
#include <fstream>
#include "Default.h"

using namespace std;

class Dictionary {
public:
	unordered_map<int, string> id2word;
	unordered_map<string, int> word2id;

	int templateType;
	string UNK;

	// for hierarchical softmax
	real** code;
	int** path;
	int* codeLength;

	Dictionary(int tt);
    ~Dictionary();
    
    void save(ofstream& f);
    static Dictionary* load(ifstream& f);
    
	static Dictionary* create(string fname, int tt);
    
	void load(string fname);
	int add(string word);

	int size();

	string preProcess(const string& str);

	int getId(const string& str);

	int getCapFeat(const string& str);

	void loadBinCode(string fname);
};

#endif /* DICTIONARY_H_ */
