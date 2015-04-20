/*
 * Dictionary.cpp
 *
 *  Created on: Dec 17, 2014
 *      Author: Phong
 */

#include "Dictionary.h"
#include "Utils.h"
#include <unordered_map>

#include <fstream>
#include <iostream>
#include <regex>
#include <string>

using namespace std;

Dictionary::Dictionary(int tt) {
	templateType = tt;

	code = NULL;
	path = NULL;
	codeLength = NULL;

	switch (templateType) {
	case TEMPLATE_NONE:
	case TEMPLATE_COLLOBERT:
	case TEMPLATE_GLOVE:
		UNK = "UNKNOWN";
		break;

	case TEMPLATE_GLOVE_BIG:
		UNK = "#UNKNOWN#";
		break;

	default:
		Utils::error("unknown type");
	}
	
}


Dictionary::~Dictionary() {
    if (code == NULL) return;
    
    for (int i = 0; i < size(); i++) {
        delete[] code[i];
        delete[] path[i];
    }
    delete[] code;
    delete[] path;
    delete[] codeLength;
}

Dictionary* Dictionary::create(string fname, int tt) {
	Dictionary* dic = new Dictionary(tt);
	dic->load(fname);
	return dic;
}

void Dictionary::load(string fname) {
	word2id.clear();
	id2word.clear();

	ifstream f(fname, ios::in);

	int id = 0;
	string line;

	while (getline(f, line)) {
		word2id[line] = id;
		id2word[id] = line;
		id++;
	}

	f.close();
}

int Dictionary::add(string str) {
    string word = preProcess(str);
	unordered_map<string,int>::iterator got = word2id.find(word);
	if (got == word2id.end()) {
		int id = size();
		word2id[word] = id;
		id2word[id] = word;
		return id;
	}
	else
		return got->second;
}

int Dictionary::size() {
	return (int)word2id.size();
}

void Dictionary::save(ofstream &f) {
    f << size() << " " << templateType << endl;
    
    for (unordered_map<string,int>::iterator it = word2id.begin(); it != word2id.end(); it++)
        f << it->first << " " << it->second << endl;
    
    f << (code == NULL ? 0 : 1) << endl; // check if need to save code, codeLen, path
    if (code == NULL) return;
    
    for (int i = 0; i < size(); i++) {
        f << codeLength[i] << endl;
        for (int j = 0; j < codeLength[i]; j++)
            f << code[i][j] << " ";
        f << endl;
        for (int j = 0; j < codeLength[i]; j++)
            f << path[i][j] << " ";
        f << endl;
    }
    f << endl;
}

Dictionary* Dictionary::load(ifstream& f) {
    int s, tt;
    f >> s;
    f >> tt;
    
    Dictionary* dic = new Dictionary(tt);
    
    for (int i = 0; i < s; i++) {
        string word;
        int key;
        f >> word >> key;
        dic->word2id[word] = key;
        dic->id2word[key] = word;
    }
    
    int next; f >> next;
    if (next == 1) {
        dic->codeLength = new int[s];
        dic->code = new real*[s];
        dic->path = new int*[s];
        
        for (int i = 0; i < s; i++) {
            f >> dic->codeLength[i];
            dic->code[i] = new real[dic->codeLength[i]];
            dic->path[i] = new int[dic->codeLength[i]];
            for (int j = 0; j < dic->codeLength[i]; j++)
                f >> dic->code[i][j];
            for (int j = 0; j < dic->codeLength[i]; j++)
                f >> dic->path[i][j];
        }
    }
    
    return dic;
}

string Dictionary::preProcess(const string& str) {
	string s = str;
	switch (templateType) {
	case TEMPLATE_NONE:
		return s;

	case TEMPLATE_COLLOBERT:
		if (s == UNK || s == "PADDING")
			return s;
		else if (str == "-LRB-") return s.append("(");
		else if (str == "-RRB-") return s.append(")");
		else if (str == "-LSB-") return s.append("[");
		else if (str == "-RSB-") return s.append("]");
		else if (str == "-LCB-") return s.append("{");
		else if (str == "-RCB-") return s.append("}");
		else {
			transform(s.begin(), s.end(), s.begin(), ::tolower);
			char* cstr = new char[s.length()+1]();
			strcpy(cstr, s.c_str());
			for (unsigned int i = 0; i < s.length(); i++) {
				if (cstr[i] > '0' && cstr[i] <= '9')
					cstr[i] = '0';
			}
			return string(cstr);
		}

	case TEMPLATE_GLOVE:
		if (s == UNK)
			return s;
		else if (str == "-LRB-") return s.append("(");
		else if (str == "-RRB-") return s.append(")");
		else if (str == "-LSB-") return s.append("[");
		else if (str == "-RSB-") return s.append("]");
		else if (str == "-LCB-") return s.append("{");
		else if (str == "-RCB-") return s.append("}");
		else {
			transform(s.begin(), s.end(), s.begin(), ::tolower);
			return s;
		}

	case TEMPLATE_GLOVE_BIG:
		if (s == UNK)
			return s;
		else if (str == "-LRB-") return s.append("(");
		else if (str == "-RRB-") return s.append(")");
		else if (str == "-LSB-") return s.append("[");
		else if (str == "-RSB-") return s.append("]");
		else if (str == "-LCB-") return s.append("{");
		else if (str == "-RCB-") return s.append("}");
		else return s;

	default:
		return s;
	}
}

int Dictionary::getId(const string& str) {
    string word = preProcess(str);
	unordered_map<string,int>::iterator got = word2id.find(word);
	if (got == word2id.end())
		return word2id.find(UNK)->second;
	else
		return got->second;
}

int Dictionary::getCapFeat(const string& str) {
	return 0;
}

void Dictionary::loadBinCode(string fname) {
	codeLength = new int[size()]();
	code = new real*[size()];
	path = new int*[size()];
	string line;
	fstream f(fname, ios::in);

	while (getline(f, line)) {
		vector<string> comps = Utils::splitString(line, "[\t ]\+");
		int id = getId(comps[0]);
		string codeStr = comps[1];
		vector<string> pathStr = Utils::splitString(comps[2], "-");
		int len = (int)codeStr.length();

		real* c = new real[len];
        int* p = new int[len];
		for (int i = 0; i < len; i++) {
			c[i] = (codeStr[i] - '0') * 2 - 1; // -1 or + 1
            p[i] = stoi(pathStr[i]);
        }
		codeLength[id] = len;
		code[id] = c;
		path[id] = p;
	}
	f.close();
}


