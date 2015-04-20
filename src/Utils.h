/*
 * Utils.h
 *
 *  Created on: Dec 18, 2014
 *      Author: Phong
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <regex>
#include <cmath>
#include <dirent.h>
#include <sys/stat.h>
using namespace std;

#include "Default.h"
#include <math.h>
#include <time.h>

class Utils {
public:
	Utils();
	virtual ~Utils();

	static inline vector<string> splitStringWoRegex(string str, string delimiter) {
		char* s = new char[str.size()+1];
		memcpy(s, str.c_str(), sizeof(char)*str.size());
		s[str.size()] = '\0';
		const char* d = delimiter.c_str();

		vector<string> ret;
		char* token = strtok(s, d);
		if (token == NULL)
			ret.push_back("");

		else {
			ret.push_back(string(token));
			while ((token = strtok(NULL, d))) {
				ret.push_back(string(token));
			}
		}

		delete[] s;
		return ret;
	}

	static inline vector<string> splitString(string str, string delimiter) {
		vector<string> ret;
		regex rgx(delimiter);
		regex_token_iterator<string::iterator> iter(str.begin(), str.end(), rgx, -1);
		regex_token_iterator<string::iterator> end;
		for ( ; iter != end; ++iter)
			ret.push_back(*iter);
		return ret;
	}

	static real max(real* xs, int n) {
		real max = xs[0];
		for (int i = 1; i < n; i++)
			if (max < xs[i]) max = xs[i];
		return max;
	}

	static int maxPos(real* xs, int n) {
		int id = 0;
		for (int i = 1; i < n; i++)
			if (xs[id] < xs[i]) id = i;
		return id;
	}

	static real logSumOfExp(real* xs, int n) {
		if (n == 1) return xs[0];
		real m = max(xs, n);
		real sum = 0.0;
		for (int i = 0; i < n; ++i)
			sum += exp(xs[i] - m);
		return m + log(sum);
	}

	static void safelyComputeSoftmax(real* ys, real* xs, int n) {
		real logSum = logSumOfExp(xs, n);
		for (int i = 0; i < n; i++)
			ys[i] = exp(xs[i] - logSum);
	}

	static real sumLog(real* xs, int n) {
		real ret = 0;
		for (int i = 0; i < n; i++)
			ret += log(xs[i]);
		return ret;
	}

	static real sumSqr(real* xs, int n) {
		real ret = 0;
		for (int i = 0; i < n; i++)
			ret += xs[i] * xs[i];
		return ret;
	}

	static void error(string msg) {
		cerr << msg << endl;
		throw std::exception();
	}

	static string currentDateTime() {
		time_t     now = time(0);
		struct tm  tstruct;
		char       buf[80];
		tstruct = *localtime(&now);
		strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

		return buf;
	}

	static vector<string> getAllFiles(string path) {
		vector<string> ret;
		DIR *dir;
		dirent *ent;
		struct stat st;

		dir = opendir(path.c_str());
		while ((ent = readdir(dir)) != NULL) {
			string file_name = ent->d_name;
			string full_file_name = path + "/" + file_name;

			if (file_name[0] == '.')
				continue;

			if (stat(full_file_name.c_str(), &st) == -1)
				continue;

			const bool is_directory = (st.st_mode & S_IFDIR) != 0;

			if (is_directory)
				continue;

			ret.push_back(full_file_name);
		}
		closedir(dir);
		return ret;
	}
};

#endif /* UTILS_H_ */
