/*
 * AbstractTrainer.cpp
 *
 *  Created on: Jan 22, 2015
 *      Author: phong
 */

#include "AbstractTrainer.h"
#include "Classifier.h"
#include "RNN.h"
#include <stdlib.h>

void AbstractTrainer::train(AbstractNN* net, Treebank* tb, Treebank* devTb) {
	int nSample = tb->size();
	Classifier classifier;

	AbstractParam** gradList = new AbstractParam*[nThreads];
	for (int i = 0; i < nThreads; i++)
		gradList[i] = net->createGrad();

	int epoch = 0;
	struct timeval start, finish;

	int j = 0;
	real totalCost = 0;
	real bestScore = -1e10;

	epoch++;
	cout << "====================== train =====================" << endl;
	printf("----------- epoch %3.d ---------- \t", epoch);
	cout << Utils::currentDateTime() << "\t";

	tb->shuffle();
	//std::pair<real,real> dacc = classifier.eval(net, devTb);

	while (true) {
		j++;
		startId = (j-1) * batchSize;
		endId = min(nSample-1, j*batchSize-1);

		if (startId >= nSample) {
			cout << totalCost << "\t";

			if (epoch % evalDevStep == 0) {
				// eval
				cout << " === accuracy ";
				gettimeofday(&start, NULL);
				std::pair<real,real> dacc = classifier.eval(net, devTb);
				gettimeofday(&finish, NULL);
				double time = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
				printf("%.2f\t%.2f\t[%.2fs]\t", dacc.first*100, dacc.second*100, time); fflush(stdout);

				
				if (dacc.first > bestScore) {
					bestScore = dacc.first;
					net->save(modelPath);
				}
			}

			cout << endl;

			j = 1;
			startId = (j-1) * batchSize;
			endId = min(nSample-1, j*batchSize-1);
			epoch++;
			totalCost = 0;

			if (epoch > maxNEpoch) break;
			printf("----------- epoch %3.d ---------- \t", epoch);
			cout << Utils::currentDateTime() << "\t";
		}

		gettimeofday(&start, NULL);
		void** ret = net->computeCostAndGrad(tb, startId, endId, gradList);
		totalCost += *(real*)ret[0];
		unordered_set<int> *tWords = (unordered_set<int>*)ret[1];
		adagrad(net, gradList[0], *tWords);

		gettimeofday(&finish, NULL);
		double time = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;

		if (j % 500000 == 0) {
			printf("\n batch %4.d\t%.5f[%.2fs]", j, *(real*)ret[0], time);
			fflush(stdout);
		}


		delete ret[0]; delete tWords; delete[] ret;
	}

	for (int i = 0; i < nThreads; i++)
		delete gradList[i];
	delete[] gradList;
}

