lstm-rnn
========

A C++ implementation of the long short-term memory recursive neural network model (LSTM-RNN) described in

[1] Phong Le and Willem Zuidema (2015). [Compositional Distributional Semantics with Long Short Term Memory](http://arxiv.org/abs/1503.02510). In  Proceedings of Joint Conference on Lexical and Computational Semantics (\*SEM).

Written and maintained by Phong Le (p.le [at] uva.nl)

###Package
This package contains three components

+ `src` - source code files in C++ of the LSTM-RNN,

+ `data` - [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/treebank.html) and [GloVe word embeddings](http://nlp.stanford.edu/projects/glove/) 

+ `Release` - for compiling the source code.


###Installation

Install [OpenBlas](http://www.openblas.net) at `/opt/OpenBLAS`.
Go to `Release`, execute `make`. It should work with gcc 4.9 or later. 


###Usage

The following instruction is for replicating the second experiment reported in [1]. Some small changes are needed for your own cases.


####Data

`data/trees` contains the train/dev/test files of the Stanford Sentiment Treebank (STB).

`data/dic/glove-300d-840B` contains the 300-D GloVe word embeddings (vectors of words not in the STB are removed).  

If you want to use other word-embeddings, you should follow the following format: 

- create `words.lst` containing all frequent words (e.g. words appear at least 2 times), each word per line. The first line is always `#UNKNOWN#`
- create `wembs.txt` containing word vectors. The first line is `<number-of-vectors> <dimension>`. For each following line: `<word> <vector>`. Words in `words.lst` but not in `wembs.txt` will be randomly generated. 
	

####Train
In `Release`, open `train.config`, which stores the default parameter values. It should look like

	dim 50

	lambda 1e-3
	lambdaL 1e-3
	lambdaC 1e-3
	dropoutRate 0
	normGradThresh 1e10

	learningRateDecay 0.
	paramLearningRate 0.01
	wembLearningRate 0.01
	classLearningRate 0.01
	evalDevStep 1

	maxNEpoch 10
	batchSize 5
	nThreads 1

	dataDir ../data/trees/
	dicDir ../data/dic/glove-300d-840B/

	compositionType LSTM
	functionType tanh

You can try the traditional RNN by setting `compositionType NORMAL`, and try other activation functions by setting `functionType <NAME>` where `<NAME>` could be `softsign`, `sigmoid`, `rlu`.

Execute

	./lstm-rnn --train train.config model.dat

where the resulted model will be stored in file `model.dat`.


####Evaluation

Execute

    ./lstm-rnn --test <test-file> <model-file>

