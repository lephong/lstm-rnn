//
//  IORNNTrainer.h
//  iornn-parser-c
//
//  Created by Phong Le on 25/12/14.
//  Copyright (c) 2014 Phong Le. All rights reserved.
//

#ifndef IORNNTrainer_h
#define IORNNTrainer_h

#include "AbstractTrainer.h"
#include "AbstractNN.h"
#include "AbstractParam.h"


class RNNTrainer : public AbstractTrainer {
public:

    RNNTrainer() :  AbstractTrainer() {}
    
    void adagrad(AbstractNN* net, AbstractParam* grad, unordered_set<int>& tWords);
    void adadelta(AbstractNN* net, AbstractParam* grad, unordered_set<int>& tWords);
    void sgd(AbstractNN* net, AbstractParam* grad, unordered_set<int>& tWords);
};

#endif
