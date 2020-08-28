#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.io as sio
import bayes_classifier as bc

# load data
dat = sio.loadmat('fisherIrisData.mat')
X = dat['X']
y = np.squeeze(dat['y'])

# initialize model
model = bc.bayes_classifier(dist_type='full_gauss',prior_type='equal')

# cross-validation 
model.crossvalidate(X,y,rand_seed=100)

# training accuracy
training_acc = model.accuracy(X,y)
print('Training accuracy: {:1.3f}'.format(training_acc))


