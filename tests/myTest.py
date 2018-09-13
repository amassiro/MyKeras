#!/usr/bin/env python

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D as plt3d

from keras.models import Sequential
from keras.layers import Dense



# generate samples

num_events = 1000

signal_mean = [1.0, 1.0]
signal_cov = [[1.0, 0.0], [0.0, 1.0]]

signal_train = np.random.multivariate_normal( signal_mean, signal_cov, num_events )
signal_val = np.random.multivariate_normal(   signal_mean, signal_cov, num_events )

background_mean = [-1.0, -1.0]
background_cov = [[1.0, 0.0], [0.0, 1.0]]

background_train = np.random.multivariate_normal( background_mean, background_cov, num_events)
background_val = np.random.multivariate_normal(   background_mean, background_cov, num_events)

data_train = np.vstack([signal_train, background_train])
labels_train = np.vstack([np.ones((num_events, 1)), np.zeros((num_events, 1))])

data_val = np.vstack([signal_val, background_val])
labels_val = np.vstack([np.ones((num_events, 1)), np.zeros((num_events, 1))])


# plot samples

range_ = ((-3, 3), (-3, 3))
plt.figure(0, figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Signal")
plt.xlabel("x")
plt.ylabel("y")
plt.hist2d(signal_train[:,0], signal_train[:,1], range=range_, bins=20, cmap=cm.coolwarm)
plt.subplot(1,2,2); plt.title("Background")
plt.hist2d(background_train[:,0], background_train[:,1], range=range_, bins=20, cmap=cm.coolwarm)
plt.xlabel("x"), plt.ylabel("y");
