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


########################################################
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



########################################################
# plot samples

#range_ = ((-3, 3), (-3, 3))
#plt.figure(0, figsize=(8,4))
#plt.subplot(1,2,1)
#plt.title("Signal")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.hist2d(signal_train[:,0], signal_train[:,1], range=range_, bins=20, cmap=cm.coolwarm)
#plt.subplot(1,2,2); plt.title("Background")
#plt.hist2d(background_train[:,0], background_train[:,1], range=range_, bins=20, cmap=cm.coolwarm)
#plt.xlabel("x"), plt.ylabel("y")

# -> to actually plot it!
# plt.show()




########################################################
# define NN achitecture
model = Sequential()
model.add(Dense(100, activation="relu", input_dim=2))
model.add(Dense(1, activation="sigmoid"))

model.summary()


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



########################################################
# run the training

history = model.fit(data_train, labels_train,
          validation_data=(data_val, labels_val),
          batch_size=len(data_train),
          epochs=100)


########################################################
# validation plot

#epochs = range(1, len(history.history["loss"])+1)
#plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
#plt.plot(epochs, history.history["val_loss"], lw=3, label="Validation loss")
#plt.xlabel("Gradient step"), plt.ylabel("Cross-entropy loss");

#plt.show()



########################################################
# learning curve

num_points = 100
grid = np.zeros((num_points**2, 2))
count = 0
for x_ in np.linspace(-3.0, 3.0, num_points):
    for y_ in np.linspace(-3.0, 3.0, num_points):
        grid[count, :] = (x_, y_)
        count += 1
        
f_ = model.predict(grid)

x = np.reshape(grid[:, 0], (num_points, num_points)).T
y = np.reshape(grid[:, 1], (num_points, num_points)).T
z = np.reshape(f_, (num_points, num_points)).T

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection="3d")
ax.view_init(30, -70)
ax.plot_surface(x, y, z, cmap=cm.coolwarm)
ax.set_xlabel("x", labelpad=12)
ax.set_ylabel("y", labelpad=12)
ax.set_zlabel("f", labelpad=12);

plt.show()







