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

range_ = ((-3, 3), (-3, 3))
fig = plt.figure(0, figsize=(12,12))
plt.subplot(2,2,1)
plt.title("Signal")
plt.xlabel("x")
plt.ylabel("y")
plt.hist2d(signal_train[:,0], signal_train[:,1], range=range_, bins=20, cmap=cm.coolwarm)
plt.subplot(2,2,2)
plt.title("Background")
plt.hist2d(background_train[:,0], background_train[:,1], range=range_, bins=20, cmap=cm.coolwarm)
plt.xlabel("x"), plt.ylabel("y")

#-> to actually plot it!
#plt.show()




########################################################
# define NN achitecture
model = Sequential()
model.add(Dense(100, activation="relu", input_dim=2))
model.add(Dense(100, activation="relu", input_dim=100))
model.add(Dense(50, activation="tanh"))
model.add(Dense(100, activation="tanh"))
model.add(Dense(50, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))

model.summary()


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



########################################################
# run the training

history = model.fit(data_train, labels_train,
          validation_data=(data_val, labels_val),
          batch_size=len(data_train),
          epochs=200)
          #epochs=100)


########################################################
# validation plot

ax = fig.add_subplot(2,2,3)

epochs = range(1, len(history.history["loss"])+1)
plt.plot(epochs, history.history["loss"], lw=3, label="Training loss")
plt.plot(epochs, history.history["val_loss"], lw=3, label="Validation loss")
plt.xlabel("Gradient step"), plt.ylabel("Cross-entropy loss");

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

ax = fig.add_subplot(2,2,4, projection="3d")
ax.view_init(30, -70)
ax.plot_surface(x, y, z, cmap=cm.coolwarm)
ax.set_xlabel("x", labelpad=12)
ax.set_ylabel("y", labelpad=12)
ax.set_zlabel("f", labelpad=12);

#plt.show()





########################################################
# plot structure of NN

from keras.utils import plot_model, print_summary
print_summary(model)

#figStructureNN = plt.figure(1, figsize=(5,5))

#plot_model(model)
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
plot_model(model, show_shapes=True, show_layer_names=True)

#tensorboard = TensorboardCallback()
#tensorboard.set_model(model) # your model here, will write graph etc
#tensorboard.on_train_end() # will close the writer


########################################################
# plots only at the end

plt.show()




