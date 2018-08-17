# This code provides template for practicing DNN in python using tensorflow
# and keras
# You could read the file "digit_recognizer.py" for references.
# fill the missing code marked with "TODO" label
from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import pickle
import warnings

import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from utils import (read_audio_files, performance_evaluate,
                   extract_acoustic_features, one_hot, stack_frames)
from plot_utils import (plot_save, plot_confusion_matrix, plot_spectrogram)
from ml_utils import show_tsne_clusters
from path import CACHE_PATH, FIG_PATH

# use the same random seed for reproducibility
np.random.seed(123456)
# ===========================================================================
# Constants control the training
# ===========================================================================
CONTEXT_LENGTH = 6 # i.e. 5 for left and 5 for right
BATCH_SIZE = 32 # 32
LEARNING_RATE = 0.01
NUM_EPOCH = 12
INPUT_FEATURE = 1 # 0 for power-spec, 1 for mel-spec, 2 for MFCCs
# ===========================================================================
# Reading audio and preprocessing features
# ===========================================================================
# ====== load audio ====== #
sample_rate, raw_data = read_audio_files()
# ====== acoustic features ====== #
feat_data = {}
# TODO: extractin acoustic feature and save the `feat_data` variable
# ====== infer digit and speaker information from file name ====== #
all_name = sorted(raw_data.keys())
digits = sorted(set([i.split('_')[0] for i in all_name]))
speakers = sorted(set([i.split('_')[1] for i in all_name]))
indices = sorted(set([i.split('_')[2] for i in all_name]))
print("Digits:", digits)
print("Speakers:", speakers)
# ===========================================================================
# Prepare data for training
# ===========================================================================
ids = np.random.permutation(len(speakers))
# TODO: splitting the dataset by speakers, adding speaker for
# train and score (test) to the two list:
#  - first 2 speakers for training
#  - last speaker for scoring
train_speakers = []
score_speakers = []
# ====== generate training and validating data ====== #
assert len(train_speakers) == 2 and len(score_speakers) == 1, "You must finish the above TODO"
X_train = []
y_train = []
X_score = []
y_score = []
first_sample = None
for name in all_name:
  features = feat_data[name]
  # change the INPUT_FEATURE index to
  # use different feature for training
  x = features[INPUT_FEATURE]
  num_frames, num_features = x.shape
  # adding context window
  # TODO: replace `None` with appropriate `frame_length` based on
  # `CONTEXT_LENGTH`, remember we have left context, right context
  # and main frame in the middle
  x = stack_frames(x, frame_length=None)
  # sequencing the image
  x = np.reshape(x, newshape=(num_frames, CONTEXT_LENGTH * 2 + 1, num_features))
  y = [int(name.split('_')[0])] * len(x)
  # add to appropriate set
  if any(spk in name for spk in train_speakers):
    X_train.append(x)
    y_train += y
    if first_sample is None:
      first_sample = name
  else:
    X_score.append(x)
    y_score += y
# ====== merge all array into a matrix ====== #
X_train = np.concatenate(X_train, axis=0)
y_train = np.array(y_train)
X_score = np.concatenate(X_score, axis=0)
y_score = np.array(y_score)
# convert labels to one-hot encoded vector
y_train = one_hot(y_train, nb_classes=len(digits))
y_score = one_hot(y_score, nb_classes=len(digits))
# ====== print some logs ====== #
print("Train speakers:", train_speakers, X_train.shape, y_train.shape)
print("Score speakers:", score_speakers, X_score.shape, y_score.shape)
input_shape = (None,) + X_train.shape[1:]
# ===========================================================================
# Create keras network using tensorflow
# ===========================================================================
# ====== prepare the Session ====== #
X = tf.placeholder(dtype=tf.float32, shape=(None,) + X_train.shape[1:], name="X")
y_true = tf.placeholder(dtype=tf.float32, shape=(None,) + y_train.shape[1:], name="y")
# ====== Create the network ====== #
model = keras.Sequential()
model.add(keras.layers.Dropout(rate=0.3))
# TODO: construct your own network statisfying following condition:
#  - Has Conv1D at the beginning
#  - Contain a Bidirectional recurrent neural network
#  - Using at 3 Dense layer (not counting the output layer)
# ====== output layer ====== #
# just to make sure everything is 2-D before output layer
model.add(keras.layers.Flatten(name='latent'))
# TODO: replace `None` here with appropriate number for output layer
model.add(keras.layers.Dense(units=None, activation='softmax'))
# ====== Configure a model for classification ====== #
optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.mean_squared_error])
model.build(input_shape=input_shape)
print("Input shape:", input_shape)
model.summary()
# ====== start the training ====== #
records = keras.callbacks.History()
model.fit(X_train, y_train,
          callbacks=[records],
          epochs=NUM_EPOCH, batch_size=BATCH_SIZE,
          validation_split=0.1)
# ====== plot the learning curve ====== #
# TODO: ploting mean squared error metrics extracted from
# training history
plt.figure(figsize=(8, 3)) # (ncol, nrow)
# ===========================================================================
# Evaluate the model
# ===========================================================================
# ====== evaluate the test data ====== #
y_pred_probas = model.predict(X_score, batch_size=BATCH_SIZE)
# TODO: we have `y_pred_probas` is the predicted probabilities for
# each classes, and `y_pred` is the predicted labels, replace `None`
# with appropriate value
y_pred = None
score_cm = confusion_matrix(y_true=np.argmax(y_score, axis=-1), y_pred=y_pred)
# ====== plotting the results ====== #
plt.figure(figsize=(16, 8)) # (ncol, nrow)
plot_confusion_matrix(score_cm, labels=digits, fontsize=8, title="Score Set")
plot_save(FIG_PATH)
