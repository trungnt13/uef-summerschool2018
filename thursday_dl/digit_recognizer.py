import matplotlib
# use simple 'Agg' backend for figure to PDF, switch to 'TkAgg'
# if you want to show figure real time
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

from utils import (read_audio_files, performance_evaluate,
                   extract_acoustic_features, one_hot,
                   stack_frames, segment_axis)
from plot_utils import (plot_multiple_features, plot_save)

# use the same random seed for reproducibility
np.random.seed(123456)
# ===========================================================================
# Constants control the training
# ===========================================================================
CONTEXT_LENGTH = 4 # i.e. 5 for left and 5 for right
BATCH_SIZE = 32
# ===========================================================================
# Reading audio and preprocessing features
# ===========================================================================
# ====== load audio ====== #
with performance_evaluate(name="Reading Audio"):
  sample_rate, raw_data = read_audio_files()
all_name = sorted(raw_data.keys())
digits = sorted(set([i.split('_')[0] for i in all_name]))
speakers = sorted(set([i.split('_')[1] for i in all_name]))
indices = sorted(set([i.split('_')[2] for i in all_name]))
print("Digits:", digits)
print("Speakers:", speakers)
# ====== acoustic features ====== #
with performance_evaluate(name="Extract Features"):
  feat_data = {}
  for name, dat in raw_data.items():
    pow_spec, mel_spec, mfcc = extract_acoustic_features(dat)
    feat_data[name] = (pow_spec, mel_spec, mfcc)
# ====== pick out 1 number from 3 person for plotting ====== #
picked_digit = np.random.choice(a=digits, size=1, replace=False)[0]
picked_index = np.random.choice(a=indices, size=1, replace=False)[0]
for spk in speakers:
  name = '_'.join([picked_digit, spk, picked_index])
  raw = raw_data[name]
  pow_spec, mel_spec, mfcc = feat_data[name]
  plot_multiple_features(features={'raw': raw, 'spec': pow_spec,
                                   'mspec': mel_spec, 'mfcc': mfcc},
                         fig_width=2, title=name)
# ===========================================================================
# Prepare data for training
# ===========================================================================
ids = np.random.permutation(len(speakers))
# first 2 speakers for training
train_speakers = [speakers[i] for i in ids[:2]]
# last speaker for scoring
score_speakers = [speakers[i] for i in ids[2:]]
# ====== generate training and validating data ====== #
X_train = []
y_train = []
X_score = []
y_score = []
for name in all_name:
  pow_spec, mel_spec, mfcc = feat_data[name]
  # we use mfcc feature here, but can be changed
  x = mfcc
  # adding context window
  x = stack_frames(x, frame_length=CONTEXT_LENGTH * 2 + 1)
  y = [int(name.split('_')[0])] * len(x)
  # add to appropriate set
  if any(spk in name for spk in train_speakers):
    X_train.append(x)
    y_train += y
  else:
    X_score.append(x)
    y_score += y

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
# ===========================================================================
# Create keras network using tensorflow
# ===========================================================================
# ====== prepare the Session ====== #
X = tf.placeholder(dtype=tf.float32, shape=(None,) + X_train.shape[1:], name="X")
y_true = tf.placeholder(dtype=tf.float32, shape=(None,) + y_train.shape[1:], name="y")
# ====== Create the network ====== #
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(len(digits), activation='softmax'))
# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
# ====== start the training ====== #
model.fit(X_train, y_train, epochs=1, batch_size=BATCH_SIZE,
          validation_split=0.1)
# ===========================================================================
# Evaluate the model
# ===========================================================================
y_pred_probas = model.predict(X_score, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_probas, axis=-1)
report = classification_report(y_true=np.argmax(y_score, axis=-1), y_pred=y_pred)
cm = confusion_matrix(y_true=np.argmax(y_score, axis=-1), y_pred=y_pred)
print(report)
print(cm)
