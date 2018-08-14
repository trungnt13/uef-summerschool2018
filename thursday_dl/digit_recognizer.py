import matplotlib
# use simple 'Agg' backend for figure to PDF, switch to 'TkAgg'
# if you want to show figure real time
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

from utils import (read_audio_files, performance_evaluate,
                   extract_acoustic_features, one_hot, stack_frames)
from plot_utils import (plot_save, plot_confusion_matrix, plot_spectrogram)
from ml_utils import show_tsne_clusters

# use the same random seed for reproducibility
np.random.seed(123456)
# ===========================================================================
# Constants control the training
# ===========================================================================
CONTEXT_LENGTH = 4 # i.e. 5 for left and 5 for right
BATCH_SIZE = 32
NUM_EPOCH = 1
# Two different input mode:
#  - one using sequencing: organize feature and context into 3-D tensor
#  - one using stacking: stack context with feature into long vector
INPUT_MODE = 'sequencing'
INPUT_FEATURE = 1 # 0 for power-spec, 1 for mel-spec, 2 for MFCCs
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
  # visualizing the features
  plt.figure(figsize=(4, 12)) # (ncol, nrow)
  plt.subplot(4, 1, 1)
  plt.plot(raw)
  plt.title('File: "%s" - Raw signal' % name)
  plt.gca().set_aspect(aspect='auto')
  plt.xticks([], [])
  plt.yticks([], [])

  plt.subplot(4, 1, 2)
  plot_spectrogram(pow_spec.T)
  plt.title("Power Spectrogram")
  plt.gca().set_aspect(aspect='auto')

  plt.subplot(4, 1, 3)
  plot_spectrogram(mel_spec.T)
  plt.title("Mel-filter banks Spectrogram")
  plt.gca().set_aspect(aspect='auto')

  plt.subplot(4, 1, 4)
  plot_spectrogram(mfcc.T)
  plt.title("MFCCs coefficients")
  plt.gca().set_aspect(aspect='auto')
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
first_sample = None
for name in all_name:
  features = feat_data[name]
  # change the INPUT_FEATURE index to
  # use different feature for training
  x = features[INPUT_FEATURE]
  num_frames, num_features = x.shape
  # adding context window
  x = stack_frames(x, frame_length=CONTEXT_LENGTH * 2 + 1)
  if INPUT_MODE == 'sequencing':
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
# ====== plotting some random samples ====== #
vmin = np.abs(feat_data[first_sample][INPUT_FEATURE]).min()
vmax = np.abs(feat_data[first_sample][INPUT_FEATURE]).max()
plt.figure(figsize=(12, 4)) # (ncol, nrow)
data = {}
for i in range(60):
  frame = X_train[i]
  y = np.argmax(y_train[i]) # note: y is one-hot
  if INPUT_MODE == 'stacking':
    frame = np.reshape(frame, newshape=(CONTEXT_LENGTH * 2 + 1, num_features))
  plot_spectrogram(x=frame.T, ax=(2, 30, i + 1),
                   vmin=vmin, vmax=vmax, title='#%d' % y)
# ====== print some logs ====== #
print("Train speakers:", train_speakers, X_train.shape, y_train.shape)
print("Score speakers:", score_speakers, X_score.shape, y_score.shape)
# writing comment for your code
# ====== extra fun, visualizing T-SNE clusters ====== #
show_tsne_clusters(X=X_train, y=y_train, title='Training set',
                   is_3D=False)
# ===========================================================================
# Create keras network using tensorflow
# ===========================================================================
# ====== prepare the Session ====== #
X = tf.placeholder(dtype=tf.float32, shape=(None,) + X_train.shape[1:], name="X")
y_true = tf.placeholder(dtype=tf.float32, shape=(None,) + y_train.shape[1:], name="y")
# ====== Create the network ====== #
model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(len(digits), activation='softmax'))
# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
# ====== start the training ====== #
model.fit(X_train, y_train, epochs=NUM_EPOCH, batch_size=BATCH_SIZE,
          validation_split=0.1)
# ===========================================================================
# Evaluate the model
# ===========================================================================
# ====== evaluate the train data ====== #
y_pred_probas = model.predict(X_train, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_probas, axis=-1)
train_report = classification_report(y_true=np.argmax(y_train, axis=-1), y_pred=y_pred)
train_cm = confusion_matrix(y_true=np.argmax(y_train, axis=-1), y_pred=y_pred)
# ====== evaluate the test data ====== #
y_pred_probas = model.predict(X_score, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_probas, axis=-1)
score_report = classification_report(y_true=np.argmax(y_score, axis=-1), y_pred=y_pred)
score_cm = confusion_matrix(y_true=np.argmax(y_score, axis=-1), y_pred=y_pred)
# ====== ploting the results ====== #
plt.figure(figsize=(16, 8)) # (ncol, nrow)
plot_confusion_matrix(train_cm, ax=(1, 2, 1), labels=digits, fontsize=8, title="Train")
plot_confusion_matrix(score_cm, ax=(1, 2, 2), labels=digits, fontsize=8, title="Score")
plot_save()
