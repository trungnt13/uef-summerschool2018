import matplotlib
from matplotlib import pyplot as plt

import os
import pickle
import warnings

import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

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
BATCH_SIZE = 128
LEARNING_RATE = 0.01
NUM_EPOCH = 12
INPUT_FEATURE = 1 # 0 for power-spec, 1 for mel-spec, 2 for MFCCs
# ===========================================================================
# Reading audio and preprocessing features
# ===========================================================================
# found cached file
if os.path.exists(CACHE_PATH):
  try:
    with open(file=CACHE_PATH, mode='rb') as f:
      sample_rate, raw_data, feat_data = pickle.load(f)
      print("Loaded cached features at:", CACHE_PATH)
  except Exception as e:
    # somebody messed up the cached file, do everything again
    warnings.warn("Couldn't load cached file at path: %s, Error: %s" %
                  (CACHE_PATH, str(e)))
    os.remove(CACHE_PATH)
# check if cached file exist
if not os.path.exists(CACHE_PATH):
  # ====== load audio ====== #
  with performance_evaluate(name="Reading Audio"):
    sample_rate, raw_data = read_audio_files()
  # ====== acoustic features ====== #
  with performance_evaluate(name="Extract Features"):
    feat_data = {}
    for name, dat in raw_data.items():
      pow_spec, mel_spec, mfcc = extract_acoustic_features(dat)
      feat_data[name] = (pow_spec, mel_spec, mfcc)
  # ====== save cached features ====== #
  with open(file=CACHE_PATH, mode='wb') as f:
    pickle.dump((sample_rate, raw_data, feat_data), f)
    print("Saved cached features at:", CACHE_PATH)
# ====== infer digit and speaker information from file name ====== #
all_name = sorted(raw_data.keys())
digits = sorted(set([i.split('_')[0] for i in all_name]))
speakers = sorted(set([i.split('_')[1] for i in all_name]))
indices = sorted(set([i.split('_')[2] for i in all_name]))
print("Digits:", digits)
print("Speakers:", speakers)
# ====== pick out 1 number, then pick 1 sample from 3 people for plotting ====== #
picked_digit = np.random.choice(a=digits, size=1, replace=False)[0]
picked_index = np.random.choice(a=indices, size=1, replace=False)[0]
plt.figure(figsize=(12, 12)) # (ncol, nrow)
for spk_idx, spk in enumerate(speakers):
  name = '_'.join([picked_digit, spk, picked_index])
  raw = raw_data[name]
  pow_spec, mel_spec, mfcc = feat_data[name]
  # visualizing the features
  plt.subplot(4, 3, 1 + spk_idx)
  plt.plot(raw)
  plt.title('File: "%s" - Raw signal' % name)
  plt.gca().set_aspect(aspect='auto')
  plt.xticks([], []); plt.yticks([], [])

  plt.subplot(4, 3, 4 + spk_idx)
  plot_spectrogram(pow_spec.T)
  plt.title("Power Spectrogram")
  plt.gca().set_aspect(aspect='auto')

  plt.subplot(4, 3, 7 + spk_idx)
  plot_spectrogram(mel_spec.T)
  plt.title("Mel-filter banks Spectrogram")
  plt.gca().set_aspect(aspect='auto')

  plt.subplot(4, 3, 10 + spk_idx)
  plot_spectrogram(mfcc.T)
  plt.title("MFCCs coefficients")
  plt.gca().set_aspect(aspect='auto')

  # plt.show(block=True)
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
# ====== plotting some random samples ====== #
vmin = np.abs(feat_data[first_sample][INPUT_FEATURE]).min()
vmax = np.abs(feat_data[first_sample][INPUT_FEATURE]).max()
plt.figure(figsize=(12, 4)) # (ncol, nrow)
data = {}
for i in range(60):
  frame = X_train[i]
  y = np.argmax(y_train[i]) # note: y is one-hot
  plot_spectrogram(x=frame.T, ax=(2, 30, i + 1),
                   vmin=vmin, vmax=vmax, title='#%d' % y)
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
your_choice = 1
if your_choice == 1: # Dense require 2-D input so flatten everything
  model.add(keras.layers.Flatten())

  model.add(keras.layers.Dense(64, bias_initializer=None, activation='linear'))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Activation(activation='relu'))

  model.add(keras.layers.Dense(64, bias_initializer=None, activation='linear'))
  model.add(keras.layers.BatchNormalization())
  model.add(keras.layers.Activation(activation='relu'))
# ====== CNN ====== #
elif your_choice == 2: # CNN and RNN require at least 3-D input
  assert X.get_shape().ndims == 3
  data_format = 'channels_first'  # or'channels_last' (for frequency feature map)
  model.add(keras.layers.Conv1D(filters=16, kernel_size=3,
                                data_format=data_format))
  model.add(keras.layers.MaxPool1D(pool_size=2, data_format=data_format))
  model.add(keras.layers.Conv1D(filters=32, kernel_size=3,
                                data_format=data_format))
  model.add(keras.layers.MaxPool1D(pool_size=2, data_format=data_format))
elif your_choice == 3: # more advance CNN 2D
  assert X.get_shape().ndims == 3
  model.add(keras.layers.Lambda(function=lambda x: tf.expand_dims(x, axis=-1)))
  model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3)))
  model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
  model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3)))
  model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
# ====== RNN ====== #
elif your_choice == 4:
  assert X.get_shape().ndims == 3
  model.add(keras.layers.SimpleRNN(units=16, activation='relu', return_sequences=True))
  model.add(keras.layers.SimpleRNN(units=16, activation='relu', return_sequences=False))
elif your_choice == 5: # more advance Bidirectional RNN
  assert X.get_shape().ndims == 3
  model.add(keras.layers.Bidirectional(
      layer=keras.layers.SimpleRNN(units=8, activation='relu', return_sequences=True)))
  model.add(keras.layers.Bidirectional(
      layer=keras.layers.SimpleRNN(units=8, activation='relu', return_sequences=False)))
# ====== Exercise: CNN + RNN + Dense ====== #
elif your_choice == 6:
  raise NotImplementedError
# ====== output layer ====== #
# just to make sure everything is 2-D before output layer
model.add(keras.layers.Flatten(name='latent'))
model.add(keras.layers.Dense(len(digits), activation='softmax'))
# ====== Configure a model for classification ====== #
optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
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
plt.figure(figsize=(8, 3)) # (ncol, nrow)
plt.subplot(1, 2, 1)
plt.plot(records.history['loss'], color='red', label='Train')
plt.plot(records.history['val_loss'], color='blue', label='Valid')
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(records.history['categorical_accuracy'], color='red', label='Train')
plt.plot(records.history['val_categorical_accuracy'], color='blue', label='Valid')
plt.legend()
plt.title("Accuracy")

# plt.show(block=True)
# ===========================================================================
# Get intermediate representation and plot it
# ===========================================================================
intermediate_model = keras.Model(inputs=model.input,
                                 outputs=model.get_layer(name='latent').output)
intermediate_train = intermediate_model.predict(X_train, batch_size=BATCH_SIZE)
intermediate_score = intermediate_model.predict(X_score, batch_size=BATCH_SIZE)
# ====== extra fun, visualizing T-SNE clusters ====== #
show_tsne_clusters(X=X_score, y=y_score, title='Score - Acoustic Feat')
show_tsne_clusters(X=intermediate_score, y=y_score, title='Score - Latent Space')

# plt.show(block=True)
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
# ====== plotting the results ====== #
plt.figure(figsize=(16, 8)) # (ncol, nrow)
plot_confusion_matrix(train_cm, ax=(1, 2, 1), labels=digits, fontsize=8, title="Train")
plot_confusion_matrix(score_cm, ax=(1, 2, 2), labels=digits, fontsize=8, title="Score")

# plt.show(block=True)
plot_save(FIG_PATH)
