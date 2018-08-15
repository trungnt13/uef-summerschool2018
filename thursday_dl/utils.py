# -*- coding: utf-8 -*-
import os
import timeit
from contextlib import contextmanager

import numpy as np
from scipy.io import wavfile
from scipy import linalg, fftpack, signal

import librosa
from librosa import feature as acoustic_feature

from path import FSDD_PATH

def read_audio_files():
  """
  Return
  ------
  sample_rate : int
  outputs : dictionary (mapping name -> audio_raw_data)
  """
  all_files = [os.path.join(FSDD_PATH, i)
               for i in os.listdir(FSDD_PATH) if '.wav' == i[-4:]]
  assert len(all_files) > 0, "Cannot find .wav file at path: %s" % FSDD_PATH
  outputs = {}
  sample_rate = []
  print('======== Reading Audio Files ========')
  print('Found: %d audio files' % len(all_files))
  for i, path in enumerate(all_files):
    name = os.path.basename(path).replace('.wav', '')
    rate, data = wavfile.read(path)
    # store results
    sample_rate.append(rate)
    outputs[name] = data
    # logging
    if (i + 1) % 500 == 0:
      print("Loaded %d files ..." % len(outputs))
  assert len(set(sample_rate)) == 1, "Found multiple sample rate: %s" % str(set(sample_rate))
  return sample_rate[0], outputs

def extract_acoustic_features(data, sample_rate=8000,
                              n_fft=512, hop_length=0.005, win_length=0.025,
                              n_mels=40, n_mfcc=20, fmin=64.0, fmax=None):
  """
  data : array (n_samples,)
  sample_rate : int
  n_fft : int
  hop_length : float (in second)
  win_length : flaot (in second)
  """
  get_pow_spec = True
  get_mel_spec = True
  get_mfcc = True
  # ====== check arguments ====== #
  data = pre_emphasis(data)
  win_length = int(win_length * sample_rate)
  hop_length = int(hop_length * sample_rate)
  if fmax is None:
    fmax = sample_rate // 2
  results = []
  # ====== extract features ====== #
  s = librosa.stft(data.astype('float32'),
                   n_fft=n_fft, hop_length=hop_length, win_length=win_length)
  mag_spec = np.abs(s)
  if get_pow_spec:
    pow_spec = librosa.amplitude_to_db(mag_spec)
    results.append(pow_spec)
  if get_mel_spec or get_mfcc:
    mel_spec = acoustic_feature.melspectrogram(sr=sample_rate, S=mag_spec,
                                               n_mels=n_mels, fmin=fmin, fmax=fmax)
    results.append(mel_spec)
  if get_mfcc:
    mfcc = acoustic_feature.mfcc(sr=sample_rate, S=mel_spec, n_mfcc=n_mfcc)
    mfcc = rastafilt(mfcc.T).T
    results.append(mfcc)
  # ====== return results ====== #
  # normalizing features
  results = [cmvn(i) for i in results]
  # all the features are shape [feat_dim, time_dim]
  # deep network require order [time_dim, feat_dim]
  # so we transpose everythign
  return tuple([i.astype('float32').T for i in results])

# ===========================================================================
# Others
# ===========================================================================
def one_hot(y, nb_classes=None, dtype='float32'):
  '''Convert class vector (integers from 0 to nb_classes)
  to binary class matrix, for use with categorical_crossentropy

  Note
  ----
  if any class index in y is smaller than 0, then all of its one-hot
  values is 0.
  '''
  if 'int' not in str(y.dtype):
    y = y.astype('int32')
  if nb_classes is None:
    nb_classes = np.max(y) + 1
  else:
    nb_classes = int(nb_classes)
  return np.eye(nb_classes, dtype=dtype)[y]

def cmvn(frames):
  m = np.mean(frames, axis=1, keepdims=True)
  s = np.std(frames, axis=1, keepdims=True)
  frames = frames - m
  frames = frames / s
  return frames

def pre_emphasis(s, coeff=0.97):
  """Pre-emphasis of an audio signal.
  Parameters
  ----------
  s: np.ndarray
      the input vector of signal to pre emphasize
  coeff: float (0, 1)
      coefficience that defines the pre-emphasis filter.
  """
  if s.ndim == 1:
    return np.append(s[0], s[1:] - coeff * s[:-1])
  else:
    return s - np.c_[s[:, :1], s[:, :-1]] * coeff

def stack_frames(X, frame_length, step_length=1,
                 keep_length=True, make_contigous=True):
  """
  Parameters
  ----------
  X: numpy.ndarray
      2D arrray
  frame_length: int
      number of frames will be stacked into 1 sample.
  step_length: {int, None}
      number of shift frame, if None, its value equal to
      `frame_length // 2`
  keep_length: bool
      if True, padding zeros to begin and end of `X` to
      make the output array has the same length as original
      array.
  make_contigous: bool
      if True, use `numpy.ascontiguousarray` to ensure input `X`
      is contiguous.

  Example
  -------
  >>> X = [[ 0  1]
  ...      [ 2  3]
  ...      [ 4  5]
  ...      [ 6  7]
  ...      [ 8  9]
  ...      [10 11]
  ...      [12 13]
  ...      [14 15]
  ...      [16 17]
  ...      [18 19]]
  >>> frame_length = 5
  >>> step_length = 2
  >>> stack_frames(X, frame_length, step_length)
  >>> [[ 0  1  2  3  4  5  6  7  8  9]
  ...  [ 4  5  6  7  8  9 10 11 12 13]
  ...  [ 8  9 10 11 12 13 14 15 16 17]]
  """
  if keep_length:
    if step_length != 1:
      raise ValueError("`keepdims` is only supported when `step_length` = 1.")
    add_frames = (int(np.ceil(frame_length / 2)) - 1) * 2 + \
        (1 if frame_length % 2 == 0 else 0)
    right = add_frames // 2
    left = add_frames - right
    X = np.pad(X,
               pad_width=((left, right),) + ((0, 0),) * (X.ndim - 1),
               mode='constant')
  # ====== check input ====== #
  assert X.ndim == 2, "Only support 2D matrix for stacking frames."
  if not X.flags['C_CONTIGUOUS']:
    if make_contigous:
      X = np.ascontiguousarray(X)
    else:
      raise ValueError('Input buffer must be contiguous.')
  # ====== stacking ====== #
  frame_length = int(frame_length)
  if step_length is None:
    step_length = frame_length // 2
  shape = (1 + (X.shape[0] - frame_length) // step_length,
           frame_length * X.shape[1])
  strides = (X.strides[0] * step_length, X.strides[1])
  return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

def rastafilt(x):
  """ Based on rastafile.m by Dan Ellis
     rows of x = critical bands, cols of x = frame
     same for y but after filtering
     default filter is single pole at 0.94

  The filter is applied on frequency axis

  Parameters
  ----------
  x: [t, f]
      time x frequency
  """
  x = x.T # lazy style to reuse the code from [f, t] libraries
  ndim, nobs = x.shape
  numer = np.arange(-2, 3)
  # careful with division here (float point suggested by Ville Vestman)
  numer = -numer / np.sum(numer * numer)
  denom = [1, -0.94]
  y = np.zeros((ndim, 4))
  z = np.zeros((ndim, 4))
  zi = [0., 0., 0., 0.]
  for ix in range(ndim):
    y[ix, :], z[ix, :] = signal.lfilter(numer, 1, x[ix, :4], zi=zi, axis=-1)
  y = np.zeros((ndim, nobs))
  for ix in range(ndim):
    y[ix, 4:] = signal.lfilter(numer, denom, x[ix, 4:], zi=z[ix, :], axis=-1)[0]
  return y.T

@contextmanager
def performance_evaluate(name=None):
  start_time = timeit.default_timer()
  yield
  duration = timeit.default_timer() - start_time
  if name is None:
    name = "Task"
  print('[%s] finished in "%f" seconds' % (name, duration))
