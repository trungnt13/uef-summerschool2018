# -*- coding: utf-8 -*-
import os
import sys
from numbers import Number
from collections import OrderedDict, Mapping
import itertools

import numpy as np

# ===========================================================================
# For matplotlib
# ===========================================================================
def to_axis(ax, is_3D=False):
  """ Convert: int, tuple, None, Axes object
  to proper matplotlib Axes (2D and 3D)
  """
  from matplotlib import pyplot as plt
  # 3D plot
  if is_3D:
    from mpl_toolkits.mplot3d import Axes3D
    if ax is not None:
      assert isinstance(ax, (Axes3D, Number, tuple, list)), \
      'Axes3D must be used for 3D plot (z is given)'
      if isinstance(ax, Number):
        ax = plt.gcf().add_subplot(ax, projection='3d')
      elif isinstance(ax, (tuple, list)):
        ax = plt.gcf().add_subplot(*ax, projection='3d')
    else:
      ax = Axes3D(fig=plt.gcf())
  # 2D plot
  else:
    if isinstance(ax, Number):
      ax = plt.gcf().add_subplot(ax)
    elif isinstance(ax, (tuple, list)):
      ax = plt.gcf().add_subplot(*ax)
    elif ax is None:
      ax = plt.gca()
  return ax

def plot_spectrogram(x, vad=None, ax=None, colorbar=False,
                     linewidth=0.5, vmin=None, vmax=None,
                     title=None):
  '''
  Parameters
  ----------
  x : np.ndarray (frequency, time)
      2D matrix of the spectrogram in frequency-time domain
  vad : np.ndarray, list
      1D array, a red line will be draw at vad=1.
  ax : matplotlib.Axis
      create by fig.add_subplot, or plt.subplots
  colorbar : bool, 'all'
      whether adding colorbar to plot, if colorbar='all', call this
      methods after you add all subplots will create big colorbar
      for all your plots
  path : str
      if path is specified, save png image to given path

  Notes
  -----
  Make sure nrow and ncol in add_subplot is int or this error will show up
   - ValueError: The truth value of an array with more than one element is
      ambiguous. Use a.any() or a.all()

  Example
  -------
  >>> x = np.random.rand(2000, 1000)
  >>> fig = plt.figure()
  >>> ax = fig.add_subplot(2, 2, 1)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 2)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 3)
  >>> dnntoolkit.visual.plot_weights(x, ax)
  >>> ax = fig.add_subplot(2, 2, 4)
  >>> dnntoolkit.visual.plot_weights(x, ax, path='/Users/trungnt13/tmp/shit.png')
  >>> plt.show()
  '''
  from matplotlib import pyplot as plt

  # colormap = _cmap(x)
  # colormap = 'spectral'
  colormap = 'nipy_spectral'

  if x.ndim > 2:
    raise ValueError('No support for > 2D')
  elif x.ndim == 1:
    x = x[:, None]

  if vad is not None:
    vad = np.asarray(vad).ravel()
    if len(vad) != x.shape[1]:
      raise ValueError('Length of VAD must equal to signal length, but '
                       'length[vad]={} != length[signal]={}'.format(
                           len(vad), x.shape[1]))
    # normalize vad
    vad = np.cast[np.bool](vad)

  ax = to_axis(ax, is_3D=False)
  ax.set_aspect('equal', 'box')
  # ax.tick_params(axis='both', which='major', labelsize=6)
  ax.set_xticks([])
  ax.set_yticks([])
  # ax.axis('off')
  if title is not None:
    ax.set_ylabel(str(title) + '-' + str(x.shape), fontsize=6)
  img = ax.imshow(x, cmap=colormap, interpolation='kaiser', alpha=0.9,
                  vmin=vmin, vmax=vmax, origin='lower')
  # ====== draw vad vertical line ====== #
  if vad is not None:
    for i, j in enumerate(vad):
      if j: ax.axvline(x=i, ymin=0, ymax=1, color='r', linewidth=linewidth,
                       alpha=0.3)
  # plt.grid(True)
  if colorbar == 'all':
    fig = ax.get_figure()
    axes = fig.get_axes()
    fig.colorbar(img, ax=axes)
  elif colorbar:
    plt.colorbar(img, ax=ax)
  return ax

def plot_save(path='/tmp/tmp.pdf', figs=None, dpi=180,
              tight_plot=False, clear_all=True, log=True):
  """
  Parameters
  ----------
  clear_all: bool
      if True, remove all saved figures from current figure list
      in matplotlib
  """
  try:
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    if tight_plot:
      plt.tight_layout()
    if os.path.exists(path) and os.path.isfile(path):
      os.remove(path)
    pp = PdfPages(path)
    if figs is None:
      figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
      fig.savefig(pp, format='pdf', bbox_inches="tight")
    pp.close()
    if log:
      sys.stderr.write('Saved pdf figures to:%s \n' % str(path))
    if clear_all:
      plt.close('all')
  except Exception as e:
    sys.stderr.write('Cannot save figures to pdf, error:%s \n' % str(e))

def plot_confusion_matrix(cm, labels, ax=None, fontsize=12, colorbar=False,
                          title=None):
  from matplotlib import pyplot as plt
  cmap = plt.cm.Blues
  ax = to_axis(ax, is_3D=False)
  # calculate F1
  N_row = np.sum(cm, axis=-1)
  N_col = np.sum(cm, axis=0)
  TP = np.diagonal(cm)
  FP = N_col - TP
  FN = N_row - TP
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  F1 = 2 / (1 / precision + 1 / recall)
  F1[np.isnan(F1)] = 0.
  F1_mean = np.mean(F1)
  # column normalize
  nb_classes = cm.shape[0]
  cm = cm.astype('float32') / np.sum(cm, axis=1, keepdims=True)
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  # axis.get_figure().colorbar(im)
  tick_marks = np.arange(len(labels))
  ax.set_xticks(tick_marks)
  ax.set_yticks(tick_marks)
  ax.set_xticklabels(labels, rotation=-57, fontsize=fontsize)
  ax.set_yticklabels(labels, fontsize=fontsize)
  ax.set_ylabel('True label', fontsize=fontsize)
  ax.set_xlabel('Predicted label', fontsize=fontsize)
  # center text for value of each grid
  worst_index = {i: np.argmax([val if j != i else -1
                               for j, val in enumerate(row)])
                 for i, row in enumerate(cm)}
  for i, j in itertools.product(range(nb_classes),
                                range(nb_classes)):
    color = 'black'
    weight = 'normal'
    fs = fontsize
    text = '%.2f' % cm[i, j]
    if i == j: # diagonal
      color = 'magenta'
      # color = "darkgreen" if cm[i, j] <= 0.8 else 'forestgreen'
      weight = 'bold'
      fs = fontsize
      text = '%.2f\nF1:%.2f' % (cm[i, j], F1[i])
    elif j == worst_index[i]: # worst mis-classified
      color = 'red'
      weight = 'semibold'
      fs = fontsize
    plt.text(j, i, text,
             weight=weight, color=color, fontsize=fs,
             verticalalignment="center",
             horizontalalignment="center")
  # Turns off grid on the left Axis.
  ax.grid(False)
  # ====== colorbar ====== #
  if colorbar == 'all':
    fig = ax.get_figure()
    axes = fig.get_axes()
    fig.colorbar(im, ax=axes)
  elif colorbar:
    plt.colorbar(im, ax=ax)
  # ====== set title ====== #
  if title is None:
    title = ''
  title += ' (F1: %.3f)' % F1_mean
  ax.set_title(title, fontsize=fontsize + 2, weight='semibold')
  # axis.tight_layout()
  return ax
