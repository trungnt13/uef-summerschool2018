from __future__ import print_function, division, absolute_import

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils import performance_evaluate
from plot_utils import plot_save, plot_text_scatter

def show_tsne_clusters(X, y, title,
                       downsample_factor=32, is_3D=False,
                       seed=5218):
  from matplotlib import pyplot as plt
  # ====== check the arguments ====== #
  if X.ndim >= 3:
    X = np.reshape(X, newshape=(X.shape[0], -1))
  if y.ndim == 2:
    y = np.argmax(y, axis=-1)
  num_samples = X.shape[0]
  num_features = X.shape[1]
  rand = np.random.RandomState(seed=seed)
  # ====== good practice to apply PCA first ====== #
  with performance_evaluate(name="Fitting PCA"):
    pca = PCA(n_components=num_features // 3, random_state=rand)
    X_pca = pca.fit_transform(X)
  # ====== Down sampling if it take so long, only need for TSNE ====== #
  if downsample_factor > 1:
    ids = rand.permutation(num_samples)
    num_samples = num_samples // downsample_factor
    ids = ids[:num_samples]
    X = X[ids]
    X_pca = X_pca[ids]
    y = y[ids]
  # ====== TSNE ====== #
  with performance_evaluate(name="Fitting TSNE"):
    tsne = TSNE(n_components=3 if is_3D else 2, random_state=rand)
    X_tsne = tsne.fit_transform(X_pca)
  # ====== plotting ====== #
  font_size = 5
  plt.figure(figsize=(18, 8)) # (ncol, nrow)
  plot_text_scatter(X=X_pca[:, :(3 if is_3D else 2)],
                    text=y, font_size=font_size,
                    ax=(1, 2, 1), title="[PCA]   %s" % str(title))
  plot_text_scatter(X=X_tsne,
                    text=y, font_size=font_size,
                    ax=(1, 2, 2), title="[T-SNE] %s" % str(title))
