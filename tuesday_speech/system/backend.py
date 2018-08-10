#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
This module contains tools for backend modeling and scoring
EDITED FOR UEF SUMMERSCHOOL
"""

__version__ = '1.1'
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'

import numpy as np
from scipy.linalg import eigh, cholesky, inv, svd, solve
import time

class GPLDA:

    def __init__(self,tv_dim, nphi, niter):
        self.tv_dim = tv_dim
        self.nphi = nphi
        self.niter = niter
        self.Sigma = 1./self.tv_dim * np.eye(self.tv_dim)
        self.Phi = np.r_[np.eye(self.nphi), np.zeros((self.tv_dim-self.nphi, self.nphi))]
        self.Sb = np.zeros((self.tv_dim, self.tv_dim))
        self.St = np.zeros((self.tv_dim, self.tv_dim))

    def train_ml(self, data, spk_labs):
        classes, labels = unique(spk_labs, return_ind = True)
        nclasses = classes.size
        Sw = compute_within_cov(data, labels, nclasses)
        self.St = np.cov(data)
        self.Sb = self.St - Sw

    def train_em(self, data, spk_labs):
        # make sure the labels are sorted
        spk_labs = unique(spk_labs, return_ind = True)[1]
        spk_labs, I = np.sort(spk_labs), np.argsort(spk_labs)
        data = data[:, I]
        spk_counts = np.bincount(spk_labs) # sessions per speaker
        print('\n\nRandomly initializing the PLDA hyperparameters ...\n\n')
        # Sigma = np.cov(data.T)
        # Phi = np.random.randn((self.tv_dim, nphi))
        nspks = spk_counts.size
        F = np.zeros((self.tv_dim, nspks))
        cnt = 0
        for spk in range(nspks):
            # Speaker indices
            idx = np.arange(spk_counts[spk]) + cnt
            F[:, spk] = data[:, idx].sum(1)
            cnt += spk_counts[spk]
        data_cov = data.dot(data.T)
        print('Re-estimating the Eigenvoice subspace with {} factors ...\n'.format(self.nphi))
        for iter in range(self.niter):
            print('EM iter#: {} \t'.format(iter+1), end=" ")
            tic = time.time()
            # expectation
            Ey, Eyy = self.expectation_plda(data, F, spk_counts);
            # maximization
            self.maximization_plda(data, data_cov, F, Ey, Eyy)
            llk = self.comp_llk(data)
            toc = time.time() - tic
            print('[llk = {0:.2f}] \t [elaps = {1:.2f} s]'.format(llk, toc))
        self.Sb = self.Phi.dot(self.Phi.T)
        self.St = self.Sb + self.Sigma

    def expectation_plda(self, data, F, spk_counts):
        # computes the posterior mean and covariance of the factors
        nsamples = data.shape[1]
        nspks = spk_counts.size

        Eyy = np.zeros((self.nphi,self.nphi))
        Ey_spk = np.zeros((self.nphi, nspks))

        # initialize common terms to save computations
        uniqFreqs = unique(spk_counts)
        nuniq     = uniqFreqs.size
        invTerms  = np.empty((nuniq,self.nphi,self.nphi))
        PhiT_invS = solve(self.Sigma.T, self.Phi).T
        PhiT_invS_Phi = PhiT_invS.dot(self.Phi)
        I = np.eye(self.nphi)
        for ix in range(nuniq):
            nPhiT_invS_Phi = uniqFreqs[ix] * PhiT_invS_Phi
            invTerms[ix] = inv(I + nPhiT_invS_Phi)
        
        for spk in range(nspks):
            nsessions = spk_counts[spk]
            PhiT_invS_y = PhiT_invS.dot(F[:, spk])
            idx = np.flatnonzero(uniqFreqs == nsessions)[0]
            Cyy = invTerms[idx]
            Ey_spk[:, spk] = Cyy.dot(PhiT_invS_y)
            Eyy += nsessions * Cyy

        Eyy += (Ey_spk * spk_counts.T).dot(Ey_spk.T)
        return Ey_spk, Eyy
    
    def comp_llk(self, data):
        nsamples = data.shape[1]
        S = self.Phi.dot(self.Phi.T) + self.Sigma
        llk = -0.5 * (self.tv_dim * nsamples * np.log(2*np.pi) \
              + nsamples * logdet(S) + np.sum(data*solve(S,data)))
        return llk

    def maximization_plda(self, data, data_cov, F, Ey, Eyy):
        # ML re-estimation of the Eignevoice subspace and the covariance of the
        # residual noise (full).
        nsamples = data.shape[1]
        Ey_FT     = Ey.dot(F.T)
        self.Phi      = solve(Eyy.T,Ey_FT).T
        self.Sigma    = 1./nsamples * (data_cov - self.Phi.dot(Ey_FT))

    def score_trials(self, model_iv, test_iv):
        nphi = self.Phi.shape[0] 
        iSt = inv(self.St)
        iS = inv(self.St-self.Sb.dot(iSt).dot(self.Sb))
        Q = iSt-iS
        P = iSt.dot(self.Sb).dot(iS)
        U, s, V = svd(P, full_matrices=False)
        Lambda = np.diag(s[:nphi])
        Uk     = U[:,:nphi]
        Q_hat  = Uk.T.dot(Q).dot(Uk)
        model_iv = Uk.T.dot(model_iv)
        test_iv  = Uk.T.dot(test_iv)
        score_h1 = np.sum(model_iv.T.dot(Q_hat) * model_iv.T, 1, keepdims=True)
        score_h2 = np.sum(test_iv.T.dot(Q_hat) * test_iv.T, 1, keepdims=True)
        score_h1h2 = 2 * model_iv.T.dot(Lambda).dot(test_iv)
        scores = score_h1h2 + score_h1 + score_h2.T
        return scores


def unique(arr, return_ind=False):
    if return_ind:
        k = 0
        d = dict()
        uniques = np.empty(arr.size, dtype=arr.dtype)
        indexes = np.empty(arr.size, dtype='i')
        for i, a in enumerate(arr):
            if a in d:
                indexes[i] = d[a]
            else:
                indexes[i] = k
                uniques[k] = a
                d[a] = k
                k += 1
        return uniques[:k], indexes
    else:
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]


def unit_len_norm(data):
    data_norm = np.sqrt(np.sum(data * data, 0))
    data_norm[data_norm == 0] = 1.
    return data / data_norm


def calc_white_mat(Sw):
    # calculates the whitening transformation for cov matrix X
    w = cholesky(inv(Sw), lower=True)
    return w


def logdet(A):
    u = cholesky(A)
    y = 2*np.log(np.diag(u)).sum()
    return y


def wccn(data, labels):
    nclasses = np.unique(labels).size
    Sw = compute_within_cov(data, labels, nclasses)
    Sw = Sw + 1e-6 * np.eye(Sw.shape[0])
    return calc_white_mat(Sw)


def compute_class_avg(data, labels, nclasses):
    ndim = data.shape[0]
    mu_c = np.zeros((nclasses, ndim))
    for c in labels:  # numeric labels are assumed
        idx = np.flatnonzero(labels == c)
        mu_c[c] = data[:, idx].mean(1)
    return mu_c


def compute_within_cov(data, labels, nclasses, adapt=False):
    mu_c = compute_class_avg(data, labels, nclasses)  # numeric labels are assumed
    data_mu = data - mu_c[labels].T
    Sw = np.cov(data_mu)
    # Sw = data_mu.dot(data_mu.T)
    return Sw


def lda(data, labels, adapt=False):
    ndim, nobs = data.shape
    if nobs != len(labels):
        raise ValueError("oh dear! number of data samples ({}) should match the label size ({})!".format(nobs, len(labels)))

    M = data.mean(1, keepdims=True)  # centering the data
    data = data - M
    classes, labels = unique(labels, return_ind=True)  # make sure labels are numerical
    nclasses = classes.size
    Sw = compute_within_cov(data, labels, nclasses)
    St = np.cov(data)
    Sb = St - Sw
    D, V = eigh(Sb, Sw)
    D = D[::-1]
    V = V[:, ::-1]
    # the effective dimensionality of the transformation matrix
    Vdim = min(V.shape[1], nclasses - 1)
    V = V[:, :Vdim]
    D = D[:Vdim]
    # normalize the eigenvalues and eigenvectors
    D = D/D.sum()

    return V, D


def compute_mean(data, axis=-1):
    return data.mean(axis=axis, keepdims=True)


def preprocess(data, M=0., W=1., len_norm=True):
    data = data - M  # centering the data
    data = W.T.dot(data)  # whitening the data
    if len_norm:
        data = unit_len_norm(data)
    return data


def cosine_similarity(model_ivectors, test_ivectors):
    """ calculates a score matrix using the cosine similarity measure

        Inputs:
            - model_ivectors  : enrollment i-vectors, one speaker per column
            - test_ivectors   : test i-vectors, one sample per column

        Outputs:
            - scores          : score matrix, comparing all models against all tests
    """
    model_ivectors = unit_len_norm(model_ivectors)
    test_ivectors = unit_len_norm(test_ivectors)
    scores = model_ivectors.T.dot(test_ivectors)
    return scores
