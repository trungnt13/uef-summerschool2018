#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
This module contains tools for Gaussian mixture modeling (GMM)
EDITED FOR UEF SUMMERSCHOOL
"""

__version__ = '1.1'
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'

import time
import numpy as np
import multiprocessing as mp
import copy

EPS = np.finfo(float).eps


def unwrap_expectation(args):
    return GMM.expectation(*args)


class GmmUtils:

    def __init__(self):
        pass

    def postprob(self, data):
        post = self.lgmmprob(data)
        llk = logsumexp(post, 0)
        post = np.exp(post - llk)
        return post, llk

    def compute_C(self):
        precision = 1 / self.sigma
        log_det = np.sum(np.log(self.sigma), 0, keepdims=True)
        return np.sum(self.mu*self.mu*precision, 0, keepdims=True) + log_det - 2*np.log(self.w)

    def lgmmprob(self, data):
        precision = 1/self.sigma
        D = precision.T.dot(data*data)-2*(self.mu*precision).T.dot(data) + self.ndim * np.log(2*np.pi)
        return -0.5 * (self.C_.T + D)

    @staticmethod
    def compute_zeroStat(post):
        return np.sum(post, 1, keepdims=True).T

    @staticmethod
    def compute_firstStat(data, post):
        return data.dot(post.T)

    @staticmethod
    def compute_secondStat(data, post):
        return (data * data).dot(post.T)

    @staticmethod
    def compute_llk(post):
        return logsumexp(post, 1)


class GMM(GmmUtils):

    def __init__(self, ndim, nmix, ds_factor, final_niter, nworkers):
        if ((nmix & (nmix - 1)) == 0) and nmix > 0:
            self.nmix = nmix
        else:
            print('\n***GMM: oh dear! nmix ({}) should be a power of two!'.format(nmix))
            self.nmix = nextpow2(nmix)
            print('***rounding up to the nearest power of 2 number ({}).\n'.format(self.nmix))

        self.final_iter = final_niter
        self.ds_factor = ds_factor
        self.nworkers = nworkers
        self.ndim = ndim
        self.mu = np.zeros((ndim, 1), dtype='f4')
        self.sigma = np.ones((ndim, 1), dtype='f4')
        self.w = np.ones((1, 1), dtype='f4')
        self.C_ = self.compute_C()

    def fit(self, data_list):
        # binding of the main procedure gmm_em
        p = mp.Pool(processes=self.nworkers)
        if type(data_list) == str:
            features_list = np.genfromtxt(data_list, dtype='str')
        else:
            features_list = data_list
        nparts = min(self.nworkers, len(features_list))
        data_split = np.array_split(features_list, nparts)
        print('\nInitializing the GMM hyperparameters ...\n')
        # supports 4096 components, modify for more components
        niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 10, 10, 10]
        niter[int(np.log2(self.nmix))] = self.final_iter
        mix = 1
        while mix <= self.nmix:
            print('\nRe-estimating the GMM hyperparameters for {} components ...'.format(mix))
            for iter in range(niter[int(np.log2(mix))]):
                print('EM iter#: {} \t'.format(iter+1), end=" ")
                self.C_ = self.compute_C()
                tic = time.time()
                res = p.map(self.expectation, data_split)
                N, F, S, L, nframes = GMM.reduce_expectation_res(res)
                self.maximization(N, F, S)
                print("[llk = {:.2f}]\t[elaps = {:.2f}s]".format(L/nframes,time.time() - tic))
                del res
            if mix < self.nmix:
                self.gmm_mixup()
            mix *= 2
        p.close()

    # Added by Ville:
    def adapt_means(self, data, relevance_factor):
        N, F, S, L, nframes = self.expectation(data)
        alpha = N / (N + relevance_factor)  # tradeoff between ML mean and UBM mean
        m_ML = F / N
        m = self.mu * (1 - alpha) + m_ML * alpha
        return m

    # Added by Ville (scoring all models vs. all test segments):
    def score_with_adapted_means(self, model_means, test_features):
        n_models = model_means.size
        n_test_segments = test_features.size
        adapted_gmm = copy.deepcopy(self)
        scores = np.zeros((n_models, n_test_segments))
        for test_segment in range(n_test_segments):
            for model in range(n_models):
                adapted_gmm.mu = model_means[model]
                adapted_llk = adapted_gmm.compute_log_lik(test_features[test_segment])
                scores[model, test_segment] = np.mean(adapted_llk)
        return scores


    def expectation(self, data):
        N, F, S, L = 0., 0., 0., 0.
        nfr = 0
        for utterance in range(data.size):
            data_b = data[utterance]
            post, llk = self.postprob(data_b)
            N += GMM.compute_zeroStat(post)
            F += GMM.compute_firstStat(data_b, post)
            S += GMM.compute_secondStat(data_b, post)
            L += llk.sum()
            nfr += data_b.shape[1]
        return N, F, S, L, nfr

    @staticmethod
    def reduce_expectation_res(res):
        N, F, S, L, nframes = res[0]
        for r in res[1:]:
            n, f, s, l, nfr = r
            N += n
            F += f
            S += s
            L += l
            nframes += nfr
        return N, F, S, L, nframes

    def maximization(self, N, F, S):
        # TheReduce
        iN = 1. / (N + EPS)
        self.w = N / N.sum()
        self.mu = F * iN
        self.sigma = S * iN - self.mu*self.mu
#        self.apply_var_floors()

    def gmm_mixup(self):
        ndim, nmix = self.sigma.shape
        sig_max, arg_max = self.sigma.max(0), self.sigma.argmax(0)
        eps = np.zeros((ndim, nmix), dtype='f')
        eps[arg_max, np.arange(nmix)] = np.sqrt(sig_max)
        perturb = 0.55 * eps
        self.mu = np.c_[self.mu - perturb, self.mu + perturb]
        self.sigma = np.c_[self.sigma, self.sigma]
        self.w = 0.5 * np.c_[self.w, self.w]

    def apply_var_floors(self, floor_const=1e-3):
        vFloor = self.sigma.dot(self.w.T) * floor_const
        self.sigma = self.sigma.clip(vFloor)

    def compute_centered_stats(self, data):
        post = self.postprob(data)[0]
        N = GMM.compute_zeroStat(post)
        F = GMM.compute_firstStat(data, post)
        F_hat = np.reshape(F - self.mu * N, (self.ndim * self.nmix, 1), order='F')
        return N, F_hat

    def compute_log_lik(self, data):
        return self.postprob(data)[1]

    def load(self, gmmFilename):
        vars = np.load(gmmFilename)
        self.mu, self.sigma, self.w = vars['m'], vars['s'], vars['w']
        self.C_ = self.compute_C()

    def save(self, gmmFilename):
        np.savez(gmmFilename, m=self.mu, s=self.sigma, w=self.w)


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def logsumexp(x, dim):
    xmax = x.max(axis=dim, keepdims=True)
    y = xmax + np.log(np.sum(np.exp(x-xmax), axis=dim, keepdims=True))
    return y



