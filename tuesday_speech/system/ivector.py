#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
This module contains tools T matrix training and i-vector extraction
EDITED FOR UEF SUMMERSCHOOL
"""

__version__ = '1.1'
__author__ = 'Omid Sadjadi, Timothee Kheyrkhah'
__email__ = 'omid.sadjadi@nist.gov'

import time
import numpy as np
from scipy.linalg import inv, svd, cholesky, solve


def unwrap_expectation_tv(args):
    return TMatrix.expectation_tv(*args)


class Ivector:

    def __init__(self, tv_dim, ndim, nmix):
        self.tv_dim = tv_dim
        self.ndim = ndim
        self.nmix = nmix
        self.itril = np.tril_indices(tv_dim)
        self.Sigma = np.empty((self.ndim * self.nmix, 1))
        self.T_iS = None  # np.empty((self.tv_dim, self.ndim * self.nmix))
        self.T_iS_Tt = None  # np.empty((self.nmix, self.tv_dim * (self.tv_dim+1)/2))
        self.Tm = np.empty((self.tv_dim, self.ndim * self.nmix))
        self.Im = np.eye(self.tv_dim)

    def load_ubm(self, ubm):
        sigma = ubm.sigma
        ndim, nmix = sigma.shape
        if self.ndim != ndim or self.nmix != nmix:
            raise ValueError('UBM nmix and ndim do not match what was specified!')
        self.Sigma = sigma.reshape((1, self.ndim * self.nmix), order='F')

    def tmat_init(self, T_mat_filename=""):
        np.random.seed(7)
        print('\n\nRandomly initializing T matrix ...\n')
        self.Tm = np.random.randn(self.tv_dim, self.ndim * self.nmix) * self.Sigma.sum() * 0.001

    def initialize(self, ubm, tmatrix):
        self.load_ubm(ubm)
        self.Tm = tmatrix
        self.T_iS = self.Tm / self.Sigma
        self.T_iS_Tt = self.comp_T_invS_Tt()

    def comp_T_invS_Tt(self):
        T_invS2 = self.Tm / np.sqrt(self.Sigma)
        T_invS_Tt = np.zeros((self.nmix, self.tv_dim * (self.tv_dim+1)//2))
        for mix in range(self.nmix):
            idx = np.arange(self.ndim) + mix * self.ndim
            tmp = T_invS2[:, idx].dot(T_invS2[:, idx].T)
            T_invS_Tt[mix] = tmp[self.itril]
        return T_invS_Tt

    def extract(self, N, F):
        F = np.squeeze(F)
        L = np.zeros((self.tv_dim, self.tv_dim))
        L[self.itril] = N.dot(self.T_iS_Tt)
        L += np.tril(L, -1).T + self.Im
        Cxx = inv(L)
        B = self.T_iS.dot(F)
        Ex = Cxx.dot(B)
        return Ex


class TMatrix(Ivector):

    def __init__(self, tv_dim, ndim, nmix, niter, nworkers):
        super().__init__(tv_dim, ndim, nmix)
        self.niter = niter
        self.nworkers = nworkers

    def train(self, dataList, ubm):
        nfiles = dataList.size
        self.load_ubm(ubm)
        self.tmat_init()

        print('Re-estimating the total subspace with {} factors ...'.format(self.tv_dim))

        for iter in range(self.niter):
            print('EM iter#: {} \t'.format(iter+1), end=" ")
            tic = time.time()
            RU, LU, LLK, nframes = self.expectation_tv(dataList)
            self.maximization_tv(LU, RU)
            self.min_div_est(LU, nframes)
            self.make_orthogonal()
            tac = time.time() - tic
            print('[llk = {0:.2f}] \t\t [elaps = {1:.2f}s]'.format(LLK/nfiles, tac))



    def expectation_tv(self, data):
        #N, F = data
        #nfiles = F.shape[0]
        #nframes = N.sum()
        LU = np.zeros((self.nmix, self.tv_dim * (self.tv_dim+1)//2))
        RU = np.zeros((self.tv_dim, self.nmix * self.ndim))
        LLK = 0.
        T_invS = self.Tm / self.Sigma
        T_iS_Tt = self.comp_T_invS_Tt()

        n_utterances = data.size
        length = 1

        nframes = 0

        for utterance in range(n_utterances):
            N1 = data[utterance][0]
            nframes += N1.sum()
            F1 = data[utterance][1].T
            L1 = N1.dot(T_iS_Tt)
            B1 = F1.dot(T_invS.T)
            Ex = np.empty((length, self.tv_dim))
            Exx = np.empty((length, self.tv_dim * (self.tv_dim + 1)//2))
            llk = np.zeros((length, 1))
            for ix in range(length):
                L = np.zeros((self.tv_dim, self.tv_dim))
                L[self.itril] = L1[ix]
                L += np.tril(L, -1).T + self.Im
                Cxx = inv(L)
                B = B1[ix][:, np.newaxis]
                this_Ex = Cxx.dot(B)
                llk[ix] = self.res_llk(this_Ex, B)
                Ex[ix] = this_Ex.T
                Exx[ix] = (Cxx + this_Ex.dot(this_Ex.T))[self.itril]
            RU += Ex.T.dot(F1)
            LU += N1.T.dot(Exx)
            LLK += llk.sum()
        return RU, LU, LLK, nframes

    def res_llk(self, Ex, B):
        return -0.5 * Ex.T.dot(B - Ex) + Ex.T.dot(B)

    @staticmethod
    def reduce_expectation_res(res):
        RU, LU, LLK, nframes = res[0]
        for r in res[1:]:
            ru, lu, llk, nfr = r
            RU += ru
            LU += lu
            LLK += llk
            nframes += nfr
        return LU, RU, LLK, nframes

    def maximization_tv(self, LU, RU):
        # ML re-estimation of the total subspace matrix or the factor loading
        # matrix
        for mix in range(self.nmix):
            idx = np.arange(self.ndim) + mix * self.ndim
            Lu = np.zeros((self.tv_dim, self.tv_dim))
            Lu[self.itril] = LU[mix, :]
            Lu += np.tril(Lu, -1).T
            self.Tm[:, idx] = solve(Lu, RU[:, idx])

    def min_div_est(self, LU, nframes):
        Lu = np.zeros((self.tv_dim, self.tv_dim))
        Lu[self.itril] = LU.sum(0)/nframes
        Lu += np.tril(Lu, -1).T
        self.Tm = cholesky(Lu).dot(self.Tm)

    def make_orthogonal(self):
        # orthogonalize the columns
        U, s, V = svd(self.Tm, full_matrices=False)
        self.Tm = np.diag(s).dot(V)
