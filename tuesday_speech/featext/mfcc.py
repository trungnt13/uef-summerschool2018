"""
Created: Jul 2018

@author: ville
"""

from featext.base_extractor import BaseExtractor
import numpy as np
import librosa.util
import scipy.signal
import featext.feature_utils as feature_utils
import scipy.fftpack


class Mfcc(BaseExtractor):

    def __init__(self):

        BaseExtractor.__init__(self)

        self.n_fft = 512
        self.sad_threshold = 40

        self.frame_duration = 0.025  # in seconds
        self.frame_overlap_duration = 0.010

        self.include_base_coeffs = True
        self.include_deltas = True
        self.include_double_deltas = True
        self.delta_reach = 1

        self.min_frequency = 100
        self.max_frequency = 4000

        self.n_coeffs = 20
        self.include_energy = True

        self.pre_emphasis = 0.97  # 0 == no pre-emphasis
        self.rasta_coeff = 0.97  # 0 == no RASTA

        self.cmvn = True

        self._frame_size = 0
        self._frame_overlap = 0
        self._filterbank = 0
        self._filter_count = 0
        self._spectrum_size = 0


    def initialize(self):

        self._filter_count = self.n_coeffs + (not self.include_energy)
        self._spectrum_size = np.floor(self.n_fft / 2 + 1).astype(int)
        self._frame_size = np.round(self.frame_duration * self.fs).astype(int)
        self._frame_overlap = np.round(self.frame_overlap_duration * self.fs).astype(int)

        mel_limits = 2595 * np.log10(1 + np.array([self.min_frequency, self.max_frequency]) / 700)
        mel_freqs = np.linspace(mel_limits[0], mel_limits[1], self._filter_count + 2)
        hz_freqs = 700 * (np.power(10, (mel_freqs / 2595)) - 1)

        fft_bin_indices = np.floor((self.n_fft + 1) * hz_freqs / self.fs)
        self._filterbank = np.zeros((self._filter_count, self._spectrum_size))
        for filter in range(self._filter_count):
            self._filterbank[filter, :] = self._create_mel_filter(fft_bin_indices[filter:filter+3])

    def extract(self, audio):

        if self.pre_emphasis:
            audio[1:] -= self.pre_emphasis * audio[:-1]

        frames = librosa.util.frame(audio, self._frame_size, self._frame_size - self._frame_overlap)
        frames = frames * scipy.signal.windows.hamming(self._frame_size)[:, None]

        speech_activity_labels = feature_utils.get_sa_labels(frames, self.sad_threshold)

        frames = np.power(np.abs(scipy.fftpack.fft(frames, self.n_fft, axis=0)), 2)
        frames = frames[:self._spectrum_size, :]
        frames = np.log(self._filterbank @ frames + scipy.finfo(float).eps)
        frames = scipy.fftpack.dct(frames, axis=0)

        if not self.include_energy:
            frames = frames[1:, :]
        if self.rasta_coeff:
            frames = feature_utils.rasta_filter(frames, self.rasta_coeff)
        frames = feature_utils.append_deltas(frames, self.include_base_coeffs, self.include_deltas,
                                             self.include_double_deltas, self.delta_reach)
        frames = frames[:, speech_activity_labels]
        if self.cmvn:
            frames = feature_utils.cmvn(frames)
        return frames

    def _create_mel_filter(self, edges):
        leading_zeros = np.clip(edges[0], 0, self._spectrum_size).astype(int)
        trailing_zeros = np.clip(self._spectrum_size - edges[2] - 1, 0, self._spectrum_size).astype(int)
        triangle = np.concatenate((np.linspace(0, 1, edges[1] - edges[0], endpoint=False),
                                   np.linspace(1, 0, edges[2] - edges[1] + 1)))
        clipped_size = self._spectrum_size - (leading_zeros + trailing_zeros)
        clip_amount = np.size(triangle) - clipped_size;
        if clipped_size > 0:
            if leading_zeros == 0:
                return np.concatenate((triangle[clip_amount:], np.zeros(trailing_zeros)))
            elif trailing_zeros == 0:
                return np.concatenate((np.zeros(leading_zeros), triangle[:-clip_amount or None]))
            else:
                return np.concatenate((np.zeros(leading_zeros), triangle, np.zeros(trailing_zeros)))
        else:
            return np.zeros(self._spectrum_size)

    def get_feature_dim(self):
        return self._filter_count * (self.include_base_coeffs + self.include_deltas + self.include_double_deltas)