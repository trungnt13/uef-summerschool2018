"""
Created: Jul 2018

@author: ville
"""


import numpy as np
from scipy.signal import lfilter


def get_sa_labels(frames, sad_threshold):
    energies = 20 * np.log10(np.std(frames, axis=0) + np.finfo(float).eps)
    max_energy = np.max(energies)
    return (energies > max_energy - sad_threshold) & (energies > -55)


def rasta_filter(frames, rasta_coeff):
    numerator = np.arange(-2, 3)
    numerator = -numerator / np.sum(numerator * numerator)
    denominator = np.array([1, -rasta_coeff])
    y, z = lfilter(numerator, 1, frames[:, :4], axis=1, zi=np.zeros((1, 4)))
    y = 0*y
    return np.concatenate((y, lfilter(numerator, denominator, frames[:, 4:], axis=1, zi=z)[0]), axis=1)


def _delta_function(frames, delta_reach):
    n_frames = frames.shape[1]
    if n_frames == 0:
        return frames
    win = np.arange(delta_reach, -delta_reach-1, -1)
    frames = np.concatenate((np.tile(frames[:, 0][:, None], [1, delta_reach]), frames,
                             np.tile(frames[:, -1][:, None], [1, delta_reach])), axis=1)
    frames = lfilter(win, 1, frames, axis=1)
    return frames[:, 2 * delta_reach:]


def append_deltas(base_coeffs, include_base_coeffs, include_deltas, include_double_deltas, delta_reach):
    n_coeffs = base_coeffs.shape[0]
    if include_deltas or include_double_deltas:
        deltas = _delta_function(base_coeffs, delta_reach)
        if include_deltas:
            base_coeffs = np.vstack((base_coeffs, deltas))
        if include_double_deltas:
            double_deltas = _delta_function(deltas, delta_reach)
            base_coeffs = np.vstack((base_coeffs, double_deltas))
    if not include_base_coeffs:
        base_coeffs = base_coeffs[n_coeffs:, :]
    return base_coeffs


def cmvn(frames):
    m = np.mean(frames, axis=1, keepdims=True)
    s = np.std(frames, axis=1, keepdims=True)
    frames = frames - m
    frames = frames / s
    return frames
