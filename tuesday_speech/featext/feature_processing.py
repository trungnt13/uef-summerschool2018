#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:40:01 2018

@author: ville
"""
import os
from os import listdir
from os.path import isfile, join
import librosa
import numpy as np
import multiprocessing as mp
from functools import partial


def extract_features_from_file(file_path, feature_extractor, channel=0):
    audio, fs = load_audio(file_path, feature_extractor.fs)
    features = feature_extractor.extract(audio[:, channel])
    return features


def _extract_from_filelist_line(filelist_line, feature_extractor, output_folder):
    parts = filelist_line.split(',')
    audio, fs = load_audio(parts[0], feature_extractor.fs)
    for index in range(1, len(parts), 2):
        channel = int(parts[index]) - 1
        id = parts[index + 1].strip()
        features = feature_extractor.extract(audio[:, channel])
        filename = id + '.npy'
        output_file = os.path.join(output_folder, filename)
        np.save(output_file, features)


def extract_features_from_filelist(filelist_path, feature_extractor, output_folder, nworkers):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(filelist_path) as filelist_file:
        lines = filelist_file.readlines()
        with mp.Pool(nworkers) as p:
            p.map(partial(_extract_from_filelist_line, feature_extractor=feature_extractor, output_folder=output_folder), lines)

def load_features(feature_folder, feature_ids):
    feature_ids = open(feature_ids)
    feature_ids = feature_ids.readlines()
    features = np.empty((len(feature_ids), ), dtype=object)
    for index in range(len(feature_ids)):
        file = os.path.join(feature_folder, feature_ids[index].strip() + '.npy')
        features[index] = np.load(file)
    return features

def extract_all_stats(feature_folder, ubm, output_folder, nworkers):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    feat_files = [f for f in listdir(feature_folder) if isfile(join(feature_folder, f))]
    with mp.Pool(nworkers) as p:
        p.map(partial(_extract_stats_from_feature_file, input_folder=feature_folder, output_folder=output_folder, ubm=ubm),
              feat_files)


def _extract_stats_from_feature_file(file, input_folder, output_folder, ubm):
    features = np.load(join(input_folder, file))
    N, F = ubm.compute_centered_stats(features)
    np.savez(join(output_folder, file[:-4]), n=N, f=F)

def load_stats(stats_folder, feature_ids):
    stats = np.empty((len(feature_ids), ), dtype=object)
    for index in range(len(feature_ids)):
        file = os.path.join(stats_folder, feature_ids[index].strip() + '.npz')
        data = np.load(file)
        n = data['n']
        f = data['f']
        stats[index] = (n, f)
    return stats

def load_audio(file_path, target_fs=-1):
    parts = file_path.split('.')
    extension = parts[-1].lower()
    if extension == 'wav':
        if target_fs == -1:
            audio, fs = librosa.load(file_path)
        else:
            audio, fs = librosa.load(file_path, target_fs)
    else:
        print('Audio format not supported!')
        return

    return audio[:, None], fs


