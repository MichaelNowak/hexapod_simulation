#!/usr/bin/env python
# coding: utf-8

import numpy as np


def wave(dim):
    seq = np.linspace(1, -1, dim)
    wave_matrix = np.array([np.roll(seq, i) for i in range(dim)])
    return wave_matrix


def test_gait():
    dy = 0.1
    gait_matrix = np.array(
        [[1.0, dy, -1.0 - dy, 1.0, dy, -1.0 - dy],
         [-1.0 - dy, 1.0, dy, -1.0 - dy, 1.0, dy],
         [dy, -1.0 - dy, 1.0, dy, -1.0 - dy, 1.0],
         [1.0, dy, -1.0 - dy, 1.0, dy, -1.0 - dy],
         [-1.0 - dy, 1.0, dy, -1.0 - dy, 1.0, dy],
         [dy, -1.0 - dy, 1.0, dy, -1.0 - dy, 1.0]]
    )
    return gait_matrix


def roll_gait(dim, sets):
    # descending equidistant discretization of interval [-1, 1] with respect to number of in-phase oscillators
    weights = np.linspace(1, -1, sets)
    # initialization of phase-shift array with respect to number of oscillators
    seq = np.zeros(dim)
    # predefined permutable sequence of phase shifts:
    # loop continuously through weights
    # sequence could be used from arguments
    for i in range(dim):
        seq[i] = np.roll(weights, -i)[0]  # weights array is permuted continuously and first entry is taken
    # gait matrix is permutation (np.roll()) of sequence array
    gait_matrix = np.array([np.roll(seq, i) for i in range(dim)])
    return gait_matrix


def index_all(lst, value):
    indices = [i for i, v in enumerate(lst) if v == value]
    return indices


def gait(dim, sets, shuffle=False):
    weights = np.linspace(1, -1, sets)
    seq = np.concatenate([np.arange(sets) for i in range(int(dim / sets))])
    cycle = np.zeros(dim)
    if shuffle:
        copy = seq[1:]
        np.random.shuffle(copy)
        seq[1:] = copy  # overwrite the original
    gait_matrix = np.zeros((dim, dim))
    for j in range(dim):
        for i in range(dim):
            cycle[i] = (seq[i] - seq[j] + sets) % sets
        for i in range(sets):
            for m in index_all(cycle, i):
                gait_matrix[j][m] = weights[i]
    return gait_matrix
