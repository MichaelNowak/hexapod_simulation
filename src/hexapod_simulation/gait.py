#!/usr/bin/env python
# coding: utf-8

import numpy as np


def wave(dim):
    seq = np.linspace(1, -1, dim)
    wave_matrix = np.array([np.roll(seq, i) for i in range(dim)])
    return wave_matrix


def test_gait():
    #dy = 0.1
    # gait_matrix = np.array(
    #     [[1.0, dy, -1.0 - dy, 1.0, dy, -1.0 - dy],
    #      [-1.0 - dy, 1.0, dy, -1.0 - dy, 1.0, dy],
    #      [dy, -1.0 - dy, 1.0, dy, -1.0 - dy, 1.0],
    #      [1.0, dy, -1.0 - dy, 1.0, dy, -1.0 - dy],
    #      [-1.0 - dy, 1.0, dy, -1.0 - dy, 1.0, dy],
    #      [dy, -1.0 - dy, 1.0, dy, -1.0 - dy, 1.0]]
    # )
    # gait_matrix = np.array([
    #     [0, 1],
    #     [1, 0]
    # ])
    # gait_matrix = np.array([
    #     [0, 1, -1],
    #     [1, 0, -1],
    #     [0, -1, 1]
    # ])
    def j(x):   # -1 -> 0
        return -6 * x + 2

    def k(x):   # 1 -> -1
        return 12 * x - 5

    def l(x):   # 1 -> 0.6
        return 1.2 * x + 0.4

    def m(x):   # 1 -> 0.2
        return 2.4 * x - 0.2

    def n(x):   # 1 -> -0.2
        return 3.6 * x - 0.8

    def o(x):   # 1 -> -0.6
        return 4.8 * x - 1.4

    def p(x):   # 1 -> -1
        return -j(x)

    def r(x):   # 0 -> 1
        return -6 * x + 2

    def s(x):   # 0 -> 0.6
        return -3.5 + 1.2

    def s(x):   # 0 -> 0.2
        return -1.2 + 0.4


    y = 0.18

    # tetrapod / wave
    # gait_matrix = np.array([
    #     [1, s(y), ]
    # ])

    # # tripod / tetrapod
    # gait_matrix = np.array([
    #     [1, j(y), k(y), -k(y), k(y), j(y)],
    #     [-1, 1, j(y), k(y), j(y), 1],
    #     [-j(y), -1, 1, j(y), 1, -1],
    #     [l(y), -j(y), -1, 1, -1, -j(y)],
    #     [m(y), -1, 1, j(y), 1, -1],
    #     [-1, 1, j(y), k(y), j(y), 1]
    # ])

    ##tripod / wave
    gait_matrix = np.array([
        [1, -o(y), m(y), -m(y), o(y), -1],
        [-1, 1, -o(y), m(y), -m(y), o(y)],
        [o(y), -1, 1, -o(y), m(y), -m(y)],
        [-m(y), o(y), -1, 1, -o(y), m(y)],
        [m(y), -m(y), o(y), -1, 1, -o(y)],
        [-o(y), m(y), -m(y), o(y), -1, 1]
    ])

    gait_matrix = np.array([
        [1, 0.1],
        [0.1, 1]
    ])

    gait_matrix = np.array([
        [0, 1, -1],
        [1, 0, -1],
        [0, -1, 1]
    ])

    # gait_matrix = np.array([
    #     [1, 0, -1],
    #     [0, 1, -1],
    #     [0, -1, 1]
    # ])

    gait_matrix = np.array([
        [1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1]
    ])

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
