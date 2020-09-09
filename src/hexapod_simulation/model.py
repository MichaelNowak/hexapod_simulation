#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy

import numpy as np


def sigmoid(a, b, x):
    return 1 / (1 + np.exp(a * (b - x)))


def dynamical_system_nd(t, xb, scal, gain, ybar, taux, taub, gait_matrix):
    x, b = np.split(xb, 2)
    y = sigmoid(gain, b, x)
    xf = scal * np.dot(gait_matrix, y)
    x_dot = (xf - x) / taux
    b_dot = (y - ybar) / taub
    xb_dot = np.concatenate((x_dot, b_dot))
    return xb_dot


def solve_ivp_euler_nd(t_span, init_vals, dt, scal, gain, ybar, taux, taub, gait_matrix):
    xb = deepcopy(init_vals)
    xb_arr = deepcopy(xb)
    for t in range(len(t_span) - 1):
        xb_dot = dynamical_system_nd(t, xb, scal, gain, ybar, taux, taub, gait_matrix)
        xb += xb_dot * dt
        xb_arr = np.column_stack((xb_arr, xb))
    return xb_arr


def solve_euler_hexapod_nd(t_span, init_vals, dt, scal, gain, ybar, taux, taub, gait_matrix):
    x, b = np.split(init_vals, 2)
    x_arr = deepcopy(x)
    b_arr = deepcopy(b)
    for n in range(len(t_span) - 1):
        y = sigmoid(gain, b, x)
        xf = scal * np.dot(gait_matrix, y)
        x += (xf - x) / taux * dt
        b += (y - ybar) / taub * dt
        x_arr = np.column_stack((x_arr, x))
        b_arr = np.column_stack((b_arr, b))
    xb = np.concatenate((x_arr, b_arr))
    return xb


def solve_euler_hexapod_nd_sensor(t_span, init_vals, dt, scal, gain, ybar, taux, taub, gait_matrix, tau_sens, y_h_max,
                                  transpose_time=0, sensor_time=0, vsensor=False, hsensor=False, noise=False,
                                  constant_time_delay=False):
    sensor_vertical_bool = vsensor
    sensor_horizontal_bool = hsensor

    add_noise = noise

    # init_vals defines the dimension through the length of the array
    # dimensions of gait_matrix and init_vals must be suitable
    # initial arrays are 1d arrays
    x, b = np.split(init_vals, 2)
    x_arr = deepcopy(x)
    b_arr = deepcopy(b)

    y_h_arr = np.zeros(len(x))

    b_max = np.full_like(b, scal)
    b_min = np.full_like(b, 0)

    b_sens = deepcopy(b)

    sensor_act_vertical = np.zeros(len(x))
    sensor_act_horizontal = np.zeros(len(x))

    sensor_act_vertical_arr = np.zeros(len(x))
    sensor_act_horizontal_arr = np.zeros(len(x))

    dt_arr = np.array([0])

    # setting hardware specific motion range of actuators
    # here arbitrarily set equal with range of neural activation ([0, 1])
    # actuators' motion range is set equal to sensor value range
    # sensor values might differ from position values
    # in practice another mapping function might be needed
    # for the simulation not necessary

    # simulation with n steps
    for n in range(len(t_span) - 1):

        if sensor_horizontal_bool:
            lamda = 1
            y = sigmoid(gain, (1 - lamda) * b + lamda * b_sens, x)
            # y = sigmoid(gain, b_sens, x)
        elif sensor_vertical_bool:
            y = sigmoid(gain, b, x)
        else:
            y = sigmoid(gain, b, x)

        step_size = dt  # - np.random.normal(0, 0.01)

        if sensor_vertical_bool:
            kappa = 1
            xf = scal * np.dot(gait_matrix, (1 - kappa) * y + kappa * sensor_act_vertical)
            # xf = scal * np.dot(gait_matrix, sensor_act_vertical)
            b += ((1 - kappa) * y + kappa * sensor_act_vertical - ybar) / taub * step_size
            # b += (sensor_act_vertical - ybar) / taub * step_size
        else:
            xf = scal * np.dot(gait_matrix, y)
            b += (y - ybar) / taub * step_size

        x += (xf - x) / taux * step_size

        # algorithm for bias extrema b_min and b_max
        if n > 1:
            b_max = np.where(
                np.logical_and(b - b_arr[:, -1] < 0, (b - b_arr[:, -1]) * (b_arr[:, -1] - b_arr[:, -2]) < 0),
                b_arr[:, -1], b_max)
            b_max = np.where(b > b_max, b, b_max)
            b_min = np.where(
                np.logical_and(b - b_arr[:, -1] > 0, (b - b_arr[:, -1]) * (b_arr[:, -1] - b_arr[:, -2]) < 0),
                b_arr[:, -1], b_min)
            b_min = np.where(b < b_min, b, b_min)

        # horizontal actuator
        b_bar = (b_max + b_min) / 2
        gain_h = np.log(1 / y_h_max - 1) * 2 / (b_min - b_max)
        y_h = sigmoid(gain_h, b_bar, b)

        # sensor simulation on vertical and horizontal actuators
        if n == int(len(t_span) * (1 - transpose_time)):
            gait_matrix = np.transpose(gait_matrix)

        # added to normalized neural activation, therefore coefficient of variation as noise's standard deviation
        if add_noise:
            noise = np.random.normal(0, 0.03)  # empirical value after determining the coefficient of variation
        else:
            noise = 0

        #constant_time_delay = True
        if constant_time_delay:
            if n > 2:
                if n > int(len(t_span) * (1 - sensor_time)):
                    dh = 4
                else:
                    dh = 2

                sensor_act_vertical = sigmoid(gain, b_arr[:, n - dh], x_arr[:, n - dh]) + noise
                sensor_act_horizontal = y_h_arr[:, n - dh] + noise
            else:
                sensor_act_vertical = y
                sensor_act_horizontal = y_h
        else:
            if n > int(len(t_span) * (1 - sensor_time)):
                tau_s_h = tau_sens + 1. * (1 - deepcopy(y_h))
                tau_s_v = tau_sens + 1. * (1 - deepcopy(y))
            else:
                tau_s_h = tau_sens
                tau_s_v = tau_sens

            # continuously updating
            sensor_act_vertical += (y - sensor_act_vertical) / tau_s_v * dt + noise
            sensor_act_horizontal += (y_h - sensor_act_horizontal) / tau_s_h * dt + noise

        # sensor_act_vertical = np.where(sensor_act_vertical > 1, 1, sensor_act_vertical)
        # sensor_act_vertical = np.where(sensor_act_vertical < 0, 0, sensor_act_vertical)
        # bin_edges = np.arange(0, 1 + 1/25, 1/25)
        # bin_indices = np.digitize(sensor_act_vertical, bin_edges)
        # sensor_act_vertical = bin_edges[bin_indices-1]
        # print(sensor_act_vertical, sensor_act_vertical_new)

        # horizontal closed loop variable
        b_sens = (1 - sensor_act_horizontal) * b_min + sensor_act_horizontal * b_max

        # result arrays
        dt_arr = np.append(dt_arr, step_size)
        x_arr = np.column_stack((x_arr, x))
        b_arr = np.column_stack((b_arr, b))
        y_h_arr = np.column_stack((y_h_arr, y_h))
        sensor_act_vertical_arr = np.column_stack((sensor_act_vertical_arr, sensor_act_vertical))
        sensor_act_horizontal_arr = np.column_stack((sensor_act_horizontal_arr, sensor_act_horizontal))

    # concatenating result arrays
    xb = np.concatenate((x_arr, b_arr))
    vh = np.concatenate((xb, y_h_arr))
    sensors = np.concatenate((sensor_act_vertical_arr, sensor_act_horizontal_arr))
    vhsens = np.concatenate((vh, sensors))
    vhsens_time = np.concatenate((vhsens, [dt_arr]))

    return vhsens_time
