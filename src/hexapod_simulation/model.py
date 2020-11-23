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

    y_arr = np.zeros(len(x))
    y_h_arr = np.zeros(len(x))

    delta_y = np.full_like(x, 0)
    delta_y_arr = deepcopy(y_arr)

    b_max = np.full_like(b, scal * np.sum(np.where(gait_matrix[0] == 1, 1, 0)))
    b_min = np.full_like(b, 0)

    b_dot = np.full_like(b, 0)
    b_dot_arr = deepcopy(b_dot)
    b_dot_max = np.full_like(b, 0)
    b_dot_min = np.full_like(b, 0)

    b_max_arr = deepcopy(b_max)
    b_min_arr = deepcopy(b_min)

    b_sens = deepcopy(b)
    b_sens_arr = deepcopy(b)

    sensor_act_vertical = np.zeros(len(x))
    sensor_act_horizontal = np.zeros(len(x))

    sensor_norm_vertical_arr = np.zeros(len(x))
    sensor_norm_horizontal_arr = np.zeros(len(x))

    #dh = np.zeros(len(x)).astype(int)

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
            #y = sigmoid(gain, b_sens, x)
        elif sensor_vertical_bool:
            y = sigmoid(gain, b, x)
        else:
            y = sigmoid(gain, b, x)

        step_size = dt  # - np.random.normal(0, 0.01)

        if sensor_vertical_bool and n > 1:
            kappa = 1
            xi = 1
            xf = scal * np.dot(gait_matrix, (1 - kappa) * y + kappa * sensor_norm_vertical)
            # xf = scal * np.dot(gait_matrix, sensor_act_vertical)
            b += ((1 - xi) * y + xi * sensor_norm_vertical - ybar) / taub * step_size
            b_dot = ((1 - xi) * y + xi * sensor_norm_vertical - ybar) / taub
            #b += (sensor_norm_vertical - ybar) / taub * step_size
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

            b_dot_max = np.where(
                np.logical_and(b_dot - b_dot_arr[:, -1] < 0, (b_dot - b_dot_arr[:, -1]) * (b_dot_arr[:, -1] - b_dot_arr[:, -2]) < 0),
                b_dot_arr[:, -1], b_dot_max)
            b_dot_max = np.where(b_dot > b_dot_max, b_dot, b_dot_max)
            b_dot_min = np.where(
                np.logical_and(b_dot - b_dot_arr[:, -1] > 0, (b_dot - b_dot_arr[:, -1]) * (b_dot_arr[:, -1] - b_dot_arr[:, -2]) < 0),
                b_dot_arr[:, -1], b_dot_min)
            b_dot_min = np.where(b_dot < b_dot_min, b_dot, b_dot_min)

            delta_y = taub * b_dot_min + ybar

            #print(delta_y)
            #print(b_dot_min)
            #print(ybar)

        # if n > 2:
        #     for i in range(len(b)):
        #         if (b[i] - b_arr[:, -1][i] < 0) \
        #                 and ((b[i] - b_arr[:, -1][i]) * (b_arr[:, -1][i] - b_arr[:, -2][i]) < 0):
        #             b_max[i] = b_arr[:, -1][i]
        #         if b[i] > b_max[i]:
        #             b_max[i] = b[i]
        #
        #         if (b[i] - b_arr[:, -1][i] > 0) \
        #                 and ((b[i] - b_arr[:, -1][i]) * (b_arr[:, -1][i] - b_arr[:, -2][i]) < 0):
        #             b_min[i] = b_arr[:, -1][i]
        #         if b[i] < b_min[i]:
        #             b_min[i] = b[i]

        # horizontal actuator
        b_bar = (b_max + b_min) / 2
        gain_h = 1 * np.log(1 / y_h_max - 1) * 2 / (b_min - b_max)
        #y_h = (b - b_min) / (b_max - b_min)
        y_h = sigmoid(gain_h, b_bar, b)

        # sensor simulation on vertical and horizontal actuators
        if n == int(len(t_span) * (1 - transpose_time)):
            gait_matrix = np.transpose(gait_matrix)

        # added to normalized neural activation, therefore coefficient of variation as noise's standard deviation
        if add_noise:
            noise = np.random.normal(0, 0.03)  # empirical value after determining the coefficient of variation
        else:
            noise = 0
        #if n > int(len(t_span) * (1 - sensor_time)):
        # continuously updating
        tau_pow_v = 0.1
        tau_pow_h = 0.1
        tau_sens_v = tau_sens + tau_pow_v * (1 - y)
        tau_sens_h = tau_sens + tau_pow_h * (1 - y_h)

        if constant_time_delay:
            if n > 3:
                delay = 2
                sensor_act_vertical = y_arr[:, -1 - delay] + noise
                sensor_act_horizontal = y_h_arr[:, -1 - delay] + noise
            else:
                sensor_act_vertical = y + noise
                sensor_act_horizontal = y_h + noise
        else:
            if n > 3:
                delay = 2
                sensor_act_vertical += (y_arr[:, -1 - delay] - sensor_act_vertical) / tau_sens_v * dt + noise
                sensor_act_horizontal += (y_h_arr[:, -1 - delay] - sensor_act_horizontal) / tau_sens_h * dt + noise
            else:
                sensor_act_vertical += (y - sensor_act_vertical) / tau_sens_v * dt + noise
                sensor_act_horizontal += (y_h - sensor_act_horizontal) / tau_sens_h * dt + noise

        cutoff_min_max = False
        if cutoff_min_max:
            sensor_act_vertical = np.where(sensor_act_vertical > 0.95, 1, sensor_act_vertical)
            sensor_act_vertical = np.where(sensor_act_vertical < 0.05, 0, sensor_act_vertical)
            sensor_act_horizontal = np.where(sensor_act_horizontal > 0.95, 1, sensor_act_horizontal)
            sensor_act_horizontal = np.where(sensor_act_horizontal < 0.05, 0, sensor_act_horizontal)
            
        cutoff_max = False
        if cutoff_max:
            #sensor_act_horizontal = np.where(sensor_act_horizontal > 0.8, 0.8, sensor_act_horizontal)
            sensor_act_horizontal = np.where(sensor_act_horizontal < 0.3, 0.3, sensor_act_horizontal)
            
            #sensor_act_vertical = np.where(sensor_act_vertical > 0.8, 0.8, sensor_act_vertical)
            #sensor_act_vertical = np.where(sensor_act_vertical < 0.2, 0.2, sensor_act_vertical)

        if n > int(len(t_span) * (1 - sensor_time)):
            sens_tar_v_min = 0
            sens_tar_v_max = 1.
            sens_tar_h_min = 0
            sens_tar_h_max = 1.

            #ybar = 0.4 + delta_y*0.9 - 0.36
        else:
            sens_tar_v_min = 0
            sens_tar_v_max = 1
            sens_tar_h_min = 0
            sens_tar_h_max = 1

        sensor_norm_vertical = (sensor_act_vertical - sens_tar_v_min) / (sens_tar_v_max - sens_tar_v_min)
        sensor_norm_horizontal = (sensor_act_horizontal - sens_tar_h_min) / (sens_tar_h_max - sens_tar_h_min)

        # #constant_time_delay = True
        # if constant_time_delay:
        #     sens_v_min = 0
        #     sens_v_max = 1
        #     sens_h_min = 0
        #     sens_h_max = 1
        #     if n > 5:
        #         if n > int(len(t_span) * (1 - sensor_time)):
        #             limit = 0.19
        #             dh_v = np.full_like(sensor_act_vertical, 1).astype(int)
        #             dh_h = np.full_like(sensor_act_horizontal, 1).astype(int)
        #             dh_pow = 1
        #             limit_high = False
        #             if limit_high:
        #                 dh_v[sensor_act_vertical > (1 - limit)] = dh_pow
        #                 dh_h[sensor_act_horizontal > (1 - limit)] = dh_pow
        #             else:
        #                 dh_v[sensor_act_vertical < limit] = dh_pow
        #                 dh_h[sensor_act_horizontal < limit] = dh_pow
        #             for i in range(len(sensor_act_vertical)):
        #                 sensor_act_vertical[i] = (1 - y_arr[i, -1 - dh_v[i]]) * sens_v_min \
        #                                          + y_arr[i, -1 - dh_v[i]] * sens_v_max + noise
        #                 sensor_act_horizontal[i] = (1 - y_h_arr[i, -1 - dh_h[i]]) * sens_h_min \
        #                                            + y_h_arr[i, -1 - dh_h[i]] * sens_h_max + noise
        #             if limit_high:
        #                 sensor_act_vertical[sensor_act_vertical > 1 - limit] = 1 + limit + noise
        #                 sensor_act_horizontal[sensor_act_horizontal > 1 - limit] = 1 + limit + noise
        #             else:
        #                 sensor_act_vertical[sensor_act_vertical < limit] = -limit + noise
        #                 sensor_act_horizontal[sensor_act_horizontal < limit] = -limit + noise
        #         else:
        #             dh_ret = 1
        #             sensor_act_vertical = y_arr[:, -1 - dh_ret] + noise
        #             sensor_act_horizontal = y_h_arr[:, -1 - dh_ret] + noise
        #     else:
        #         sensor_act_vertical = y
        #         sensor_act_horizontal = y_h
        # else:
        #     if n > int(len(t_span) * (1 - sensor_time)):
        #         tau_s_h = tau_sens + 0.5 * (1 - deepcopy(y_h))
        #         tau_s_v = tau_sens + 0.5 * (1 - deepcopy(y))
        #     else:
        #         tau_s_h = tau_sens
        #         tau_s_v = tau_sens
        #
        #     # continuously updating
        #     sensor_act_vertical += (y - sensor_act_vertical) / tau_s_v * dt + noise
        #     sensor_act_horizontal += (y_h - sensor_act_horizontal) / tau_s_h * dt + noise

        # sensor_act_vertical = np.where(sensor_act_vertical > 1, 1, sensor_act_vertical)
        # sensor_act_vertical = np.where(sensor_act_vertical < 0, 0, sensor_act_vertical)
        # bin_edges = np.arange(0, 1 + 1/25, 1/25)
        # bin_indices = np.digitize(sensor_act_vertical, bin_edges)
        # sensor_act_vertical = bin_edges[bin_indices-1]
        # print(sensor_act_vertical, sensor_act_vertical_new)

        # horizontal closed loop variable
        # b_sens = (1 - sensor_norm_horizontal) * b_min + sensor_norm_horizontal * b_max
        b_sens = b_bar + (1) * ((1 - sensor_norm_horizontal) * (b_min - b_bar) + sensor_norm_horizontal * (b_max - b_bar))

        # result arrays
        dt_arr = np.append(dt_arr, step_size)
        x_arr = np.column_stack((x_arr, x))
        b_arr = np.column_stack((b_arr, b))
        b_dot_arr = np.column_stack((b_dot_arr, b_dot))
        y_arr = np.column_stack((y_arr, y))
        y_h_arr = np.column_stack((y_h_arr, y_h))
        delta_y_arr = np.column_stack((delta_y_arr, delta_y))
        sensor_norm_vertical_arr = np.column_stack((sensor_norm_vertical_arr, sensor_norm_vertical))
        sensor_norm_horizontal_arr = np.column_stack((sensor_norm_horizontal_arr, sensor_norm_horizontal))

        b_sens_arr = np.column_stack((b_sens_arr, b_sens))
        b_max_arr = np.column_stack((b_max_arr, b_max))
        b_min_arr = np.column_stack((b_min_arr, b_min))

    # concatenating result arrays
    xb = np.concatenate(([x_arr], [b_arr]))
    xby = np.concatenate((xb, [y_arr]))
    xbyyh = np.concatenate((xby, [y_h_arr]))
    sensors = np.concatenate(([sensor_norm_vertical_arr], [sensor_norm_horizontal_arr]))
    xbyyh_sens = np.concatenate((xbyyh, sensors))
    xbyyh_sens_bsens = np.concatenate((xbyyh_sens, [b_sens_arr]))
    #xbyyh_sens_bsens_time = np.concatenate((xbyyh_sens_bsens, [dt_arr]))
    print(xbyyh_sens_bsens[0])


    return xbyyh_sens_bsens
