import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
ospath = os.path.dirname(__file__)


def print_params(dim, sets, a, y_bar, tau_x, tau_b, theta, gamma, g_min, g_r, g, g_dy, dy, gait_matrix):
    rnd = 4
    print_str = [
        'dim:  ' + str(dim),
        'sets:  ' + str(sets),
        'a:  ' + str(a),
        'g:  ' + str(g),
        'y_bar:  ' + str(np.around(y_bar, rnd)),
        'tau_x:  ' + str(tau_x),
        'tau_b:  ' + str(tau_b),
        'b_dot neg.:  ' + str(np.around(-y_bar / tau_b, rnd)),
        'b_dot pos.:  ' + str(np.around((1 - y_bar) / tau_b, rnd)),
        'gamma:  ' + str(np.around(gamma, rnd)),
        'theta:  ' + str(np.around(theta, rnd)),
        # 'g_min (= gamma - theta):  ' + str(np.around(g_min, rnd)),
        'g_min:  ' + str(np.around(g_min, rnd)),
        'g_r:  ' + str(np.around(g_r, rnd)),
        'g_dy:  ' + str(np.around(g_dy, rnd)),
        'dy:  ' + str(dy)
    ]

    ljust_val = 30
    print('\n-------------------------------------------------------------------------------')
    for i in np.arange(0, 15, 3):
        print(print_str[i].ljust(ljust_val) + print_str[i + 1].ljust(ljust_val) + print_str[i + 2].ljust(ljust_val))
    print('-------------------------------------------------------------------------------\n')
    # print('gait_matrix:\n\n' + str(gait_matrix.tolist()) + '\n\n')
    print('gait_matrix:\n\n', gait_matrix, '\n\n')


def subplot(grid_size, xypos, cols, rows, xstr, ystr, xarr, yarr, label):
    ax = plt.subplot2grid(grid_size, xypos, colspan=cols, rowspan=rows)
    ax.plot(xarr, yarr, label=label)
    ax.legend()
    ax.set_xlabel(xstr)
    ax.set_ylabel(ystr)
    ax.grid(True, which='both')


def y(a, b, x):
    return 1 / (1 + np.exp(a * (b - x)))


def plots(dim, sets, g, eul_time, eul_x, eul_b, eul_y_h, sci_time, sci_x, sci_b, gain):
    def y_eul(k):
        return y(gain, eul_b[k], eul_x[k])

    def y_sci(k):
        return y(gain, sci_b[k], sci_x[k])

    k = 0  # choosing index of variable for x-axis
    j = 0  # choosing index of variable for y-axis

    eul_y = y_eul(k)
    sci_y = y_sci(k)

    start = 0
    stop = -1

    #plt.figure(figsize=(20,28), dpi=50)
    #grid_size = (10, 4)

    plt.figure(figsize=(12,8), dpi=80)
    grid_size = (2,4)

    subplot(grid_size, (0, 0), 1, 2, '$x$', '$b$', eul_x[k][start:stop], eul_b[j][start:stop], 'eul ' + str(k))
    subplot(grid_size, (0, 1), 1, 2, '$x$', '$b$', sci_x[k][start:stop], sci_b[j][start:stop], 'sci ' + str(k))

    start = 0
    stop = -1

    subplot(grid_size, (0, 2), 2, 1, '$t$', '$y$', eul_time[start:stop], eul_y[start:stop], 'eul')
    subplot(grid_size, (1, 2), 2, 1, '$t$', '$y$', sci_time[start:stop], sci_y[start:stop], 'sci')

    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    #plt.close()

    plt.figure(figsize=(12,8), dpi=80)
    grid_size = (4, 4)

    subplot(grid_size, (0, 0), 2, 1, '$t$', '$x$', eul_time[start:stop], eul_x[k][start:stop], 'eul')
    subplot(grid_size, (1, 0), 2, 1, '$t$', '$x$', sci_time[start:stop], sci_x[k][start:stop], 'sci')

    subplot(grid_size, (0, 2), 2, 1, '$t$', '$b$', eul_time[start:stop], eul_b[k][start:stop], 'eul')
    subplot(grid_size, (1, 2), 2, 1, '$t$', '$b$', sci_time[start:stop], sci_b[k][start:stop], 'sci')

    start = 0
    stop = -1

    subplot(grid_size, (2, 0), 1, 2, '$x$', '$y$', eul_x[k][start:stop], eul_y[start:stop], 'eul')
    subplot(grid_size, (2, 1), 1, 2, '$x$', '$y$', sci_x[k][start:stop], sci_y[start:stop], 'sci')

    subplot(grid_size, (2, 2), 1, 2, '$b$', '$y$', eul_b[k][start:stop], eul_y[start:stop], 'eul')
    subplot(grid_size, (2, 3), 1, 2, '$b$', '$y$', sci_b[k][start:stop], sci_y[start:stop], 'sci')

    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    #plt.close()

    plt.figure(figsize=(12, 3), dpi=80)
    grid_size = (1, 1)
    ax30 = plt.subplot2grid(grid_size, (0, 0), colspan=4, rowspan=1)
    ax30.plot(eul_time[start:stop], eul_b[k][start:stop], label='$b$')
    ax30.plot(eul_time[start:stop], dim / sets * g * eul_y_h[k][start:stop], label='$Y$')
    ax30.legend()
    ax30.set_xlabel('$t$')
    ax30.set_ylabel('$b,Y$')

    start = 0
    stop = 250

    plt.figure(figsize=(12, 3), dpi=80)
    grid_size = (1, 1)
    ax31 = plt.subplot2grid(grid_size, (0, 0), colspan=4, rowspan=1)
    ax31.plot(eul_time[start:stop], eul_y[start:stop], label='$y$')
    ax31.plot(eul_time[start:stop], eul_y_h[k][start:stop], label='$Y$')
    ax31.legend()
    ax31.set_xlabel('$t$')
    ax31.set_ylabel('$y,Y$')

    plt.figure(figsize=(12,8), dpi=80)
    grid_size = (4, 4)

    start = 0
    stop = -1

    ax13 = plt.subplot2grid(grid_size, (0, 0), colspan=4, rowspan=1)
    ax13.plot(eul_time[start:stop], eul_x[k][start:stop], label='eul:x')
    ax13.plot(eul_time[start:stop], eul_b[k][start:stop], label='b')
    ax13.plot(eul_time[start:stop], eul_y[start:stop], label='y')
    ax13.legend()
    ax13.set_xlabel('$t$')
    ax13.set_ylabel('$x,b,y$')

    ax14 = plt.subplot2grid(grid_size, (1, 0), colspan=4, rowspan=1)
    ax14.plot(sci_time[start:stop], sci_x[k][start:stop], label='sci:x')
    ax14.plot(sci_time[start:stop], sci_b[k][start:stop], label='b')
    ax14.plot(sci_time[start:stop], sci_y[start:stop], label='y')
    ax14.legend()
    ax14.set_xlabel('$t$')
    ax14.set_ylabel('$x,b,y$')


    start = 0
    stop = -1  # int(1 / 4 * len(eul_time))

    start_sci = 0  # int(3 / 4 * len(eul_time))
    stop_sci = -1

    t_eul = eul_time[start:stop]
    t_sci = sci_time[start_sci:stop_sci]

    evenly_spaced_interval = np.linspace(0, 1, len(eul_x))
    colors = [mpl.cm.rainbow(x) for x in evenly_spaced_interval]

    ax15 = plt.subplot2grid(grid_size, (2, 0), colspan=4, rowspan=1)
    ax16 = plt.subplot2grid(grid_size, (3, 0), colspan=4, rowspan=1)
    # for i in range(len(eul_x)):
    for i, color in enumerate(colors):
        ax15.plot(t_eul, y_eul(i)[start:stop], ls='-', c=color, label='$y_' + str(i + 1) + '$')
        ax16.plot(t_sci, y_sci(i)[start_sci:stop_sci], ls='-', c=color, label='$y_' + str(i + 1) + '$')

    ax15.legend()
    ax15.set_xlabel('$t$')
    ax15.set_ylabel('$y$')

    ax16.legend()
    ax16.set_xlabel('$t$')
    ax16.set_ylabel('$y$')

    #plt.close()

    # start = 0
    # stop = int(1 / 4 * len(eul_time))
    #
    # start_sci = 0
    # stop_sci = int(1 / 4 * len(sci_time))

    # t_eul = eul_time[start:stop]
    # t_sci = sci_time[start_sci:stop_sci]

    # ax17 = plt.subplot2grid(grid_size, (10, 0), colspan=4, rowspan=1)
    # ax18 = plt.subplot2grid(grid_size, (11, 0), colspan=4, rowspan=1)
    # # for i in range(len(eul_x)):
    # for i, color in enumerate(colors):
    #     ax17.plot(t_eul, eul_b[i][start:stop] - eul_x[i][start:stop], ls='-',
    #               c=color, label='$(b-x)_' + str(i + 1) + '$')
    #     ax18.plot(t_sci, sci_b[i][start:stop] - sci_x[i][start:stop], ls='-',
    #               c=color, label='$(b-x)_' + str(i + 1) + '$')

    ## sum_bx_eul = np.sum([eul_b[i][start:stop] - eul_x[i][start:stop] for i in range(len(eul_x))], axis=0)
    ## sum_bx_sci = np.sum([sci_b[i][start:stop] - sci_x[i][start:stop] for i in range(len(sci_x))], axis=0)
    ## ax17.plot(t_eul, sum_bx_eul, ls='-', label='$\Sigma(b-x)$')
    ## ax18.plot(t_sci, sum_bx_sci, ls='-', label='$\Sigma(b-x)$')

    # ax17.legend()
    # ax17.set_xlabel('$t$')
    # ax17.set_ylabel('$b-x$')
    #
    # ax18.legend()
    # ax18.set_xlabel('$t$')
    # ax18.set_ylabel('$b-x$')

    # left = 0.125  # the left side of the subplots of the figure
    # right = 0.9  # the right side of the subplots of the figure
    # bottom = 0.2  # the bottom of the subplots of the figure
    # top = 0.9  # the top of the subplots of the figure
    # wspace = 0.3  # the amount of width reserved for space between subplots,
    # # expressed as a fraction of the average axis width
    # hspace = 0.5  # the amount of height reserved for space between subplots,
    # # expressed as a fraction of the average axis height
    #
    # plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    plt.subplots_adjust(hspace=0.7, wspace=0.4)

    plt.show()
    plt.close()


# def save_as_hexy_file(stepsize, membrane, bias, gain):
#     neuron = 1 / (1 + np.exp(gain * (bias - membrane)))
#
#     # time_concat = np.concatenate(([time], [np.diff(time, axis=0, prepend=0)]))
#     time_concat = np.concatenate(([[i for i in range(len(membrane[0]))]], [stepsize * 1000 for i in range(len(membrane[0]))]))
#     # first_concat = np.concatenate((time_concat, neuron))
#     first_concat = np.concatenate((time_concat, bias * 1000))
#     second_concat = np.concatenate((first_concat, membrane * 1000))
#
#     filename = 'test_data_file.csv'
#     np.savetxt(os.path.join(ospath, '../data/' + filename), np.column_stack(second_concat),
#                delimiter=',', fmt='%i', header=',0,1,2,3,4,5,6,7,8,9,10,11,12', comments='')
