import os
import warnings
from copy import deepcopy

import numpy as np
import scipy.integrate

from gait import gait, test_gait
from model import dynamical_system_nd, \
    solve_euler_hexapod_nd_sensor, solve_ivp_euler_nd
from plot import plots, print_params, save_as_hexy_file

ospath = os.path.dirname(__file__)


class Sim:
    t_start = 0

    def __init__(self,
                 dim, sets, g, y_bar, a, tau_x, tau_b, tau_sens, y_h_max, step_size, dy, t_end, identity_inphase,
                 test, transpose, shuffle, vsensor, hsensor, noise, eul_ics_fun, sci_ics_fun, rnd_ics, transpose_time,
                 sensor_time, constant_time_delay):

        self.t_end = t_end
        # gait_matrix dimension
        self.dim = dim
        # number of main oscillators
        self.sets = sets
        self.vsensor = vsensor
        self.hsensor = hsensor
        self.noise = noise
        # corresponding gait matrix
        if identity_inphase == 'identity':
            self.gait_matrix = np.identity(self.dim)
        elif identity_inphase == 'inphase':
            self.gait_matrix = np.ones((self.dim, self.dim))
        else:
            if test:
                self.gait_matrix = test_gait()
            else:
                if transpose:
                    self.gait_matrix = np.transpose(gait(self.dim, self.sets))
                else:
                    self.gait_matrix = gait(self.dim, self.sets, shuffle)

        # parameters
        self.a = a
        self.y_bar = y_bar
        self.tau_x = tau_x
        self.tau_b = tau_b
        self.tau_sens = tau_sens
        self.y_h_max = y_h_max

        self.theta = self.tau_x / self.tau_b
        self.gamma = 1 / (self.a * (self.y_bar - self.y_bar ** 2))

        self.g_min = self.gamma - self.theta
        self.g_r = self.g_min + 2 * np.sqrt(self.theta * self.gamma)

        self.g = g

        # scaling factor 'g' as a function of neural sensitivity
        self.neural_sensitivity = dy
        self.g_dy = (self.sets - 1) / self.a * np.log(1 / self.neural_sensitivity - 1)

        # 1d fixed point
        if self.dim == 1:
            self.xb_fp = np.array([self.g * self.y_bar, self.g * self.y_bar + 1 / self.a * np.log(1 / self.y_bar - 1)])
        # nd fixed point
        else:
            self.xb_fp = np.concatenate(
                ([0 for i in range(self.dim)], [1 / self.a * np.log(1 / self.y_bar - 1) for self.i in range(self.dim)]))

        # time properties for euler integration
        self.step_size = step_size
        self.eul_t = np.arange(Sim.t_start, self.t_end + self.step_size, self.step_size)

        # time properties for scipy.integrate.solve_ivp
        t_points = len(self.eul_t)
        # time array for scipy integration
        self.sci_t = np.linspace(Sim.t_start, self.t_end, t_points)

        # options
        self.eul_ics_fun = eul_ics_fun
        self.sci_ics_fun = sci_ics_fun
        self.rnd_ics = rnd_ics
        self.transpose_time = transpose_time
        self.sensor_time = sensor_time
        self.constant_time_delay = constant_time_delay

        self.eul_dt = None
        self.eul_x = None
        self.eul_b = None
        self.eul_y = None

        self.eul_y_h = None

        self.eul_sensor_act_vertical = None
        self.eul_sensor_act_horizontal = None

        self.sci_x = None
        self.sci_b = None
        self.sci_y = None

    def init_vals(self, id, ref):
        num_ones = np.count_nonzero(self.gait_matrix[0] == 1)
        gm_cp = np.transpose(deepcopy(self.gait_matrix))[0]
        tau = 0.1
        one_arr = np.where(gm_cp == 1, 1, 0)
        epsilon = - (one_arr - self.y_bar) / self.tau_b * tau

        func_bias = self.g * num_ones * (gm_cp + 1) / 2 + epsilon
        func_membrane = self.g * num_ones * gm_cp

        func_ics = np.concatenate((func_membrane, func_bias))
        # initial values
        if (id == 'eul' and self.eul_ics_fun) or (id == 'sci' and self.sci_ics_fun) or (
                id == 'raw' and self.eul_ics_fun):
            if self.rnd_ics:
                # randn: standard normal distribution; rand: uniform distribution over [0,1)
                ics = np.concatenate((self.g * num_ones * np.random.randn(len(self.gait_matrix[0])),
                                      self.g * num_ones * np.random.rand(len(self.gait_matrix[0]))))
                print('Solver: ' + id + '; random values as ics')
            else:
                ics = func_ics
                print('Solver: ' + id + '; function values as ics')
            return ics
        else:
            try:
                filename = str(id) + str(ref)
                ics = np.load(os.path.join(ospath, '../ics/' + filename) + '.npy')
                if len(ics) != 2 * self.dim:
                    raise
                else:
                    print('Solver: ' + id + '; ics loaded: ' + filename + '.npy')
            except:
                ics = func_ics
                print('Solver: ' + id + '; no related ics file; function values as ics')
            finally:
                return ics

    def param_test(self):
        if self.theta >= self.gamma:
            warnings.warn('g_min is negative! (theta >= gamma)')

        if self.sets > self.dim:
            warnings.warn('sets > dim')

    def sol_scipy(self, ivp):
        sol = scipy.integrate.solve_ivp(dynamical_system_nd, (Sim.t_start, self.t_end), ivp, t_eval=self.sci_t,
                                        args=(self.g, self.a, self.y_bar, self.tau_x, self.tau_b, self.gait_matrix))
        return np.split(sol.y, 2)

    def sol_euler(self, ivp):
        sol = solve_ivp_euler_nd(self.eul_t, ivp, self.step_size, self.g, self.a, self.y_bar, self.tau_x,
                                 self.tau_b, self.gait_matrix)
        return np.split(sol, 2)

    def sol_euler_raw(self, ivp):
        sol = solve_euler_hexapod_nd_sensor(self.eul_t, ivp, self.step_size, self.g, self.a, self.y_bar, self.tau_x,
                                            self.tau_b, self.gait_matrix, self.tau_sens, self.y_h_max,
                                            self.transpose_time, self.sensor_time, self.vsensor, self.hsensor,
                                            self.noise, self.constant_time_delay)
        split_arr = np.split(sol[:-1], 5)
        return split_arr, sol[-1]

    def print(self):
        print_params(self.dim, self.sets, self.a, self.y_bar, self.tau_x, self.tau_b, self.theta, self.gamma,
                     self.g_min, self.g_r, self.g, self.g_dy, self.neural_sensitivity, self.gait_matrix)

    def params_str(self):
        labels = ['d', 's', 'a', 'ybar', 'taux', 'taub', 'g']
        params = np.array([self.dim, self.sets, self.a, self.y_bar, self.tau_x, self.tau_b, self.g]).round(2)
        params_it = '_ics_'
        for i in range(len(labels)):
            params_it += labels[i] + str(params[i])
        params_str = np.char.replace(params_it, '.', '-')
        return params_str

    def save_ics(self, sol, id, ref):
        filename = str(id) + str(ref)
        ics = np.concatenate(
            ([sol[0][i][-1] for i in range(self.dim)], [sol[1][i][-1] for i in range(self.dim)]))
        np.save(os.path.join(ospath, '../ics/' + filename), ics)

    def solve_sci(self, ics_save):
        id = 'sci'
        ref = Sim.params_str(self)
        ics = Sim.init_vals(self, id, ref)
        sci = Sim.sol_scipy(self, ics)
        if ics_save:
            Sim.save_ics(self, sci, id, ref)
        return sci

    def solve_eul(self, ics_save):
        id = 'eul'
        ref = Sim.params_str(self)
        ics = Sim.init_vals(self, id, ref)
        eul = Sim.sol_euler(self, ics)
        if ics_save:
            Sim.save_ics(self, eul, id, ref)
        return eul

    def solve_eul_raw(self, ics_save):
        id = 'raw'
        ref = Sim.params_str(self)
        ics = Sim.init_vals(self, id, ref)
        raw = Sim.sol_euler_raw(self, ics)
        if ics_save:
            Sim.save_ics(self, raw[0], id, ref)
        return raw

    def solve(self, print_params=True, plot=True, save=True, ics_save=True):
        Sim.param_test(self)

        eul = Sim.solve_eul_raw(self, ics_save)
        sci = Sim.solve_sci(self, ics_save)

        # Simulation results as class objects containing all values
        # Euler method
        self.eul_dt = eul[1]    # time step
        self.eul_x = eul[0][0]  # membrane potential
        self.eul_b = eul[0][1]  # neural bias
        self.eul_y = 1 / (1 + np.exp(self.a * (self.eul_b - self.eul_x)))   # neural activation

        self.eul_y_h = eul[0][2]    # horizontal neural activation

        self.eul_sensor_act_vertical = eul[0][3]    # simulated vertical sensor values
        self.eul_sensor_act_horizontal = eul[0][4]  # simulated horizontal sensor values

        # Scipy Runge-Kutta-4 method
        self.sci_x = sci[0]     # membrane potential
        self.sci_b = sci[1]     # neural bias
        self.sci_y = 1 / (1 + np.exp(self.a * (self.sci_b - self.sci_x)))   # neural activation

        if print_params:    # print parameters
            Sim.print(self)
        if plot:    # plot results
            plots(self.eul_t, self.eul_x, self.eul_b, self.sci_t, self.sci_x, self.sci_b, self.a)
        if save:    # save results in file for analysis
            # save_as_hexy_file(self.sci_t, self.sci_x, self.sci_b, self.a)
            save_as_hexy_file(self.eul_dt, self.eul_x, self.eul_b, self.a)
