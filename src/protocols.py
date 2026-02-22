# TimedArray is brian2's way of representing a time varying signal
# holds a series of values for current or other metric such that each corresponds to a successive dt within time T

import numpy as np

from brian2 import prefs

prefs.codegen.target = "numpy"

from brian2 import *


# step current: at time t_on, jump instantly to a constant current amplitude 'amp', and then at time t_off jump back to zero current
def step_current(T, dt, amp, t_on, t_off):
    """params"""
    # T: sim duration
    # dt: time step
    # amp: current amplitude during step
    # t_on: time current turns on
    # t_off: time current turns off

    # how many time steps fit in the sim
    num_steps = int(np.round((T / dt)))

    # creats a num_steps long array of zeros and uses 'amp' to attach units
    I = np.zeros(num_steps) * amp

    # creates time axis aligned with sim, holds actual times of each dt
    t = np.arange(num_steps) * dt

    # turn on the step
    # if t is in the interval, I(t)=amp
    I[(t >= t_on) & (t < t_off)] = amp

    # return the whole current whose spacing is dt, provides I(t) in the sim
    return TimedArray(I, dt=dt)


# skeleton same as step_current, introduces gaussian noise
# refer to step_current comments for explanation & repeated params
def noisy_step_current(T, dt, amp, t_on, t_off, sigma, seed=None):
    """params"""
    # sigma: strength of fluctations
    # seed: optional base for rng

    num_steps = int(np.round(T / dt))
    I = np.zeros(num_steps) * amp
    t = np.arange(num_steps) * dt

    # creates bool array with length T/dt for when current is active
    expr = (t >= t_on) & (t < t_off)
    I[expr] = amp

    # avoid work if no noise needed, aka sigma==0
    # * amp required bc sigma is a brian2 quantity
    # simulates background synaptic bombardment
    if sigma != 0 * amp:
        # random number generator
        rng = np.random.default_rng(seed)

        # .standard_normal draws samples from a gaussian dist
        # np.sum(expr) counts the num of True values in expr, aka how many time points need noise
        noise = rng.standard_normal(np.sum(expr))
        I[expr] = I[expr] + sigma * noise

    return TimedArray(I, dt=dt)


# helps determine refractory behavior
# generates two current pulses with a determined width that are delta ms apart
def two_pulse_current(T, dt, amp, t1_on, pulse_width, delta, sigma=0 * amp, seed=None):
    num_steps = int(np.round(T / dt))
    I = np.zeros(num_steps) * amp
    t = np.arange(num_steps) * dt

    t2_on = t1_on + delta

    p1 = (t >= t1_on) & (t < t1_on + pulse_width)
    p2 = (t >= t2_on) & (t < t2_on + pulse_width)

    I[p1] = amp
    I[p2] = amp

    return TimedArray(I, dt=dt)
