from brian2 import prefs

prefs.codegen.target = "numpy"

import time
import numpy as np
import matplotlib.pyplot as plt
from brian2 import ms, mV, nA, Mohm, second, uF, nA, umetre, cm, msiemens

from src.protocols import step_current, noisy_step_current, two_pulse_current
from src.models.lif import sim_lif
from src.models.hh import sim_hh

# run command in terminal as python -m src.run_experiments

# helper functions


# returns spike times that occured within the stimulation window
def window_spikes(result, t_on, t_off):
    t_on_ms = float(t_on / ms)
    t_off_ms = float(t_off / ms)
    spikes = np.asarray(result.spike_times_ms, dtype=float)
    return spikes[(spikes >= t_on_ms) & (spikes < t_off_ms)]


# we want to measure variability of inter-spike intervals (ISIs)
# we do so using the Coefficient of Variation
# returns NAN if not enough spikes to compute ISIs
# ISI_k = t_k+1 - t_k
# 0 means periodic firing
# ~0.1 means very regular
# ~0.5 means irregular
# ~1 means very noisy
def cv_isi(spike_times):

    # if not enough spikes, not enough intervals
    if len(spike_times) < 3:
        return np.nan

    # converts the spike time array into the differences between the spikes
    isis = np.diff(spike_times)  # in ms

    # takes the average of the intervals
    mean_isi = np.mean(isis)

    # only return if valid
    if mean_isi <= 0:
        return np.nan

    # CV = std(ISI) / mean(ISI)
    return float(np.std(isis) / mean_isi)


def fI_curve(model_func, model_name, model_params, T, dt, t_on, t_off, currents):

    print("Running f-I curve modeling...")
    firing_rates = []

    # sweep
    for amp in currents:
        I = step_current(T, dt, amp, t_on, t_off)

        start = time.perf_counter()
        result = model_func(
            I_func=I, T=T, dt=dt, **model_params  # unloads model parameters
        )
        end = time.perf_counter()
        model_time = end - start
        print(f"{model_name} runtime: {model_time:.4f} seconds")
        # calculate stimulation duration
        cur_duration = float((t_off - t_on) / second)
        # uses helper function to store number of spikes that occured during the stimulation window
        spikes_ms = window_spikes(result, t_on, t_off)
        rate = len(spikes_ms) / cur_duration  # Hz
        firing_rates.append(rate)

    firing_rates = np.array(firing_rates, dtype=float)

    plt.figure(figsize=(5, 4))
    plt.plot(currents / nA, firing_rates, marker="o")
    plt.xlabel("Input current (nA)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"{model_name} f–I Curve")
    plt.tight_layout()
    plt.show()

    return currents, firing_rates


# test model robusteness to noise
def spike_variability_run(
    model_func,
    model_name,
    model_params,
    T,
    dt,
    t_on,
    t_off,
    amp,
    sigma,
    n_trials=5,
):

    print("Running spike variability experiments...")
    cvs = []
    spike_counts = []

    # test a range of noisy inputs that are reproducible given the seed
    for seed in range(n_trials):
        I = noisy_step_current(T, dt, amp, t_on, t_off, sigma, seed=seed)

        result = model_func(I_func=I, T=T, dt=dt, **model_params)

        # retrieve spike times of spikes occuring in the window
        spikes_ms = window_spikes(result, t_on, t_off)
        spike_counts.append(len(spikes_ms))
        # use helper function to calculate coefficient of variability
        cvs.append(cv_isi(spikes_ms))

    # convert from python list to numpy array
    cvs = np.array(cvs, dtype=float)
    spike_counts = np.array(spike_counts, dtype=int)

    # plot cv distribution
    plt.figure(figsize=(5, 4))
    plt.hist(cvs[~np.isnan(cvs)], bins=10)
    plt.xlabel("CV of ISI")
    plt.ylabel("# of Trials")
    plt.title(
        f"{model_name} spike timing variability\n"
        f"amp={float(amp/nA):.2f} nA, sigma={float(sigma/nA):.2f} nA"
    )
    plt.tight_layout()
    plt.show()

    print(
        f"[{model_name}] mean CV={np.nanmean(cvs):.3f}, "
        f"median spikes/trial={int(np.median(spike_counts))}"
    )

    return cvs, spike_counts

def refractory_run(
    model_func,
    model_name,
    model_params,
    T,
    dt,
    amp,
    t1_on,
    pulse_width,
    deltas,
    detect_t=None,
):
    print("Running refractory behavior experiments...")
    if detect_t is None:
        detect_t = pulse_width + 20 * ms  # give it a bit of time

    # contains num spikes that occured during each delta
    p2_spike = []

    # provides a number of currents where the spikes vary in distance from each other, to test when the second spike will or will not occur based on refractory behavior of the model
    for d in deltas:
        I = two_pulse_current(T, dt, amp, t1_on, pulse_width, d)

        result = model_func(I_func=I, T=T, dt=dt, **model_params)

        t2_on = t1_on + pulse_width + d
        t2_off = t2_on + detect_t

        # count spikes after pulse 2
        spikes2 = window_spikes(result, t2_on, t2_off)
        p2_spike.append(len(spikes2))
    p2_spike = np.array(p2_spike, dtype=int)

    plt.figure(figsize=(5, 4))
    plt.plot(deltas / ms, p2_spike, marker="o")
    plt.xlabel("Inter-pulse interval Δ (ms)")
    plt.ylabel("# of Spikes produced in pulse window")
    plt.title(f"{model_name} refractory recovery")
    plt.ylim(-0.1, 3)
    plt.tight_layout()
    plt.show()

    return deltas, p2_spike


def main():
    print("Running single-neuron experiments...")

    # simulation parameters
    T = 500 * ms  # total sim time
    dt = 0.1 * ms  # time step
    t_on = 100 * ms
    t_off = 400 * ms

    t1_on = 20 * ms

    # delays for refractory runs, can change
    deltas_hh = np.arange(1, 15, 1) * ms
    deltas_lif = np.arange(1, 15, 1) * ms

    # generates 12 input levels from 0nA to 0.6nA
    currents_hh = np.linspace(0.0, 6.0, 12) * nA
    currents_lif = np.linspace(0.0, 0.6, 12) * nA

    # CAN MANIPULATE
    amp_lif = 0.8 * nA  # 0.3 is good for variability test
    amp_hh = 3.0 * nA
    sigma_lif = amp_lif * 0.1
    sigma_hh = amp_hh * 0.1
    pulse_width_lif = 3 * ms
    pulse_width_hh = 2 * ms

    lif_params = {
        "tau_m": 20 * ms,
        "R": 100 * Mohm,
        "V_rest": -65 * mV,
        "V_th": -50 * mV,
        "V_reset": -65 * mV,
        "t_ref": 5 * ms,
    }

    area = 20000 * umetre**2
    hh_params = {
        "C_m": 1 * uF / cm**2 * area,
        "gNa": 120 * msiemens / cm**2 * area,
        "gK": 36 * msiemens / cm**2 * area,
        "gL": 0.3 * msiemens / cm**2 * area,
        "ENa": 50 * mV,
        "EK": -77 * mV,
        "EL": -54.4 * mV,
    }

    # CHANGE EACH VALUE TO SWITCH EXPERIMENTAL MODEL
    # model_name = "HH"
    # model_func = sim_hh
    # model_params = hh_params
    # currents = currents_hh
    # amp = amp_hh
    # sigma = sigma_hh
    # pulse_width = pulse_width_hh
    # deltas = deltas_hh

    model_name = "LIF"
    model_func = sim_lif
    model_params = lif_params
    currents = currents_lif
    amp = amp_lif
    sigma = sigma_lif
    pulse_width = pulse_width_lif
    deltas = deltas_lif

    fI_curve(
        model_func=model_func,
        model_name=model_name,
        model_params=model_params,
        T=T,
        dt=dt,
        t_on=t_on,
        t_off=t_off,
        currents=currents,
    )

    # # may take a while for the HH model
    spike_variability_run(
        model_func=model_func,
        model_name=model_name,
        model_params=model_params,
        T=T,
        dt=dt,
        t_on=t_on,
        t_off=t_off,
        amp=amp,
        sigma=sigma,
        n_trials=25,
    )

    refractory_run(
        model_func=model_func,
        model_name=model_name,
        model_params=model_params,
        T=T,
        dt=dt,
        amp=amp,
        t1_on=t1_on,
        pulse_width=pulse_width,
        deltas=deltas,
        detect_t=10 * ms,
    )


if __name__ == "__main__":
    main()
