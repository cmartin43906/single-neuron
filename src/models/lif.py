import numpy as np

from brian2 import prefs

prefs.codegen.target = "numpy"

from brian2 import *
from src.results import SimResult


def sim_lif(I_func, T, dt, tau_m, R, V_rest, V_th, V_reset, t_ref) -> SimResult:
    # run one protocol on one LIF neuron
    # take an input current and params and return a SimResult

    """params"""
    # I_func: input current as function of time
    # T: sim duration
    # dt: sim time step
    # tau_m: membrane time constant
    # R: membrane resistance
    # V_rest: resting potential
    # V_th: spike threshold
    # V_reset: reset voltage after spike
    # t_ref: absolute refractory period

    # reset state, don't stack models
    start_scope()
    # set the integration timestep, delta t
    # all differential equation in simulation advance by dt
    # default dt is 0.1ms
    defaultclock.dt = dt

    # V = IR
    # divide by tau to isolate the derivative from the og eq.
    # volt is the unit of the state variable, V
    # derivative units are inferred
    eqs = """
    dV/dt = (-(V - V_rest) + R*I(t)) / tau_m : volt
    """

    # population of one neuron
    G = NeuronGroup(
        1,
        model=eqs,  # uses above ODE
        threshold="V > V_th",  # spike rule
        reset="V = V_reset",  # instant jump after spike
        refractory=t_ref,  # absolute refractory period
        method="euler",
    )

    # initial condition
    G.V = V_rest
    # wire the input current into the ODE
    # I_func is a brian2 TimedArray
    I = I_func

    # monitors
    # statemonitor samples continuous variables (V) at each dt
    # spikemonitor records event times
    V_monitor = StateMonitor(G, "V", record=True)
    spike_monitor = SpikeMonitor(G)

    run(T)

    # np.asarray standardizes arrays for analysis
    # division by units is a brian2 trick that says 'convert this into units of ___ and return the dimensionless numbers via operator overloading
    # retrieve brian's recorded time axis
    t_ms = np.asarray(V_monitor.t / ms)
    # retrieve brian's recorded voltage traces for the first neuron
    V_mV = np.asarray(V_monitor.V[0] / mV)
    # retrieve spike times from spike monitor
    spike_times_ms = np.asarray(spike_monitor.t / ms)

    # all numbers are normalized and stripped of units and cast to python floats for standardization
    # ex: T is brian2 quantity like 500*ms
    # T/ms -> 500
    metadata = {
        "model": "LIF",
        "T_ms": float(T / ms),
        "dt_ms": float(dt / ms),
        "tau_m_ms": float(tau_m / ms),
        "R_Mohm": float(R / Mohm),  # megaOhm
        "V_rest_mV": float(V_rest / mV),
        "V_th_mV": float(V_th / mV),
        "V_reset_mV": float(V_reset / mV),
        "t_ref_ms": float(t_ref / ms),
        "method": "euler",
    }

    # probably good to avoid a positional call here
    return SimResult(
        t_ms=t_ms,
        V_mV=V_mV,
        spike_times_ms=spike_times_ms,
        metadata=metadata,
    )
