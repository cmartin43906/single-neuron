import numpy as np

import numpy as np
from brian2 import prefs

prefs.codegen.target = "numpy"

from brian2 import *
from src.results import SimResult


def sim_hh(
    I_func,
    T,
    dt,
    C_m,
    gNa,
    gK,
    gL,
    ENa,
    EK,
    EL,
    V_th=0 * mV,
    detection_ref=2 * ms,
    method="exponential_euler",
) -> SimResult:

    start_scope()
    defaultclock.dt = dt

    # Make the TimedArray visible inside the equation string as I(t)
    I_in = I_func

    # parameterization of Hodgkin-Huxley 1952 eqs
    eqs = Equations("""
        I = I_in(t) : amp
        dv/dt = (I - (gNa * m ** 3 * h * (v - ENa)) - (gK * n ** 4 * (v - EK)) - (gL * (v - EL))) / C_m : volt

        dn/dt = alpha_n * (1 - n) - beta_n * n : 1
        dm/dt = alpha_m * (1 - m) - beta_m * m : 1
        dh/dt = alpha_h * (1 - h) - beta_h * h : 1

        alpha_n = (0.01 * (v / mV + 50) / (1 - exp(-(v / mV + 50) / 10))) / ms : Hz
        beta_n = (0.125 * exp(-(v / mV + 60) / 80)) / ms : Hz

        alpha_m = (0.1 * (v / mV + 35) / (1 - exp(-(v / mV + 35) / 10))) / ms : Hz
        beta_m = (4.0 * exp(-0.0556 * (v / mV + 60))) / ms : Hz

        alpha_h = (0.07 * exp(-0.05 * (v / mV + 60))) / ms : Hz
        beta_h = (1 / (1 + exp(-(0.1) * (v / mV + 30)))) / ms : Hz

        C_m : farad
        gNa : siemens
        gK : siemens
        gL : siemens
        ENa : volt
        EK : volt
        EL : volt
        # I : amp
        """)

    G = NeuronGroup(
        1,
        model=eqs,
        threshold="v > V_th",
        refractory=detection_ref,
        method=method,
    )

    # Initial conditions
    G.v = -65 * mV
    G.m = 0.05
    G.h = 0.6
    G.n = 0.32
    G.C_m = C_m
    G.gNa = gNa
    G.gK = gK
    G.gL = gL
    G.ENa = ENa
    G.EK = EK
    G.EL = EL

    V_mon = StateMonitor(G, True, record=True)
    sp_mon = SpikeMonitor(G)

    run(T, namespace={"I_in": I_in, "V_th": V_th})

    t_ms = np.asarray(V_mon.t / ms)
    V_mV = np.asarray(V_mon.v[0] / mV)
    spike_times_ms = np.asarray(sp_mon.t / ms)

    metadata = {
        "model": "HH",
        "T_ms": float(T / ms),
        "dt_ms": float(dt / ms),
        "C_m": str(C_m),
        "gNa": str(gNa),
        "gK": str(gK),
        "gL": str(gL),
        "ENa_mV": float(ENa / mV),
        "EK_mV": float(EK / mV),
        "EL_mV": float(EL / mV),
        # "V_init_mV": float(V_0 / mV),
        "V_det_mV": float(V_th / mV),
        "detect_ref_ms": float(detection_ref / ms),
        "method": method,
        "gating_form": "alpha_beta",
    }

    return SimResult(
        t_ms=t_ms,
        V_mV=V_mV,
        spike_times_ms=spike_times_ms,
        metadata=metadata,
    )
