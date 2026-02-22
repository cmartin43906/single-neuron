from brian2 import prefs

prefs.codegen.target = "numpy"

import numpy as np
import matplotlib.pyplot as plt
from brian2 import ms, mV, uA, uF, mS, nA, umetre, ufarad, cm, msiemens

from src.protocols import step_current
from src.models.hh import sim_hh


def main():
    T = 60 * ms
    dt = 0.01 * ms

    t_on = 20 * ms
    t_off = 50 * ms

    area = 20000 * umetre**2
    C_m = 1 * uF / cm**2 * area
    gNa = 120 * msiemens / cm**2 * area
    gK = 36 * msiemens / cm**2 * area
    gL = 0.3 * msiemens / cm**2 * area

    ENa = 50 * mV
    EK = -77 * mV
    EL = -54.4 * mV

    # CAN MANIPULATE CURRENT
    amp = 3.0 * nA
    I = step_current(T, dt, amp, t_on, t_off)

    result = sim_hh(
        I_func=I,
        T=T,
        dt=dt,
        C_m=C_m,
        gNa=gNa,
        gK=gK,
        gL=gL,
        ENa=ENa,
        EK=EK,
        EL=EL,
        V_th=0 * mV,  # spike detection threshold (analysis-only)
        detection_ref=2 * ms,  # avoid double-counting one spike waveform
        method="exponential_euler",
    )

    # reconstruct current samples for plotting
    # step_current returns a TimedArray
    t_ms = result.t_ms
    t = t_ms * ms
    I_plot = np.zeros_like(t_ms, dtype=float)
    expr = (t >= t_on) & (t < t_off)
    I_plot[expr] = float(amp / nA)

    plt.figure(figsize=(7, 3))
    plt.plot(result.t_ms, result.V_mV)
    plt.axhline(0.0, linestyle="--", label="detect threshold (0 mV)")
    plt.xlabel("Time (ms)")
    plt.ylabel("V (mV)")
    plt.title("HH membrane voltage")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
