import matplotlib.pyplot as plt

from brian2 import prefs

prefs.codegen.target = "numpy"

from brian2 import *

from src.models.lif import sim_lif
from src.protocols import step_current

T = 500 * ms
dt = 0.1 * ms

# CAN MANIPULATE CURRENT
amp = 2 * nA
I_func = step_current(T=T, dt=dt, amp=amp, t_on=100 * ms, t_off=400 * ms)

result = sim_lif(
    I_func=I_func,
    T=T,
    dt=dt,
    tau_m=20 * ms,
    R=10 * Mohm,
    V_rest=-65 * mV,
    V_th=-50 * mV,
    V_reset=-65 * mV,
    t_ref=5 * ms,
)

print("num spikes:", result.num_spikes)

plt.plot(result.t_ms, result.V_mV)
plt.axhline(-50, linestyle="--")
plt.xlabel("Time (ms)")
plt.ylabel("V (mV)")
plt.title("LIF membrane voltage")
plt.show()
