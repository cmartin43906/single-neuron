# we want LIF and HH models to return the same kind of object to ensure that protocols can be resued and analysis doesn't care which model produced the data

from dataclasses import dataclass
import numpy as np


@dataclass
class SimResult:
    t_ms: np.ndarray  # time axis
    V_mV: np.ndarray  # voltage trace at each time t_ms

    # when spikes occured
    # store separately so analysis doesn't have to re-detect spikes every time
    spike_times_ms: np.ndarray

    # holds data like runtime, input type, etc
    metadata: dict

    # returns num spikes generated during sim run
    @property
    def num_spikes(self) -> int:
        return len(self.spike_times_ms)
