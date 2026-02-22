CSCE 790 – Assignment 1
Catherine Martin

To run:
1. Create a Python 3.11 environment
2. pip install -r requirements.txt
3. Run:
   python -m src.run_experiments
   or
   python -m src.test_hh
   or
   python -m src.test_lif

/src contains experimental code and structure, while src/models contains LIF and HH implementations.

To switch the model that is being used in run_experiments.py, open this file and uncomment the appropriate block of parameters under the heading # CHANGE EACH VALUE TO SWITCH EXPERIMENTAL MODEL. Do not forget to comment out the unused paramters.

Current parameter 'amp' can be manually manipulated in run_experiments.py, test_hh.py, and test_lif.py to simulate subthreshold and suprathreshold current application, and is marked by comment # CAN MANIPULATE CURRENT. Noisy current is utilized specifically in the spike_variability_run experiment.

run_experiments.py can run experiments to generate f-I curves and measure spike-variability using CV values. A specific test for refractory behavior is under development, but refractory behavior can be inferred from f-I curves currently.

test_hh.py will generate the spike waveforms produced by the Hodkin-Huxley model under the parameters specified in the file, and test_lif.py will generate a plot of LIF analagous spike-events.

plt.show() will block execution until the plot it generates is closed - close the plot to cause the next experiment to run, if running a file that generates multiple successive plots.

protocols.py contains current generation code, and results.py contains the dataclass that holds the model-agnostic simulation result.