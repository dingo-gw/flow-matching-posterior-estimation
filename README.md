# Flow-Matching-Posterior-Estimation

This is the companion repository for the paper "Flow Matching for Scalable Simulation Based Inference" https://neurips.cc/virtual/2023/poster/72395

The code is implemented as part of the `dingo-gw` package https://github.com/dingo-gw/dingo. To access the code, please clone this repository and checkout the branch "FMPE". The scripts used to produce the results from the paper are located in this repository.

## Commands

### Benchmark 

For the SBI benchmark we have made the training and evaluation scripts available. These can be found under `sbi-benchmark/run_sbibm.py` and `sbi-benchmark/evaluate_sbibm.py` together with an example settings file `sbi-benchmark/settings.yaml`. 

### Gravitational Waves

To reproduce our gravitational wave results, you need to follow these steps usign the settings files located in `gravitational-waves/`:
1. Create a waveform dataset
2. Train the network 


Please also refer to the documentation: https://dingo-gw.readthedocs.io.
