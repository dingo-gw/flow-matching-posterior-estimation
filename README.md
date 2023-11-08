# Flow-Matching-Posterior-Estimation

This is the companion repository for the NeurIPS-2023 paper [Flow Matching for Scalable 
Simulation Based Inference](https://neurips.cc/virtual/2023/poster/72395). It contains
code to run the experiments for the SBI benchmark and gravitational wave
inference as a challenging real world example.

This repository builds on the [dingo](https://github.com/dingo-gw/dingo) package for the
implementation of the base methods for flow matching and the pipeline for the GW 
experiment.

## Set up
First, clone the [FMPE](https://github.com/dingo-gw/dingo/tree/FMPE) of the `dingo` 
repository.

```sh
git clone --branch FMPE https://github.com/dingo-gw/dingo.git
```

Next, create a new virtual environment.

```sh
python3 -m venv fmpe-venv
source fmpe-venv/bin/activate
```

Install `dingo` in this environment.

```sh
cd dingo
pip install -e ."[dev]"
```

Note that while `dingo` can in general be installed from 
[PyPI](https://pypi.org/project/dingo-gw/) with pip, this manual installation is required 
as the FMPE code is not yet contained in the main branch. 

Finally, install the [sbibm](https://github.com/sbi-benchmark/sbibm) package for the 
benchmark experiments.

```sh
pip install sbibm
```

## Experiments

### SBI Benchmark 

Training and evaluation scripts available in `./sbi-benchmark`. To train an FMPE model,
run

```sh
python run_sbibm.py --train_dir </path/to/train_dir>
```

where the training directory contains a `settings.yaml` file. Example settings can be 
found in `./sbi-benchmark/settings.yaml`.

### Gravitational Wave inference

To reproduce our gravitational wave results, you need to follow these steps usign the settings files located in `gravitational-waves/`:
1. Create a waveform dataset
2. Train the network 


Please also refer to the documentation: https://dingo-gw.readthedocs.io.


# References

