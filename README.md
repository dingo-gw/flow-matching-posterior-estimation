# Flow-Matching-Posterior-Estimation

This is the companion repository for the NeurIPS-2023 paper [Flow Matching for Scalable 
Simulation Based Inference](https://neurips.cc/virtual/2023/poster/72395), which 
introduces a simulation-based inference (SBI) method based on flow matching, called FMPE.
It contains code to run the experiments for an SBI benchmark and gravitational wave 
inference as a challenging real world example.

This repository builds on the [dingo](https://github.com/dingo-gw/dingo) package for the
implementation of the base methods for flow matching and the pipeline for the GW 
experiment.

![alt text](figures/fmpe.png)

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

```bibtex
@inproceedings{wildberger2023flow,
    title={Flow Matching for Scalable Simulation-Based Inference},
    author={Jonas Bernhard Wildberger and Maximilian Dax and Simon Buchholz and Stephen R Green and Jakob H. Macke and Bernhard Sch{\"o}lkopf},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=D2cS6SoYlP},
    eprint={2305.17161},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
}
```

In addition, if you use the GW code in your research, please refer to the Reference 
section of the [dingo README](https://github.com/dingo-gw/dingo#references) for 
appropriate citations.
