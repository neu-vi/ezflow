# Open Optical Flow

[![Build](https://github.com/neu-vig/openoptflow/actions/workflows/package-test.yml/badge.svg)](https://github.com/neu-vig/openoptflow/actions/workflows/package-test.yml)
[![Linting](https://github.com/neu-vig/openoptflow/actions/workflows/linting.yml/badge.svg)](https://github.com/neu-vig/openoptflow/actions/workflows/linting.yml)

A PyTorch library for optical flow estimation using neural networks.


<br>

### To-do

- [ ] Add more models
- [ ] Add docstrings compatible with Sphinx for documentation
- [ ] Add dataloaders for common datasets
- [ ] Integrate Optuna/HyperOpt/Ray Tune for hyperparameter tuning
- [ ] Create config parser and integrate Hydra
- [ ] Create CLI using Click
- [ ] Add Registry
- [ ] Think about creating a loss/criterion module instead of having method losses in the same file as trainer classes. Could also have a base criterion which method specific losses can inherit and modify
- [ ] Add support for [AutoFlow](https://autoflow-google.github.io/#code)
- [ ] Add functionality for testing model size and inference speed