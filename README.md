<h1 align="center">Open Optical Flow</h1>
<h3 align="center">A PyTorch library for optical flow estimation using neural networks</h3>

<div align='center'>

[![Build status](https://github.com/neu-vig/openoptflow/actions/workflows/package-test.yml/badge.svg)](https://github.com/neu-vig/openoptflow/actions/workflows/package-test.yml)
[![Code style](https://github.com/neu-vig/openoptflow/actions/workflows/linting.yml/badge.svg)](https://github.com/neu-vig/openoptflow/actions/workflows/linting.yml)
<!-- [![Code coverage](https://github.com/neu-vig/openoptflow/actions/workflows/codecov.yml/badge.svg)](https://github.com/neu-vig/openoptflow/actions/workflows/codecov.yml) -->

</div>

<br>

### To-do

<b>Short-term</b>

- [ ] Add more models
- [ ] Add docstrings compatible with Sphinx for documentation
- [ ] Add dataloaders for common datasets
- [ ] Integrate Optuna/HyperOpt/Ray Tune for hyperparameter tuning
- [ ] Create config parser and integrate Hydra
- [ ] Create CLI using Click
- [ ] Add Registry
- [ ] Add functionality for testing model size and inference speed

<b>Long-term</b>

- [ ] Think about creating a loss/criterion module instead of having method losses in the same file as trainer classes. Could also have a base criterion which method specific losses can inherit and modify
- [ ] Register repository on codecov.io and get token for generating code coverage reports
- [ ] Create documentation website using ReadTheDocs
- [ ] Create logo for the library
- [ ] Add support for [AutoFlow](https://autoflow-google.github.io/#code)
- [ ] Register library on Pip and Conda, and set up continuous deployment

<br>

### References

- [detectron2](https://github.com/facebookresearch/detectron2)