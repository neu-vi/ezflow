<p align="center">
    <br>
    <img src="./docs/assets/logo.png" height="60" width="60"/>
    <br>
</p>

<h1 align="center">EzFlow</h1>
<h3 align="center">A modular PyTorch library for optical flow estimation using neural networks</h3>

<div align='center'>

[![Tests](https://github.com/neu-vig/ezflow/actions/workflows/package-test.yml/badge.svg)](https://github.com/neu-vig/ezflow/actions/workflows/package-test.yml)
[![Docs](https://readthedocs.org/projects/ezflow/badge/?version=latest)](https://ezflow.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/ezflow)](https://pepy.tech/project/ezflow)

<!-- [![Code style](https://github.com/neu-vig/ezflow/actions/workflows/linting.yml/badge.svg)](https://github.com/neu-vig/ezflow/actions/workflows/linting.yml) -->
<!-- [![Code coverage](https://github.com/neu-vig/ezflow/actions/workflows/codecov.yml/badge.svg)](https://github.com/neu-vig/ezflow/actions/workflows/codecov.yml) -->

**[Documentation](https://ezflow.readthedocs.io/en/latest/)** | **[Tutorials](https://ezflow.readthedocs.io/en/latest/tutorials/index.html)**

</div>


## Installation

### From source (recommended)

```shell

git clone https://github.com/neu-vig/ezflow
cd ezflow/
python setup.py install

```

### From PyPI

```shell

pip install ezflow

```

### Models supported

- [x] [DICL](https://arxiv.org/abs/2010.14851)
- [x] [FlowNetS](https://arxiv.org/abs/1504.06852)
- [x] [FlowNetC](https://arxiv.org/abs/1504.06852)
- [x] [PWCNet](https://arxiv.org/abs/1709.02371)
- [x] [RAFT](https://arxiv.org/abs/2003.12039)
- [x] [VCN](https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html)

### Datasets supported

- [x] [AutoFlow](https://autoflow-google.github.io/)
- [x] [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
- [x] [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)
- [x] [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [x] [MPI Sintel](http://sintel.is.tue.mpg.de/)
- [x] [SceneFlow Monkaa](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [x] [SceneFlow Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [x] [SceneFlow FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [x] [SceneFlow FlyingThings3D subset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### References

- [RAFT](https://github.com/princeton-vl/RAFT)
- [DICL-Flow](https://github.com/jytime/DICL-Flow)
- [PWC-Net](https://github.com/NVlabs/PWC-Net)
- [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)
- [VCN](https://github.com/gengshan-y/VCN)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [CorrelationLayer](https://github.com/oblime/CorrelationLayer)
- [ptflow](https://github.com/hmorimitsu/ptlflow)


<br>

<footer>
<a target="_blank" href="https://icons8.com/icon/3Nj3FNnz36Id/pixels">Pixels</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>
</footer>



