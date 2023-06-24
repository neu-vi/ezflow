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
___

### Models supported

- [x] [DICL](https://arxiv.org/abs/2010.14851)
- [x] [FlowNetS](https://arxiv.org/abs/1504.06852)
- [x] [FlowNetC](https://arxiv.org/abs/1504.06852) ([3 checkpoints](./configs/README.md))
- [x] [PWCNet](https://arxiv.org/abs/1709.02371) ([3 checkpoints](./configs/README.md)) 
- [x] [RAFT](https://arxiv.org/abs/2003.12039) ([3 checkpoints](./configs/README.md))
- [x] [VCN](https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html)

### Datasets supported

- [x] [AutoFlow](https://autoflow-google.github.io/)
- [x] [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
- [x] [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/)
- [x] [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [x] [Kubric](https://github.com/google-research/kubric)
- [x] [MPI Sintel](http://sintel.is.tue.mpg.de/)
- [x] [SceneFlow Monkaa](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [x] [SceneFlow Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [x] [SceneFlow FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [x] [SceneFlow FlyingThings3D subset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

___

### Results and Pre-trained checkpoints


- #### FlowNetC | [model config](./configs/models/flownet_c.yaml) | [arXiv](https://arxiv.org/abs/1504.06852)

| Training Dataset | Training Config                                                         | ckpts                                                                                  | Sintel Clean (training) | Sintel Final(training)| KITTI2015 AEPE | KITTI2015 F1-all |
|------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------|-------------------------|-----------------------|----------------|------------------|
| Chairs           | [config](./configs/trainers/flownetc/flownetc_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_chairs_step1200k.pth)        | 3.41                    | 4.94                  | 14.84          | 54.23%           |
| Chairs -> Things | [config](./configs/trainers/flownetc/flownetc_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_chairs_things_step1574k.pth) | 2.93                    | 4.48                  | 12.47          | 45.89%           |
| Kubric           | [config](./configs/trainers/flownetc/flownetc_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_kubric_step1200k.pth)        | 3.57                    | 3.96                  | 12.11          | 36.35%           |

- #### PWC-Net | [model config](./configs/models/pwcnet.yaml)  | [arXiv](https://arxiv.org/abs/1709.02371)

| Training Dataset | Training Config                                                     | ckpts                                                                               | Sintel Clean (training) | Sintel Final(training)| KITTI2015 AEPE | KITTI2015 F1-all |
|------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------|-----------------------|----------------|------------------|
| Chairs           | [config](./configs/trainers/pwcnet/pwcnet_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_chairs_step1200k.pth)       | 3.5                     | 4.73                  | 17.81          | 51.76%           |
| Chairs -> Things | [config](./configs/trainers/pwcnet/pwcnet_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_chairs_things_step2400k.pth)| 2.06                    | 3.43                  | 11.04          | 32.68%           |
| Kubric           | [config](./configs/trainers/pwcnet/pwcnet_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_kubric_step1200k.pth)       | 3.08                    | 3.31                  | 9.83           | 21.94%           |


- #### RAFT | [model config](./configs/models/raft.yaml) | [arXiv](https://arxiv.org/abs/2003.12039)

| Training Dataset | Training Config                                                 | ckpts                                                                             | Sintel Clean (training) | Sintel Final(training)| KITTI2015 AEPE | KITTI2015 F1-all |
|------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------|-----------------------|----------------|------------------|
| Chairs           | [config](./configs/trainers/raft/raft_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/raft_chairs_step100k.pth)        | 2.23                    | 4.56                  | 10.45          | 38.93%           |
| Chairs -> Things | [config](./configs/trainers/raft/raft_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/raft_chairs_things_step200k.pth) | 1.66                    | 2.75                  | 5.01           | 16.87%           |
| Kubric           | [config](./configs/trainers/raft/raft_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/raft_kubric_step100k.pth)        | 2.12                    | 2.54                  | 6.01           | 17.35%           |

___

#### Additional Information

- KITTI dataset has been evaluated with a center crop of size `1224 x 370`.
- FlowNetC and PWC-Net uses `padding` of size `64` for evaluating the KITTI2015 dataset.
- RAFT uses `padding` of size `8` for evaluating the Sintel and KITTI2015 datasets.
___
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



