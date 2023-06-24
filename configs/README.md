### Results and Pre-trained checkpoints
___

##### FlowNetC | [arXiv](https://arxiv.org/abs/1504.06852)

[model config](./models/flownet_c.yaml) 

| Training Dataset | Training Config                                                 | ckpts                                                                                  | Sintel Clean (training) | Sintel Final(training)| KITTI2015 AEPE | KITTI2015 F1-all |
|------------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------------|-------------------------|-----------------------|----------------|------------------|
| Chairs           | [config](./trainers/flownetc/flownetc_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_chairs_step1200k.pth)        | 3.41                    | 4.94                  | 14.84          | 54.23%           |
| Chairs -> Things | [config](./trainers/flownetc/flownetc_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_chairs_things_step1574k.pth) | 2.93                    | 4.48                  | 12.47          | 45.89%           |
| Kubric           | [config](./trainers/flownetc/flownetc_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_kubric_step1200k.pth)        | 3.57                    | 3.96                  | 12.11          | 36.35%           |

___

##### PWC-Net | [arXiv](https://arxiv.org/abs/1709.02371)

[model config](./models/pwcnet.yaml) 

| Training Dataset | Training Config                                             | ckpts                                                                               | Sintel Clean (training) | Sintel Final(training)| KITTI2015 AEPE | KITTI2015 F1-all |
|------------------|-------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------|-----------------------|----------------|------------------|
| Chairs           | [config](./trainers/pwcnet/pwcnet_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_chairs_step1200k.pth)       | 3.5                     | 4.73                  | 17.81          | 51.76%           |
| Chairs -> Things | [config](./trainers/pwcnet/pwcnet_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_chairs_things_step2400k.pth)| 2.06                    | 3.43                  | 11.04          | 32.68%           |
| Kubric           | [config](./trainers/pwcnet/pwcnet_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_kubric_step1200k.pth)       | 3.08                    | 3.31                  | 9.83           | 21.94%           |


___

##### RAFT | [arXiv](https://arxiv.org/abs/2003.12039)

[model config](./models/raft.yaml) 

| Training Dataset | Training Config                                         | ckpts                                                                             | Sintel Clean (training) | Sintel Final(training)| KITTI2015 AEPE | KITTI2015 F1-all |
|------------------|---------------------------------------------------------|-----------------------------------------------------------------------------------|-------------------------|-----------------------|----------------|------------------|
| Chairs           | [config](./trainers/raft/raft_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/raft_chairs_step100k.pth)        | 2.23                    | 4.56                  | 10.45          | 38.93%           |
| Chairs -> Things | [config](./trainers/raft/raft_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/raft_chairs_things_step200k.pth) | 1.66                    | 2.75                  | 5.01           | 16.87%           |
| Kubric           | [config](./trainers/raft/raft_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/raft_kubric_step100k.pth)        | 2.12                    | 2.54                  | 6.01           | 17.35%           |

___

#### Additional Information

- KITTI dataset has been evaluated with a center crop of size `1224 x 370`.
- FlowNetC and PWC-Net uses `padding` of size `64` for evaluating the KITTI2015 dataset.
- RAFT uses `padding` of size `8` for evaluating the Sintel and KITTI2015 datasets.