### FlowNetC pre-trained checkpoints
___

[FlowNetC Model config](../../models/flownet_c.yaml)

| Training Dataset | Training Config                               | ckpts                                                                                  | Sintel (training)             | KITTI2015       |
|                  |                                               |                                                                                        | Clean             | Final     | AEPE  | F1-all  |
| ---------------- | ----------------------------------------------| -------------------------------------------------------------------------------------- | ----------------- | --------- |-------|---------|
| Chairs           | [config](./flownetc_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_chairs_step1200k.pth)        |                   |           |       |         |
| Chairs -> Things | [config](./flownetc_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_chairs_things_step1574k.pth) |                   |           |       |         |
| Kubric           | [config](./flownetc_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/flownetc_kubric_step1200k.pth)        |                   |           |       |         |

