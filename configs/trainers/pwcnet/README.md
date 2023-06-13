### PWC-Net pre-trained checkpoints
___

[PWC-Net Model config](../../models/pwcnet.yaml)

| Training Dataset | Training Config                             | ckpts                                                                                | Sintel (training)             | KITTI2015       |
|                  |                                             |                                                                                      | Clean             | Final     | AEPE  | F1-all  |
| ---------------- | --------------------------------------------| ------------------------------------------------------------------------------------ | ----------------- | --------- |-------|---------|
| Chairs           | [config](./pwcnet_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_chairs_step1200k.pth)        |                   |           |       |         |
| Chairs -> Things | [config](./pwcnet_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_chairs_things_step2400k.pth) |                   |           |       |         |
| Kubric           | [config](./pwcnet_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/pwcnet_kubric_step1200k.pth)           |                   |           |       |         |