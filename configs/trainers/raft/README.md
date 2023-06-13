### RAFT pre-trained checkpoints
___

[RAFT Model config](../../models/raft.yaml)

| Training Dataset | Training Config                           | ckpts                                                                             | Sintel (training)             | KITTI2015       |
|                  |                                           |                                                                                   | Clean             | Final     | AEPE  | F1-all  |
| ---------------- | ------------------------------------------| --------------------------------------------------------------------------------- | ----------------- | --------- |-------|---------|
| Chairs           | [config](./raft_chairs_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/raft_chairs_step100k.pth)        |                   |           |       |         |
| Chairs -> Things | [config](./raft_things_baseline.yaml)     | [download](https://jianghz.me/files/ezflow_ckpts/raft_chairs_things_step200k.pth) |                   |           |       |         |
| Kubric           | [config](./raft_kubric_improved_aug.yaml) | [download](https://jianghz.me/files/ezflow_ckpts/raft_kubric_step100k.pth)        |                   |           |       |         |