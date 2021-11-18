import torch.nn.functional as F


class InputPadder:
    def __init__(self, dims, scale=8, mode="sintel"):

        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // scale) + 1) * scale - self.ht) % scale
        pad_wd = (((self.wd // scale) + 1) * scale - self.wd) % scale
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):

        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]
