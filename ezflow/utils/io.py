import re
from os.path import *

import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


def read_flow_middlebury(fn):
    """
    Read .flo file in Middlebury format

    Parameters
    -----------
    fn : str
        Absolute path to flow file

    Returns
    --------
    flow : np.ndarray
        Optical flow map
    """
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (banda, columns, rows)
            return np.resize(data, (int(h), int(w), 2))


def read_flow_pfm(file):
    """
    Read optical flow from a .pfm file

    Parameters
    -----------
    file : str
        Path to flow file

    Returns
    --------
    flow : np.ndarray
        Optical flow map
    """

    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b"PF":
        color = True
    elif header == b"Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data


def read_flow_png(filename):
    """
    Read optical flow from a png file.

    Parameters
    -----------
    filename : str
        Path to flow file

    Returns
    --------
    flow : np.ndarray
        Optical flow map

    valid : np.ndarray
        Valid flow map
    """
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    return flow, valid


def write_flow(filename, uv, v=None):
    """Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Parameters
    ----------
    filename : str
        Path to file
    uv : np.ndarray
        Optical flow
    v : np.ndarray, optional
        Optional second channel
    """

    # Original code by Deqing Sun, adapted from Daniel Scharstein.

    n_bands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    f = open(filename, "wb")
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * n_bands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def read_image(file_name):
    """
    Read images from a variety of file formats

    Parameters
    -----------
    file_name : str
        Path to flow file

    Returns
    --------
    flow : np.ndarray
        Optical flow map
    """

    ext = splitext(file_name)[-1]

    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_name)

    elif ext == ".bin" or ext == ".raw":
        return np.load(file_name)

    return []


def read_flow(file_name):
    """
    Read ground truth flow from a variety of file formats

    Parameters
    -----------
    file_name : str
        Path to flow file

    Returns
    --------
    flow : np.ndarray
        Optical flow map

    valid : None if .flo and .pfm files else np.ndarray
        Valid flow map
    """

    ext = splitext(file_name)[-1]

    if ext == ".flo":
        flow = read_flow_middlebury(file_name).astype(np.float32)
        return flow, None

    elif ext == ".pfm":

        flow = read_flow_pfm(file_name).astype(np.float32)

        if len(flow.shape) == 2:
            return flow, None
        else:
            return flow[:, :, :-1], None

    elif ext == ".png":
        return read_flow_png(file_name)

    return []


class InputPadder:
    """
    Class to pad / unpad the input to a network with a given padding

    Parameters
    -----------
    dims : tuple
        Dimensions of the input
    divisor : int
        Divisor to make the input evenly divisible by
    mode : str
        Padding mode
    """

    def __init__(self, dims, divisor=8, mode="sintel"):

        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor

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
        """
        Pad the input

        Parameters
        -----------
        inputs : list
            List of inputs to pad

        Returns
        --------
        list
            Padded inputs
        """

        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        """
        Unpad the input

        Parameters
        -----------
        x : torch.Tensor
            Input to unpad

        Returns
        --------
        torch.Tensor
            Unpadded input
        """

        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]

        return x[..., c[0] : c[1], c[2] : c[3]]
