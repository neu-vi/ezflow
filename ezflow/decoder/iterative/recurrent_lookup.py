import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import configurable
from ...modules import ConvGRU
from ..build import DECODER_REGISTRY


class FlowHead(nn.Module):
    """
    Applies two 2D convolutions over an input feature map
    to generate a flow tensor of shape N x 2 x H x W.

    Parameters
    ----------
    input_dim : int, default: 128
        Number of input dimensions.
    hidden_dim : int, default: 256
        Number of hidden dimensions.
    """

    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Performs forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape N x input_dim x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 2 x H x W
        """
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    """
    Applies two Convolution GRU cells to the input signal.
    Each GRU cell uses separate convolution layers.

    Parameters
    ----------
    hidden_dim : int, default: 128
        Number of hidden dimensions.
    input_dim : int, default: 192 + 128
        Number of hidden dimensions.
    """

    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

    def forward(self, h, x):
        """
        Performs forward pass.

        Parameters
        ----------
        h : torch.Tensor
            A tensor of shape N x hidden_dim x H x W representating the hidden state

        x : torch.Tensor
            A tensor of shape N x input_dim + hidden_dim x H x W representating the input


        Returns
        -------
        torch.Tensor
            a tensor of shape N x hidden_dim x H x W
        """
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    """
    Encodes motion features from the correlation levels of the pyramid
    and the input flow estimate using convolution layers.


    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid

    """

    def __init__(self, corr_radius, corr_levels):
        super(SmallMotionEncoder, self).__init__()

        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        """
        Parameters
        ----------
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W

        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 82 x H x W
        """
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))

        return torch.cat([out, flow], dim=1)


class MotionEncoder(nn.Module):
    """
    Encodes motion features from the correlation levels of the pyramid
    and the input flow estimate using convolution layers.


    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid

    """

    def __init__(self, corr_radius, corr_levels):
        super(MotionEncoder, self).__init__()

        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        """
        Parameters
        ----------
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W

        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 128 x H x W
        """

        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))

        return torch.cat([out, flow], dim=1)


@DECODER_REGISTRY.register()
class SmallRecurrentLookupUpdateBlock(nn.Module):
    """
    Applies an iterative lookup update on all levels of the correlation
    pyramid to estimate flow with a sequence of GRU cells.
    Used in **RAFT** (https://arxiv.org/abs/2003.12039)

    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid
    hidden_dim  : int, default: 96
        Number of hidden dimensions.
    input_dim   : int, default: 64
        Number of input dimensions.
    """

    @configurable
    def __init__(self, corr_radius, corr_levels, hidden_dim=96, input_dim=64):
        super(SmallRecurrentLookupUpdateBlock, self).__init__()

        self.encoder = SmallMotionEncoder(corr_radius, corr_levels)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + input_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    @classmethod
    def from_config(cls, cfg):
        return {
            "corr_radius": cfg.CORR_RADIUS,
            "corr_levels": cfg.CORR_LEVELS,
            "hidden_dim": cfg.HIDDEN_DIM,
            "input_dim": cfg.INPUT_DIM,
        }

    def forward(self, net, inp, corr, flow):
        """
        Performs forward pass.

        Parameters
        ----------
        net : torch.Tensor
            A tensor of shape N x hidden_dim x H x W
        inp : torch.Tensor
            A tensor of shape N x input_dim x H x W
        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W


        Returns
        -------
        net : torch.Tensor
            A tensor of shape N x hidden_dim x H x W representing the output of the GRU cell
        mask : NoneType
        delta_flow : torch.Tensor
            A tensor of shape N x 2 x H x W representing the delta flow
        """

        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


@DECODER_REGISTRY.register()
class RecurrentLookupUpdateBlock(nn.Module):
    """
    Applies an iterative lookup update on all levels of the correlation
    pyramid to estimate flow with a sequence of GRU cells.
    Used in **RAFT** (https://arxiv.org/abs/2003.12039)

    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid
    hidden_dim  : int, default: 128
        Number of hidden dimensions.
    input_dim   : int, default: 128
        Number of input dimensions.
    """

    @configurable
    def __init__(self, corr_radius, corr_levels, hidden_dim=128, input_dim=128):
        super(RecurrentLookupUpdateBlock, self).__init__()

        self.encoder = MotionEncoder(corr_radius, corr_levels)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "corr_radius": cfg.CORR_RADIUS,
            "corr_levels": cfg.CORR_LEVELS,
            "hidden_dim": cfg.HIDDEN_DIM,
            "input_dim": cfg.INPUT_DIM,
        }

    def forward(self, net, inp, corr, flow):
        """
        Performs forward pass.

        Parameters
        ----------
        net : torch.Tensor
            A tensor of shape N x hidden_dim x H x W
        inp : torch.Tensor
            A tensor of shape N x input_dim x H x W
        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W


        Returns
        -------
        net : torch.Tensor
            A tensor of shape N x hidden_dim x H x W representing the output of the SepConvGRU cell.
        mask : torch.Tensor
            A tensor of shape N x 576 x H x W
        delta_flow : torch.Tensor
            A tensor of shape N x 2 x H x W representing the delta flow
        """

        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        mask = 0.25 * self.mask(net)

        return net, mask, delta_flow
