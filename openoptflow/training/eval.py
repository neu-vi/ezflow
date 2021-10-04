import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import AverageMeter
from .metrics import endpointerror


def eval_model(model, dataloader, device, distributed=False, metric=None):

    if isinstance(device, list) or isinstance(device, tuple):
        device = ",".join(map(str, device))

    if device == "-1" or device == -1 or device == "cpu":
        device = torch.device("cpu")
        print("Running on CPU\n")

    elif not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA device(s) not available. Running on CPU\n")

    else:
        if device == "all":
            device = torch.device("cuda")
            if distributed:
                model = DDP(model)
            else:
                model = DataParallel(model)
            print(f"Running on all available CUDA devices\n")

        else:
            if type(device) != str:
                device = str(device)

            device_ids = device.split(",")
            device_ids = [int(id) for id in device_ids]
            cuda_str = "cuda:" + device
            device = torch.device(cuda_str)
            if distributed:
                model = DDP(model)
            else:
                model = DataParallel(model, device_ids=device_ids)
            print(f"Running on CUDA devices {device_ids}\n")

    model = model.to(device)
    model.eval()

    metric_fn = metric or endpointerror
    metric_meter = AverageMeter()

    with torch.no_grad():
        for inp, target in dataloader:

            img1, img2 = inp
            img1, img2, target = (
                img1.to(device),
                img2.to(device),
                target.to(device),
            )

            pred = model(img1, img2)

            metric = metric_fn(pred, target)
            metric_meter.update(metric.item())

    print(f"Average evaluation metric = {metric_meter.avg}")

    return metric_meter.avg
