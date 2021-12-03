import time

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function

from ..utils import AverageMeter, endpointerror


def warmup(model, dataloader, device):
    """Performs an iteration of dataloading and model prediction to warm up CUDA device

    Parameters
    ----------
    model : torch.nn.Module
        Model to be used for prediction / inference
    dataloader : torch.utils.data.DataLoader
        Dataloader to be used for prediction / inference
    device : torch.device
        Device (CUDA / CPU) to be used for prediction / inference
    """

    inp, target = iter(dataloader).next()

    img1, img2 = inp
    img1, img2, target = (
        img1.to(device),
        img2.to(device),
        target.to(device),
    )

    _ = model(img1, img2)


def run_inference(model, dataloader, device, metric_fn, flow_scale=1.0):
    """
    Uses a model to perform inference on a dataloader and captures inference time and evaluation metric

    Parameters
    ----------
    model : torch.nn.Module
        Model to be used for prediction / inference
    dataloader : torch.utils.data.DataLoader
        Dataloader to be used for prediction / inference
    device : torch.device
        Device (CUDA / CPU) to be used for prediction / inference
    metric_fn : function
        Function to be used to calculate the evaluation metric
    flow_scale : float, optional
        Scale factor to be applied to the predicted flow

    Returns
    -------
    metric_meter : AverageMeter
        AverageMeter object containing the evaluation metric information
    avg_inference_time : float
        Average inference time

    """

    metric_meter = AverageMeter()
    times = []

    with torch.no_grad():

        for inp, target in dataloader:
            start_time = time.time()

            img1, img2 = inp
            img1, img2, target = (
                img1.to(device),
                img2.to(device),
                target.to(device),
            )

            pred = model(img1, img2)
            pred = pred * flow_scale

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

            metric = metric_fn(pred, target)
            metric_meter.update(metric.item())

    avg_inference_time = sum(times) / len(times)

    print("=" * 100)
    print(f"Average inference time: {avg_inference_time}, FPS: {1/avg_inference_time}")

    return metric_meter, avg_inference_time


def profile_inference(
    model, dataloader, device, metric_fn, profiler, flow_scale=1.0, count_params=False
):
    """
    Uses a model to perform inference on a dataloader and profiles model characteristics such as memory usage, inference time, and evaluation metric

    Parameters
    ----------
    model : torch.nn.Module
        Model to be used for prediction / inference
    dataloader : torch.utils.data.DataLoader
        Dataloader to be used for prediction / inference
    device : torch.device
        Device (CUDA / CPU) to be used for prediction / inference
    metric_fn : function
        Function to be used to calculate the evaluation metric
    profiler : torch.profiler.profile
        Profiler to be used for profiling model characteristics
    flow_scale : float, optional
        Scale factor to be applied to the predicted flow
    count_params : bool, optional
        Flag to indicate whether to count model parameters

    Returns
    -------
    metric_meter : AverageMeter
        AverageMeter object containing the evaluation metric information
    avg_inference_time : float
        Average inference time
    """

    metric_meter = AverageMeter()
    times = []

    with profile(
        activities=profiler.activites,
        record_shapes=profiler.record_shapes,
        profile_memory=profiler.profile_memory,
        schedule=profiler.schedule,
        on_trace_ready=profiler.on_trace_ready,
    ) as prof:

        with torch.no_grad():

            for inp, target in dataloader:
                start_time = time.time()

                img1, img2 = inp
                img1, img2, target = (
                    img1.to(device),
                    img2.to(device),
                    target.to(device),
                )

                with record_function(profiler.model_name):
                    pred = model(img1, img2)
                    pred = pred * flow_scale

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                prof.step()
                end_time = time.time()
                times.append(end_time - start_time)

                metric = metric_fn(pred, target)
                metric_meter.update(metric.item())

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_memory_usage", row_limit=10
        )
    )
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_memory_usage", row_limit=10
        )
    )
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_memory_usage", row_limit=10
        )
    )
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_memory_usage", row_limit=10
        )
    )

    avg_inference_time = sum(times) / len(times)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 100)
    print(f"Average inference time: {avg_inference_time}, FPS: {1/avg_inference_time}")

    if count_params:
        print(f"Number of model parameters: {n_params}")
        return metric_meter, avg_inference_time, n_params

    return metric_meter, avg_inference_time


def eval_model(
    model,
    dataloader,
    device,
    distributed=False,
    metric=None,
    profiler=None,
    flow_scale=1.0,
):
    """
    Evaluates a model on a dataloader and optionally profiles model characteristics such as memory usage, inference time, and evaluation metric

    Parameters
    ----------
    model : torch.nn.Module
        Model to be used for prediction / inference
    dataloader : torch.utils.data.DataLoader
        Dataloader to be used for prediction / inference
    device : torch.device
        Device (CUDA / CPU) to be used for prediction / inference
    distributed : bool, optional
        Flag to indicate whether to perform distributed inference on multiple GPUs
    metric : function, optional
        Function to be used to calculate the evaluation metric
    profiler : torch.profiler.profile, optional
        Profiler to be used for profiling model characteristics
    flow_scale : float, optional
        Scale factor to be applied to the predicted flow

    Returns
    -------
    float
        Average evaluation metric
    """

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

    warmup(model, dataloader, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if profiler is None:
        metric_meter, _ = run_inference(
            model, dataloader, device, metric_fn, flow_scale=flow_scale
        )
    else:
        metric_meter, _ = profile_inference(
            model, dataloader, device, metric_fn, profiler, flow_scale=flow_scale
        )

    print(f"Average evaluation metric = {metric_meter.avg}")

    return metric_meter.avg
