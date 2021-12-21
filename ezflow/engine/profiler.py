from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler


class Profiler:
    """
    This class is a wrapper to initialize the parameters of PyTorch profiler.
    An instance of this class can be passed as an argument to ezflow.engine.eval_model
    to enable profiling of the model during inference.

    `Official documentation on torch.profiler <https://pytorch.org/docs/stable/profiler.html?highlight=profiler#module-torch.profiler>`_

    Parameters
    ----------
    model_name : str
        Name of the model
    log_dir : str
        Path to save the profiling logs
    profile_cpu : bool, optional
        Enable CPU profiling, by default False
    profile_cuda : bool, optional
        Enable CUDA profiling, by default False
    profile_memory : bool, optional
        Enable memory profiling, by default False
    record_shapes : bool, optional
        Enable shape recording for tensors, by default False
    skip_first : int, optional
        Number of warmup iterations to skip, by default 0
    wait : int, optional
        Number of seconds to wait before starting the profiler, by default 0
    warmup : int, optional
        Number of iterations to warmup the profiler, by default 1
    active : int, optional
        Number of iterations to profile, by default 1
    repeat : int, optional
        Number of times to repeat the profiling, by default 10
    """

    def __init__(
        self,
        model_name,
        log_dir,
        profile_cpu=False,
        profile_cuda=False,
        profile_memory=False,
        record_shapes=False,
        skip_first=0,
        wait=0,
        warmup=1,
        active=1,
        repeat=10,
    ):

        assert warmup != 0, "warmup cannot be 0, this can skew profiler results"
        assert (
            log_dir is not None
        ), "log_dir path is not provided to save profiling logs"

        self.activites = []
        self.model_name = model_name.upper()
        if profile_cpu:
            self.activites.append(ProfilerActivity.CPU)

        if profile_cuda:
            self.activites.append(ProfilerActivity.CUDA)

        self.profile_memory = profile_memory
        self.record_shapes = record_shapes

        self.schedule = schedule(
            skip_first=skip_first,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )

        self.on_trace_ready = tensorboard_trace_handler(log_dir)
