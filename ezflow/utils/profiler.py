from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler


class Profiler:
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
