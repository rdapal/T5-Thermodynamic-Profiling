from src.config.util.base_config import _BaseConfig, _Arg

class TrainerConfig(_BaseConfig):
    def __init__(self):
        super().__init__()
        self._arg_profile_phase = _Arg(
            type=str, 
            default="all", 
            help="Isolate profiling to a single phase to reduce CUDA sync overhead. Options: all, data_transfer, forward, backward, optimizer"
        )
