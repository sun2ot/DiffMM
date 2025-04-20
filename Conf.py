from dataclasses import dataclass
import toml

"""
*The configurations in the @dataclass are initialized with default values.
*The `load_config` function loads a TOML configuration file and updates the dataclass instances with the values from the file.
"""

@dataclass
class BaseConfig:
    latdim: int = 64
    topk: int = 20
    gpu: str = "0"
    seed: int = 421
    denoise_dim: str = "[1000]"
    d_emb_size: int = 10
    trans: int = 0
    cl_method: int = 0

@dataclass
class DataConfig:
    name: str = "tiktok"
    # The following configurations will be updated in `DataHandler/LoadData()`
    user_num: int = 0
    item_num: int = 0
    image_feat_dim: int = 0
    text_feat_dim: int = 0
    audio_feat_dim: int = 0

@dataclass
class HyperConfig:
    modal_cl_temp: float = 0.5
    modal_cl_rate: float = 0.01
    cross_cl_temp: float = 0.2
    cross_cl_rate: float = 0.2
    keepRate: float = 0.5
    noise_scale: float = 0.1
    noise_min: float = 0.0001
    noise_max: float = 0.02
    steps: int = 5

    e_loss: float = 0.1
    residual_weight: float = 0.5
    modal_adj_weight: float = 0.2

    sampling_steps: int = 0

@dataclass
class TrainConfig:
    lr: float = 0.001
    batch: int = 1024
    test_batch: int = 256
    reg: float = 1e-5
    epoch: int = 50
    tstEpoch: int = 1
    gnn_layer: int = 1
    norm: bool = False
    sampling_noise: bool = False
    use_lr_scheduler: bool = True

@dataclass
class Config:
    base: BaseConfig = BaseConfig()
    data: DataConfig = DataConfig()
    hyper: HyperConfig = HyperConfig()
    train: TrainConfig = TrainConfig()


def load_config(path: str) -> Config:
    with open(path, 'r') as file:
        raw_config = toml.load(file)
    return Config(
        base = BaseConfig(**raw_config.get("base", {})),
        data = DataConfig(**raw_config.get("data", {})),
        hyper = HyperConfig(**raw_config.get("hyper", {})),
        train = TrainConfig(**raw_config.get("train", {})),
    )
