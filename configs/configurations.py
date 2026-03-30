# dataclasses for configurations objects:

from torch._C import device
from dataclasses import dataclass, field 
from typing import Optional, Union, List, Dict, Any, Literal
from hydra.core.config_store import ConfigStore
import torch

@dataclass
class DataConfig:
    dataset: str= 'mnist'
    use_torchvision: Optional[Union[None, bool]] = True
    data_path: Optional[Union[None, str]] = None
    num_classes: Optional[Union[None, int]] = 10
    shuffle: Optional[Union[None, bool]] = False
    transform: Optional[Union[None, str]] = None
    class Config:
        version_base = "1.1"


@dataclass
class NetParams:
    pretrained: Optional[Union[None, bool]] = False
    num_classes: Optional[Union[None, int]] = 10
    activation: Optional[Union[str, None]] = None  # Default value
    initialization: Optional[Union[None, str]] = 'kaiming'
    
    # for FC:
    input_size: Optional[Union[None, int]] = None  # Default for MNIST
    
    #for conv nets
    input_height: Optional[Union[None, int]] = None  # Default for MNIST
    input_width: Optional[Union[None, int]] = None
    in_channels: Optional[Union[None, int]] = 1
    out_channels: Optional[Union[None, int]] = 10
    kernel_size: Optional[Union[None, int]] = 5
    class Config:
        version_base = "1.1"

@dataclass
class LinearNetParams:
    input_size: Optional[Union[None, int]] = 784  # 784 for MNIST
    num_features: int = 2000
    num_outputs: int = 10
    num_hidden_layers: Optional[Union[None, int]] = 2
    act_type: Optional[Union[None, str]] = 'relu'
    # Initialization scheme for linear networks; default to Xavier for identity/linear activations
    initialization: Optional[Union[None, str]] = 'xavier'
    class Config:
        version_base = "1.1"

#dataclass for network configurations
@dataclass
class NetConfig:
    type: str
    netparams: Optional[Union[None, NetParams, LinearNetParams]] = None
    # High-level structural category to help learner selection ('conv' | 'fc' | 'other').
    network_class: Optional[str] = None
    class Config:
        version_base = "1.1"



@dataclass
class LlamaConfig:
    """Configuration for loading a HuggingFace LLaMA 3.1 model."""
    type: str = 'llama'
    model_name_or_path: str = 'meta-llama/Llama-3.1-8B'
    hf_token: Optional[str] = None
    torch_dtype: str = 'bfloat16'  # 'float16', 'float32', 'bfloat16'
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: Optional[str] = 'auto'
    trust_remote_code: bool = False
    class Config:
        version_base = "1.1"


@dataclass
class GrokkingTransformerConfig:
    _target_: str = "configs.configurations.GrokkingTransformerConfig"
    type: str = "GrokkingTransformer_pytorch_manual_implementation"
    vocab_size: int = 115
    max_seq_len: int = 10
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    dropout: float = 0.0



@dataclass
class BaseLearnerConfig:
    type: str 
    device: str = 'cuda'
    enable_cuda1_workarounds: bool = False  # Enable CPU eigendecomposition workarounds for cuda:1
    # Structural category of network; inferred if None.
    network_class: Optional[str] = None
    opt: str = 'adam' #or 'sgd'
    step_size: float= 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    weight_decay: float = 0.0
    momentum: Optional[float] = 0.0
    loss: str = 'cross_entropy'
    # for more complicated implementations that need to keep track of previous features
    # or for specialized regularization such as orthogonality regularization for SVD_Conv2d layers
    additional_regularization: Optional[Union[None, str]] = None
    lambda_orth: Optional[Union[None, float]] = None
    
    to_perturb: Optional[bool] = False
    perturb_scale: Optional[float] = 0.1
    # previous_features: Optional[Union[None, torch.Tensor]] = None
    # latest_gradients: Optional[Union[None, torch.Tensor]] = None
    class Config:
        version_base = "1.1"

@dataclass
class ContinuousBackpropConfig(BaseLearnerConfig):
    type: str = 'cbp'
    neurons_replacement_rate: float = 0.001
    decay_rate_utility_track: float = 0.9
    maturity_threshold: int = 100
    util_type: str = 'contribution'
    init: str = 'kaiming'
    accumulate: bool = False
    outgoing_random: bool = False
    use_grad_clip: bool = False
    grad_clip_max_norm: float = 1.0
    class Config:
        version_base = "1.1"


@dataclass
class RRContinuousBackpropConfig(ContinuousBackpropConfig):
    type: str = 'rr_cbp'
    rrcbp_enabled: bool = True
    sigma_ema_beta: float = 0.99
    sigma_ridge: float = 1e-4
    max_proj_trials: int = 4
    proj_eps: float = 1e-8
    center_bias: str = 'mean'
    nullspace_seed_epsilon: float = 0.0
    diag_sigma_only: bool = False
    orthonormalize_batch: bool = True
    improve_conditioning_if_saturated: bool = True
    log_rank_metrics_every: int = 0
    covariance_dtype: Optional[str] = None
    sigma_eig_floor: float = 1e-6
    projector_reg_epsilon: float = 1e-6
    tau: float = 1e-2
    lambda_star: Optional[float] = None
    use_lambda_star: bool = False
    epsilon_micro_seed: float = 1e-4
    use_micro_seed: bool = True
    estimate_chi0_from_batch: bool = False
    chi0_override: Optional[float] = None
    class Config:
        version_base = "1.1"


@dataclass
class RRCBP2Config(ContinuousBackpropConfig):
    """Configuration for RR-CBP2 and RR-CBP-E2 algorithms (version 2 implementation)."""
    type: str = 'rr_cbp2'
    
    # Core RR-CBP2 settings
    rrcbp_enabled: bool = True
    
    # Covariance EMA settings
    sigma_ema_beta: float = 0.99
    sigma_ridge: float = 1e-4
    
    # Projection settings
    max_proj_trials: int = 4
    proj_eps: float = 1e-8
    projector_reg_epsilon: float = 1e-6
    
    # Bias centering: 'mean' or 'median'
    center_bias: str = 'mean'
    
    # Σ-geometry settings
    diag_sigma_only: bool = False
    sigma_eig_floor: float = 1e-6
    covariance_dtype: Optional[str] = None
    
    # Orthonormalization within replacement batch
    orthonormalize_batch: bool = True
    
    # Logging
    log_rank_metrics_every: int = 0
    
    # ===== RR-CBP-E2 (Energy-Aware) Settings =====
    # Set use_energy_budget=True to enable energy-aware mode (rr_cbp_e_2)
    use_energy_budget: bool = False
    
    # Energy budget parameters (Section 6 of algorithm guide)
    tau: float = 1e-2
    use_lambda_star: bool = False
    lambda_star: Optional[float] = None
    
    # Chi-squared constant estimation
    chi0_override: Optional[float] = None
    estimate_chi0_from_batch: bool = False
    
    # Micro-seeding outgoing weights
    use_micro_seed: bool = False
    epsilon_micro_seed: float = 1e-4
    
    # Saturated regime handling
    improve_conditioning_if_saturated: bool = True
    nullspace_seed_epsilon: float = 0.0
    
    class Config:
        version_base = "1.1"


@dataclass
class BackpropConfig(BaseLearnerConfig):
    type: str = 'backprop'
    to_perturb: bool = False
    class Config:
        version_base = "1.1"

@dataclass
class EvaluationConfig:
    use_testset: bool = False
    eval_freq_epoch: int = 1
    eval_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'loss'])
    type: Optional[Union[None, str]] = None
    # save_name: str = 'basic_training'


 
# @dataclass
# class ContinualBackpropConfig(BaseLearnerConfig):
#     replacement_rate: float = 0.001
#     decay_rate: float = 0.9
#     maturity_threshold: int = 100
#     util_type: str = 'contribution'
#     init: str = 'kaiming'
#     accumulate: bool = False
#     outgoing_random: bool = False

@dataclass
class LoggingConfig:
    save_dir: str = 'results_raw'
    save_name: str = 'basic_training'
    class Config:
        version_base = "1.1"

@dataclass
class WandbConfig:
    project: str = 'basic_training'
    entity: str = ''

@dataclass
class ExperimentConfig:
    use_wandb: bool = False
    use_json: bool = False
    log_freq_every_n_task: int = 1
    wandb: WandbConfig = field(default_factory=WandbConfig)
    runs: int = 1
    run_id: Optional[int] = None
    seed: Optional[int] = None
    device: str = 'cuda'
    epochs: int = 10
    batch_size: int = 128
    data: DataConfig = field(default_factory=DataConfig)
    net: Union[NetConfig, GrokkingTransformerConfig, LlamaConfig] = field(default_factory=lambda: NetConfig(type='ConvNet'))
    learner: Union[BackpropConfig, ContinuousBackpropConfig, RRContinuousBackpropConfig] = field(default_factory=BackpropConfig)
    evaluation: Union[EvaluationConfig, None] = field(default_factory=EvaluationConfig)
    track_rank: bool = False
    prop_for_approx_or_l1_rank: float = 0.99
    numerical_rank_epsilon: float = 0.01
    use_pytorch_entropy_for_effective_rank: bool = True
    track_dead_units: bool = False
    threshold_for_non_saturating_act: float = 0.01
    track_weight_magnitude: bool = False
    layers_identifier: Optional[List[str]] = None
    num_workers: int = 2
    # task shifting (inert default fields; wired later)
    task_shift_mode: Optional[str] = None  # e.g., 'permuted_input', 'permuted_output', 'continuous_input_deformation', 'drifting_values'
    task_shift_param: Optional[Dict[str, Any]] = None  # container for mode-specific parameter groups
    num_tasks: int = 1  # number of tasks for stateless/stateful shifts
    # logging refinement: when True, only the active sub-config for the chosen task_shift_mode
    # is retained in the dict passed to external loggers (e.g., W&B); inactive sibling
    # parameter groups are pruned to reduce noise.
    prune_inactive_task_shift_params: bool = False
    
    class Config:
        version_base = "1.1"

# Register the configurations with Hydra
cs = ConfigStore.instance()
