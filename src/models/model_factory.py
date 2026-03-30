import importlib# for dynamic import
# for support of all models available in torchvision:
from torchvision import models as torchvision_models
from typing import Any
# get parent directory of project root
import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = str(PROJECT_ROOT / "experiments/basic_training/cfg")
sys.path.append(str(PROJECT_ROOT))

from src.models.conv_net import ConvNet
import warnings

from configs.configurations import NetConfig, NetParams, GrokkingTransformerConfig, LlamaConfig
from typing import Any, Union


import hydra 
from omegaconf import DictConfig, OmegaConf

def _infer_conv_input_dims(params: NetParams):
    """Infer missing conv net input dimensions with conservative heuristics.

    Priority:
    1. If both provided, keep them.
    2. If one provided, copy to the other (assume square images).
    3. If none provided:
       - If in_channels == 1 -> assume MNIST (28x28)
       - Else if in_channels == 3:
           * if num_classes in {1000} -> ImageNet (224)
           * if num_classes in {10} -> assume CIFAR10 (32)
           * fallback 32
    Writes inferred values back into params and emits a warning.
    """
    changed = False
    h, w = params.input_height, params.input_width
    if h is not None and w is not None:
        return
    if h is not None and w is None:
        params.input_width = h
        changed = True
    elif w is not None and h is None:
        params.input_height = w
        changed = True
    else:  # both None
        in_ch = getattr(params, 'in_channels', 3)
        num_classes = getattr(params, 'num_classes', 10)
        if in_ch == 1:
            inferred = 28
        elif in_ch == 3:
            if num_classes == 1000:
                inferred = 224
            elif num_classes == 10:
                inferred = 32
            else:
                inferred = 32
        else:
            inferred = 32
        params.input_height = inferred
        params.input_width = inferred
        changed = True
    if changed:
        warnings.warn(
            f"Inferred conv input dims (input_height={params.input_height}, input_width={params.input_width}). "
            "Provide explicit values in net.netparams to silence this warning (Step 6).",
            UserWarning
        )


def model_factory(config: Union[NetConfig, GrokkingTransformerConfig, LlamaConfig]) -> Any:
    """
    Factory function to create model instances based on the configuration.

    Args:
        config (NetParams): Configuration object containing model parameters.

    Returns:
        Any: An instance of the requested PyTorch model.
    
    Raises:
        ValueError: If the specified model type is unsupported.
    """
    # Get the model type from the configuration. expect a string
    model_type = config.type

    if hasattr(config, 'vocab_size'):
        # Convert OmegaConf object to a standard dictionary for safe access
        conf_dict = OmegaConf.to_container(config, resolve=True)
        params = {
            "vocab_size": conf_dict.get('vocab_size', 115),
            "max_seq_len": conf_dict.get('max_seq_len', 10),
            "n_layers": conf_dict.get('n_layers', 2),
            "n_heads": conf_dict.get('n_heads', 4),
            "d_model": conf_dict.get('d_model', 128),
            "dropout": conf_dict.get('dropout', 0.0),
        }
        if model_type == 'GrokkingTransformer_pytorch_manual_implementation':
            from src.models.grokking_transformer import GrokkingTransformerManual
            return GrokkingTransformerManual(**params)
        elif model_type == 'GrokkingTransformer_pytorch_implementation':
            from src.models.grokking_transformer import GrokkingTransformerStandard
            return GrokkingTransformerStandard(**params)

    if model_type == 'llama':
        from src.models.llama_loader import LlamaLoader
        return LlamaLoader(config)

    # if model_type = 'convnet', return an instance of ConvNet:
    if model_type == 'ConvNet':
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet")
        _infer_conv_input_dims(config.netparams)
        return ConvNet(config.netparams)
    if model_type == "ConvNet_norm":
        from src.models.conv_net_weights_normalized import ConvNet_normalized
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_norm")
        _infer_conv_input_dims(config.netparams)
        return ConvNet_normalized(config.netparams)
    if model_type == "ConvNet_SVD":  
        from src.models.ConvNet_SVD import ConvNet_SVD
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_SVD")
        _infer_conv_input_dims(config.netparams)
        return ConvNet_SVD(config.netparams)
    
    if model_type == "ConvNet_FC_layer_norm":
        from src.models.layer_norm_conv_net import ConvNetWithFCLayerNorm
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_layer_norm")
        _infer_conv_input_dims(config.netparams)
        return ConvNetWithFCLayerNorm(config.netparams)
    
    if model_type == "ConvNet_conv_and_FC_layer_norm":
        from src.models.layer_norm_conv_net import ConvNet_conv_and_FC_LayerNorm
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_conv_and_FC_layer_norm")
        _infer_conv_input_dims(config.netparams)
        return ConvNet_conv_and_FC_LayerNorm(config.netparams)
        
    if model_type == 'ConvNet_batch_norm':
        from src.models.conv_net_batch_norm import ConvNetWithBatchNorm
        if config.netparams is None:
            raise ValueError("config.params cannot be None for ConvNet_batch_norm")
        _infer_conv_input_dims(config.netparams)
        return ConvNetWithBatchNorm(config.netparams)
    
    if model_type == 'vgg_custom':
        from src.models.VGG16_custom import vgg_with_internal_performance_track_custom_classifier as VGG_custom

        if config.netparams is None:
            raise ValueError("config.params cannot be None for VGG16_custom")
        return VGG_custom(config.netparams)
        
    if model_type == 'vgg_custom_norm':   
        from src.models.VGG_normalized_conv import vgg_normalized_custom
        if config.netparams is None:
            raise ValueError("config.params cannot be None for vgg_custom_norm")
        return vgg_normalized_custom(config.netparams)
        
    if model_type == 'resnet_custom':
        from src.models.ResNet18_custom import ResNet18_with_custom_classifier as ResNet_custom
        if config.netparams is None:
            raise ValueError("config.params cannot be None for resnet_custom")
        return ResNet_custom(config.netparams)
    
    if model_type == 'full_rank_resnet_custom':
        from src.models.full_rank_resnet import ResNet18_skip_to_last_with_custom_classifier as full_rank_ResNet_custom
        if config.netparams is None:
            raise ValueError("config.params cannot be None for resnet_custom")
        return full_rank_ResNet_custom(config.netparams)
     
    
    if model_type == "deep_ffnn":
        from src.models.deep_ffnn import DeepFFNN
        if config.netparams is None:
            raise ValueError("config.params cannot be None for deep_ffnn")
        return DeepFFNN(config.netparams)
    if model_type == "deep_ffnn_weight_norm_single_rescale":
        from src.models.deep_ffnn_with_normalized_weights import DeepFFNN_weight_norm_single_rescale
        if config.netparams is None:
            raise ValueError("config.params cannot be None for weight_norm_deep_ffnn")
        return DeepFFNN_weight_norm_single_rescale(config.netparams)
    if model_type== "deep_ffnn_weight_norm_multi_channel_rescale":
        from src.models.deep_ffnn_with_normalized_weights import DeepFFNN_weight_norm_multi_channel_rescale
        if config.netparams is None:
            raise ValueError("config.params cannot be None for weight_norm_deep_ffnn")
        return DeepFFNN_weight_norm_multi_channel_rescale(config.netparams)
    if model_type == "deep_ffnn_weight_batch_norm":
        from src.models.deep_ffnn_with_normalized_weights import DeepFFNN_EMA_batch_weight_norm
        return DeepFFNN_EMA_batch_weight_norm(config.netparams)
    if model_type == "ffnn_normal_BN":
        from src.models.deep_ffnn import FFNN_with_BN
        return FFNN_with_BN(config.netparams)
    
    # New RL backbones
    elif model_type == 'rl_cnn_backbone':
        from src.models.rl_backbones import SimpleCNN
        return SimpleCNN
    elif model_type == 'rl_mlp_backbone':
        from src.models.rl_backbones import SimpleMLP
        return SimpleMLP

    elif model_type == 'GrokkingTransformer_pytorch_manual_implementation':
        from src.models.grokking_transformer import GrokkingTransformerManual
        return GrokkingTransformerManual(**config.netparams)
    elif model_type == 'GrokkingTransformer_pytorch_implementation':
        from src.models.grokking_transformer import GrokkingTransformerStandard
        return GrokkingTransformerStandard(**config.netparams)

    else:
        # If the model type is not supported, raise a ValueError
        raise ValueError(f"Unsupported model type: {model_type}")
    
    
    # if model_type =='vgg_custom':
    #     if config.params is None:
    #         raise ValueError("config.params cannot be None for VGG16_custom")
    #     return vgg_with_internal_performance_track_custom_classifier(config.params)
    
config_name = "basic_config"
@hydra.main(config_path=CONFIG_PATH, config_name=config_name, version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    # get the network configuration:
    net_config = cfg.net
    
    # use the factory function to create the model instance
    
    model = model_factory(net_config)
    print('Model created successfully')
    print(model)

# for testing purposes
if __name__ == '__main__':
    
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs')
    config_name = "default_config"
    
    @hydra.main(config_path=CONFIG_PATH, config_name=config_name, version_base=None)
    def main(cfg: ExperimentConfig) -> None:
        """
        main function for testing the model factory
        """
        print(OmegaConf.to_yaml(cfg))
        
        # get the network configuration:
        net_config = cfg.net
        
        # use the factory function to create the model instance
        
        model = model_factory(net_config)
        print('Model created successfully')
        print(model)
