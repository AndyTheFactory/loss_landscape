"""
HuggingFace LLaMA 3.1 model loader.

Wraps the HuggingFace ``transformers`` AutoModelForCausalLM / AutoTokenizer into a
standard ``torch.nn.Module`` so it can be used interchangeably with other models
registered in the model factory.

Usage via factory::

    from configs.configurations import LlamaConfig, NetConfig
    from src.models.model_factory import model_factory

    cfg = LlamaConfig(
        model_name_or_path='meta-llama/Llama-3.1-8B',
        hf_token='hf_...',   # required for gated model
        torch_dtype='bfloat16',
        device_map='auto',
    )
    model = model_factory(cfg)
    # model is a LlamaLoader (nn.Module) with .tokenizer and .model attributes.

The ``forward`` method delegates to the underlying
``transformers.AutoModelForCausalLM`` and accepts the same keyword arguments
(``input_ids``, ``attention_mask``, ``labels``, …).

Requirements:
    pip install transformers accelerate
    # For 4-bit / 8-bit quantisation:
    pip install bitsandbytes
"""

import torch
import torch.nn as nn
from typing import Any, Optional


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
}


class LlamaLoader(nn.Module):
    """Thin ``nn.Module`` wrapper around a HuggingFace LLaMA 3.1 model.

    Attributes:
        model (transformers.PreTrainedModel): The underlying causal-LM model.
        tokenizer (transformers.PreTrainedTokenizer): Paired tokenizer.
        config (LlamaConfig): The configuration used to construct the loader.
    """

    def __init__(self, llama_config: Any) -> None:
        """Load the LLaMA 3.1 model and tokenizer from HuggingFace.

        Args:
            llama_config: A :class:`~configs.configurations.LlamaConfig` instance
                (or any object with the same attributes).

        Raises:
            ImportError: If the ``transformers`` package is not installed.
            ValueError: If an unsupported ``torch_dtype`` string is provided.
        """
        super().__init__()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required to load LLaMA models. "
                "Install it with:  pip install transformers accelerate"
            ) from exc

        self.llama_config = llama_config

        dtype_str = getattr(llama_config, "torch_dtype", "bfloat16")
        if dtype_str not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported torch_dtype '{dtype_str}'. "
                f"Choose one of: {list(_DTYPE_MAP.keys())}"
            )
        torch_dtype = _DTYPE_MAP[dtype_str]

        hf_token: Optional[str] = getattr(llama_config, "hf_token", None)
        model_name: str = llama_config.model_name_or_path
        device_map: Optional[str] = getattr(llama_config, "device_map", "auto")
        trust_remote_code: bool = getattr(llama_config, "trust_remote_code", False)
        load_in_8bit: bool = getattr(llama_config, "load_in_8bit", False)
        load_in_4bit: bool = getattr(llama_config, "load_in_4bit", False)

        # Build optional BitsAndBytes quantisation config
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                )
            except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
                raise ImportError(
                    "The 'bitsandbytes' package is required for 4-bit / 8-bit "
                    "quantisation. Install it with:  pip install bitsandbytes"
                ) from exc

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        # Load the causal-LM model
        model_kwargs: dict = dict(
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        """Run a forward pass through the LLaMA model.

        Args:
            input_ids: Token id tensor of shape ``(batch, seq_len)``.
            **kwargs: Additional keyword arguments forwarded to the underlying
                ``transformers`` model (e.g. ``attention_mask``, ``labels``).

        Returns:
            A ``transformers.modeling_outputs.CausalLMOutputWithPast`` named-tuple
            containing ``logits``, ``loss`` (when ``labels`` are provided), past
            key-values, and optional hidden states / attentions.
        """
        return self.model(input_ids=input_ids, **kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Thin wrapper around the underlying model's ``generate`` method.

        Args:
            input_ids: Tokenised prompt tensor.
            **kwargs: Forwarded to ``transformers.GenerationMixin.generate``.

        Returns:
            Generated token id tensor.
        """
        return self.model.generate(input_ids, **kwargs)

    @property
    def num_parameters(self) -> int:
        """Total number of parameters in the underlying LLaMA model."""
        return sum(p.numel() for p in self.model.parameters())
