"""Tests for the HuggingFace LLaMA 3.1 model loader.

These tests use ``unittest.mock`` to patch HuggingFace ``transformers`` so that
no model weights are downloaded during CI.  Each test validates a specific
aspect of the :class:`~src.models.llama_loader.LlamaLoader` and the factory
integration without requiring a GPU or a HuggingFace API token.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.configurations import LlamaConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_transformers():
    """Return a minimal fake *transformers* module tree accepted by llama_loader."""
    import torch

    # --- fake output --------------------------------------------------------
    fake_output = MagicMock()
    fake_output.logits = torch.zeros(2, 5, 32000)

    # --- fake model ---------------------------------------------------------
    fake_model = MagicMock()
    fake_model.return_value = fake_output
    fake_model.parameters.return_value = iter([torch.zeros(1)])
    fake_model.generate.return_value = torch.zeros(2, 10, dtype=torch.long)

    # --- fake tokenizer -----------------------------------------------------
    fake_tokenizer = MagicMock()

    # --- AutoModelForCausalLM -----------------------------------------------
    MockAutoModel = MagicMock()
    MockAutoModel.from_pretrained.return_value = fake_model

    # --- AutoTokenizer ------------------------------------------------------
    MockAutoTokenizer = MagicMock()
    MockAutoTokenizer.from_pretrained.return_value = fake_tokenizer

    # --- BitsAndBytesConfig -------------------------------------------------
    MockBnBConfig = MagicMock()

    # Build a fake module
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = MockAutoModel
    fake_transformers.AutoTokenizer = MockAutoTokenizer
    fake_transformers.BitsAndBytesConfig = MockBnBConfig

    return fake_transformers, fake_model, fake_tokenizer, MockAutoModel, MockAutoTokenizer


# ---------------------------------------------------------------------------
# Config tests (no torch / transformers required)
# ---------------------------------------------------------------------------

class TestLlamaConfig:
    """Validate the LlamaConfig dataclass defaults and field types."""

    def test_default_model_name(self):
        cfg = LlamaConfig()
        assert cfg.model_name_or_path == "meta-llama/Llama-3.1-8B"

    def test_default_dtype(self):
        cfg = LlamaConfig()
        assert cfg.torch_dtype == "bfloat16"

    def test_default_quantisation_flags_false(self):
        cfg = LlamaConfig()
        assert cfg.load_in_8bit is False
        assert cfg.load_in_4bit is False

    def test_default_type_field(self):
        cfg = LlamaConfig()
        assert cfg.type == "llama"

    def test_custom_model_path(self):
        cfg = LlamaConfig(model_name_or_path="/local/path/to/llama")
        assert cfg.model_name_or_path == "/local/path/to/llama"

    def test_hf_token_default_none(self):
        cfg = LlamaConfig()
        assert cfg.hf_token is None

    def test_device_map_default_auto(self):
        cfg = LlamaConfig()
        assert cfg.device_map == "auto"

    def test_trust_remote_code_default_false(self):
        cfg = LlamaConfig()
        assert cfg.trust_remote_code is False


# ---------------------------------------------------------------------------
# LlamaLoader tests (transformers mocked)
# ---------------------------------------------------------------------------

class TestLlamaLoader:
    """Validate LlamaLoader construction and forward pass using mocked transformers."""

    def _build_loader(self, cfg: LlamaConfig, fake_transformers):
        """Patch transformers and instantiate LlamaLoader."""
        import importlib

        # Inject the fake module before importing llama_loader so that the
        # 'from transformers import …' inside __init__ picks up the mock.
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            # Force re-import so the patched sys.modules is used
            if "src.models.llama_loader" in sys.modules:
                del sys.modules["src.models.llama_loader"]
            from src.models.llama_loader import LlamaLoader
            loader = LlamaLoader(cfg)
        return loader

    def test_loader_is_nn_module(self):
        import torch.nn as nn
        ft, _, _, _, _ = _make_fake_transformers()
        cfg = LlamaConfig()
        loader = self._build_loader(cfg, ft)
        assert isinstance(loader, nn.Module)

    def test_tokenizer_attached(self):
        ft, _, fake_tokenizer, _, _ = _make_fake_transformers()
        cfg = LlamaConfig()
        loader = self._build_loader(cfg, ft)
        assert loader.tokenizer is fake_tokenizer

    def test_model_attached(self):
        ft, fake_model, _, _, _ = _make_fake_transformers()
        cfg = LlamaConfig()
        loader = self._build_loader(cfg, ft)
        assert loader.model is fake_model

    def test_from_pretrained_called_with_model_name(self):
        ft, _, _, MockAutoModel, MockAutoTokenizer = _make_fake_transformers()
        model_name = "meta-llama/Llama-3.1-8B"
        cfg = LlamaConfig(model_name_or_path=model_name)
        self._build_loader(cfg, ft)
        MockAutoModel.from_pretrained.assert_called_once()
        call_args = MockAutoModel.from_pretrained.call_args
        assert call_args[0][0] == model_name

    def test_tokenizer_from_pretrained_called_with_model_name(self):
        ft, _, _, _, MockAutoTokenizer = _make_fake_transformers()
        model_name = "meta-llama/Llama-3.1-8B"
        cfg = LlamaConfig(model_name_or_path=model_name)
        self._build_loader(cfg, ft)
        MockAutoTokenizer.from_pretrained.assert_called_once()
        call_args = MockAutoTokenizer.from_pretrained.call_args
        assert call_args[0][0] == model_name

    def test_forward_returns_output(self):
        import torch
        ft, fake_model, _, _, _ = _make_fake_transformers()
        cfg = LlamaConfig()
        loader = self._build_loader(cfg, ft)

        input_ids = torch.zeros(2, 5, dtype=torch.long)
        out = loader(input_ids)
        fake_model.assert_called_once()
        assert out is not None

    def test_generate_delegates_to_model(self):
        import torch
        ft, fake_model, _, _, _ = _make_fake_transformers()
        cfg = LlamaConfig()
        loader = self._build_loader(cfg, ft)

        input_ids = torch.zeros(1, 3, dtype=torch.long)
        loader.generate(input_ids, max_new_tokens=5)
        fake_model.generate.assert_called_once()

    def test_num_parameters_property(self):
        import torch
        ft, fake_model, _, _, _ = _make_fake_transformers()
        # Set up a real parameter so numel() works
        param = torch.zeros(10)
        fake_model.parameters.return_value = iter([param])
        cfg = LlamaConfig()
        loader = self._build_loader(cfg, ft)
        assert loader.num_parameters == 10

    def test_invalid_dtype_raises_value_error(self):
        ft, _, _, _, _ = _make_fake_transformers()
        cfg = LlamaConfig(torch_dtype="invalid_dtype")
        with pytest.raises(ValueError, match="Unsupported torch_dtype"):
            self._build_loader(cfg, ft)

    def test_hf_token_forwarded(self):
        ft, _, _, MockAutoModel, MockAutoTokenizer = _make_fake_transformers()
        cfg = LlamaConfig(hf_token="hf_test_token_123")
        self._build_loader(cfg, ft)
        # Check tokenizer call received the token
        tok_kwargs = MockAutoTokenizer.from_pretrained.call_args[1]
        assert tok_kwargs.get("token") == "hf_test_token_123"
        # Check model call received the token
        model_kwargs = MockAutoModel.from_pretrained.call_args[1]
        assert model_kwargs.get("token") == "hf_test_token_123"

    def test_missing_transformers_raises_import_error(self):
        """LlamaLoader should raise ImportError when transformers is absent."""
        # Remove llama_loader from cache to force re-import
        if "src.models.llama_loader" in sys.modules:
            del sys.modules["src.models.llama_loader"]

        with patch.dict("sys.modules", {"transformers": None}):
            from src.models.llama_loader import LlamaLoader
            cfg = LlamaConfig()
            with pytest.raises(ImportError, match="transformers"):
                LlamaLoader(cfg)


# ---------------------------------------------------------------------------
# model_factory integration tests (transformers mocked)
# ---------------------------------------------------------------------------

class TestModelFactoryLlama:
    """Validate that model_factory correctly dispatches to LlamaLoader."""

    def test_factory_returns_llama_loader(self):
        import torch.nn as nn
        ft, _, _, _, _ = _make_fake_transformers()
        cfg = LlamaConfig()

        with patch.dict("sys.modules", {"transformers": ft}):
            if "src.models.llama_loader" in sys.modules:
                del sys.modules["src.models.llama_loader"]
            from src.models.model_factory import model_factory
            from src.models.llama_loader import LlamaLoader

            loader = model_factory(cfg)

        assert isinstance(loader, LlamaLoader)
        assert isinstance(loader, nn.Module)

    def test_factory_unsupported_type_raises(self):
        from src.models.model_factory import model_factory
        from configs.configurations import LlamaConfig

        cfg = LlamaConfig(type="unsupported_llama_variant")
        # 'unsupported_llama_variant' won't match the 'llama' branch, so it
        # falls through to the ValueError at the end of the factory.
        with pytest.raises(ValueError, match="Unsupported model type"):
            model_factory(cfg)

    def test_factory_invalid_dtype_raises_value_error(self):
        """factory → LlamaLoader should surface a ValueError for bad dtype."""
        ft, _, _, _, _ = _make_fake_transformers()
        cfg = LlamaConfig(torch_dtype="float128")  # unsupported

        with patch.dict("sys.modules", {"transformers": ft}):
            if "src.models.llama_loader" in sys.modules:
                del sys.modules["src.models.llama_loader"]
            from src.models.model_factory import model_factory

            with pytest.raises(ValueError, match="Unsupported torch_dtype"):
                model_factory(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
