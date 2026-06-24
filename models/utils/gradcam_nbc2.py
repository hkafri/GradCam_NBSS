from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor


def resolve_nbc2_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last Group Conv1d inside the final NBC2 ConvFFN block."""

    nbc2 = model.arch if hasattr(model, "arch") else model
    if not hasattr(nbc2, "sa_layers"):
        raise TypeError("Expected an NBC2 model or an NBSS wrapper that contains NBC2 as .arch")
    if len(nbc2.sa_layers) == 0:
        raise ValueError("NBC2 model has no self-attention layers")

    target_block = nbc2.sa_layers[-1]
    if not hasattr(target_block, "conv"):
        raise TypeError("Final NBC2 block does not expose a conv stack")

    target_layer = target_block.conv[6]
    if not isinstance(target_layer, torch.nn.Conv1d):
        raise TypeError("Resolved target layer is not a Conv1d module")
    return target_layer


@dataclass
class _HookCache:
    activations: Optional[Tensor] = None
    gradients: Optional[Tensor] = None


class NBC2GradCAM1D:
    """Custom 1D Grad-CAM for the final NBC2 grouped Conv1d layer."""

    def __init__(
        self,
        model: torch.nn.Module,
        speaker_index: int = 0,
        sample_index: int = 0,
        target_layer: Optional[torch.nn.Module] = None,
        target_fn: Optional[Callable[[Tensor, int, int], Tensor]] = None,
    ) -> None:
        self.model = model
        self.speaker_index = speaker_index
        self.sample_index = sample_index
        self.target_layer = target_layer or resolve_nbc2_target_layer(model)
        self.target_fn = target_fn or self._default_target_fn
        self._cache = _HookCache()
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "NBC2GradCAM1D":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _forward_hook(self, _module: torch.nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self._cache.activations = output.detach()

    def _backward_hook(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[Optional[Tensor], ...],
        grad_output: tuple[Optional[Tensor], ...],
    ) -> None:
        self._cache.gradients = grad_output[0].detach() if grad_output[0] is not None else None

    def _default_target_fn(self, output: Tensor, freq_index: int, speaker_index: int) -> Tensor:
        speaker_start = speaker_index * 2
        speaker_stop = speaker_start + 2
        speaker_sequence = output[self.sample_index, freq_index, :, speaker_start:speaker_stop]
        return speaker_sequence.mean()

    def _compute_heatmap_for_frequency(self, freq_index: int, num_freqs: int) -> Tensor:
        if self._cache.activations is None or self._cache.gradients is None:
            raise RuntimeError("Hooks did not capture activations and gradients")

        flat_index = self.sample_index * num_freqs + freq_index
        activations = self._cache.activations[flat_index]
        gradients = self._cache.gradients[flat_index]

        if activations.ndim != 2 or gradients.ndim != 2:
            raise RuntimeError("Expected target layer activations and gradients to have shape [C, T]")

        weights = gradients.mean(dim=-1, keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=0))
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value
        return cam

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.generate_heatmap(inputs)

    def generate_heatmap(self, inputs: Tensor) -> Tensor:
        """Compute a 2D Grad-CAM heatmap with shape [F, T]."""

        if inputs.ndim != 4:
            raise ValueError("Expected input shape [B, F, T, H]")

        batch_size, num_freqs, _, _ = inputs.shape
        if self.sample_index >= batch_size:
            raise IndexError("sample_index is out of range for the provided batch")

        original_training_state = self.model.training
        self.model.eval()

        heatmaps = []
        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                output = self.model(inputs)

                for freq_index in range(num_freqs):
                    self._cache.gradients = None

                    target = self.target_fn(output, freq_index, self.speaker_index)
                    retain_graph = freq_index < num_freqs - 1
                    target.backward(retain_graph=retain_graph)

                    heatmaps.append(self._compute_heatmap_for_frequency(freq_index, num_freqs))
                    self.model.zero_grad(set_to_none=True)
        finally:
            self.model.train(original_training_state)

        return torch.stack(heatmaps, dim=0)
