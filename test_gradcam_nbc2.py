from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

try:
    import librosa
except Exception:  # pragma: no cover - optional dependency fallback
    librosa = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency fallback
    sf = None

from models.arch.NBC2 import NBC2
from models.utils.gradcam_nbc2 import resolve_nbc2_target_layer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO_CANDIDATES = [
    REPO_ROOT / "examples" / "NBC2" / "0_mix_8_Full.wav",
    REPO_ROOT / "examples" / "NBC2" / "0_mix_8.wav",
    REPO_ROOT / "examples" / "audio_examples" / "0_mix_8_Full.wav",
    REPO_ROOT / "audio_examples" / "0_mix_8_Full.wav",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NBC2 Grad-CAM on a real NBSS example.")
    parser.add_argument("--audio-path", type=str, default=None, help="Path to a multichannel mixture wav file. Defaults to an official example WAV shipped with this repo.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional NBC2 checkpoint path. If omitted, a fresh NBC2-small model is used.")
    parser.add_argument("--output-figure", type=str, default="gradcam_nbc2.png", help="Where to save the combined visualization.")
    parser.add_argument("--output-heatmap", type=str, default=None, help="Optional path to save the heatmap tensor with torch.save().")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for the input audio.")
    parser.add_argument("--num-channels", type=int, default=None, help="Optional override for the number of mixture channels. If omitted, the script infers the channel count from the WAV.")
    parser.add_argument("--n-speakers", type=int, default=2, help="Number of separated speakers in the model output.")
    parser.add_argument("--speaker-index", type=int, default=0, help="Zero-based speaker index to explain.")
    parser.add_argument("--ref-channel", type=int, default=0, help="Reference channel used for input normalization.")
    parser.add_argument("--plot-channel", type=int, default=0, help="Mixture channel to plot in the input spectrogram.")
    parser.add_argument("--n-fft", type=int, default=512, help="STFT FFT size.")
    parser.add_argument("--hop-length", type=int, default=256, help="STFT hop length.")
    parser.add_argument("--device", type=str, default=None, help="torch device, for example cpu, cuda, or cuda:0.")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively after saving it.")
    return parser.parse_args()


def resolve_audio_path(audio_path: Optional[str]) -> Optional[Path]:
    if audio_path is not None:
        return Path(audio_path)

    for candidate in DEFAULT_AUDIO_CANDIDATES:
        if candidate.exists():
            return candidate

    for directory in (REPO_ROOT / "examples", REPO_ROOT / "audio_examples"):
        if directory.exists():
            wavs = sorted(directory.rglob("*.wav"))
            if wavs:
                return wavs[0]

    return None


def load_audio(audio_path: Optional[str], sample_rate: int) -> Tuple[Tensor, int, Path]:
    resolved_path = resolve_audio_path(audio_path)
    if resolved_path is None:
        raise FileNotFoundError("No local NBSS example WAV was found. Place one under examples/NBC2/ or pass --audio-path.")

    if not resolved_path.exists():
        raise FileNotFoundError(resolved_path)

    if librosa is not None:
        audio, loaded_sr = librosa.load(resolved_path.as_posix(), sr=sample_rate, mono=False)
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        elif audio.shape[0] > audio.shape[1]:
            audio = audio.T
        waveform = torch.from_numpy(audio.astype(np.float32))
        return waveform, loaded_sr, resolved_path

    if sf is None:
        raise ModuleNotFoundError("librosa or soundfile is required to load audio")

    audio, loaded_sr = sf.read(resolved_path.as_posix(), always_2d=True)
    if loaded_sr != sample_rate:
        raise RuntimeError(
            f"Audio sample rate is {loaded_sr}, but the integration test expects {sample_rate} Hz. Install librosa to enable resampling or provide a 16 kHz WAV."
        )
    waveform = torch.from_numpy(audio.T.astype(np.float32))
    return waveform, loaded_sr, resolved_path


def adapt_channel_count(waveform: Tensor, num_channels: Optional[int]) -> Tensor:
    if waveform.ndim != 2:
        raise ValueError("Expected waveform shape [Channels, Time]")

    if num_channels is None:
        return waveform

    channels, _ = waveform.shape
    if channels == num_channels:
        return waveform
    if channels > num_channels:
        return waveform[:num_channels]

    pad_count = num_channels - channels
    padding = waveform[-1:].expand(pad_count, -1)
    return torch.cat([waveform, padding], dim=0)


def build_nbc2_small(num_channels: int, num_freqs: int, n_speakers: int) -> NBC2:
    block_kwargs = {
        "n_heads": 2,
        "dropout": 0,
        "conv_kernel_size": 3,
        "n_conv_groups": 8,
        "norms": ("LN", "GBN", "GBN"),
        "group_batch_norm_kwargs": {
            "share_along_sequence_dim": False,
        },
    }
    return NBC2(
        dim_input=num_channels * 2,
        dim_output=n_speakers * 2,
        n_layers=8,
        encoder_kernel_size=5,
        dim_hidden=96,
        dim_ffn=192,
        num_freqs=num_freqs,
        block_kwargs=block_kwargs,
    )


def load_checkpoint_into_model(model: NBC2, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key
        for prefix in ("_orig_mod.arch.", "arch.", "_orig_mod.", "model.", "module."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]
        cleaned_state_dict[cleaned_key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing or unexpected:
        print(f"Checkpoint loaded with missing keys={len(missing)} and unexpected keys={len(unexpected)}")


def stft_to_model_input(waveform: Tensor, sample_rate: int, n_fft: int, hop_length: int, ref_channel: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if waveform.ndim != 2:
        raise ValueError("Expected waveform shape [Channels, Time]")

    device = waveform.device
    window = torch.hann_window(n_fft, device=device)
    channels, num_samples = waveform.shape
    stft = torch.stft(
        waveform.reshape(channels, num_samples),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        win_length=n_fft,
        return_complex=True,
    )
    stft = stft.unsqueeze(0).permute(0, 2, 3, 1).contiguous()  # [1, F, T, C]

    if ref_channel < 0 or ref_channel >= stft.shape[-1]:
        raise IndexError("ref_channel is out of range for the loaded mixture")

    ref_mag_mean = torch.abs(stft[..., ref_channel]).mean(dim=2)
    stft = stft / (ref_mag_mean.reshape(1, stft.shape[1], 1, 1) + 1e-8)
    model_input = torch.view_as_real(stft).reshape(1, stft.shape[1], stft.shape[2], stft.shape[3] * 2)
    time_axis = torch.arange(stft.shape[2], device=device, dtype=torch.float32) * hop_length / sample_rate
    freq_axis = torch.linspace(0.0, sample_rate / 2.0, stft.shape[1], device=device, dtype=torch.float32)
    return model_input, stft, time_axis, freq_axis


def compute_two_gradcam_heatmaps(model: NBC2, model_input: Tensor, speaker_idx_a: int = 0, speaker_idx_b: int = 1) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute Grad-CAM heatmaps for two speaker indices from a single forward pass.

    Returns (heatmap_a, heatmap_b, speaker_complex) where heatmaps are [F, T] tensors.
    """
    target_layer = resolve_nbc2_target_layer(model)
    cache = {"activations": None, "gradients": None}

    def forward_hook(_module, _inputs, output):
        cache["activations"] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        cache["gradients"] = grad_output[0].detach() if grad_output[0] is not None else None

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        output = model(model_input)
        if output.shape[-1] % 2 != 0:
            raise RuntimeError("NBC2 output feature dimension must be divisible by 2")

        num_speakers = output.shape[-1] // 2
        for idx in (speaker_idx_a, speaker_idx_b):
            if idx < 0 or idx >= num_speakers:
                raise IndexError("speaker_index is out of range for the model output")

        speaker_complex = torch.view_as_complex(
            output.reshape(output.shape[0], output.shape[1], output.shape[2], num_speakers, 2).contiguous()
        )

        # First backward for speaker A (retain graph for second backward)
        target_a = speaker_complex[..., speaker_idx_a].abs().sum()
        target_a.backward(retain_graph=True)
        activations = cache["activations"]
        gradients_a = cache["gradients"]
        if activations is None or gradients_a is None:
            raise RuntimeError("The Grad-CAM hooks did not capture activations and gradients for speaker A")

        # Compute heatmap A
        batch_size, num_freqs, _, _ = model_input.shape
        activations_a = activations.reshape(batch_size, num_freqs, activations.shape[1], activations.shape[2])
        gradients_a = gradients_a.reshape(batch_size, num_freqs, gradients_a.shape[1], gradients_a.shape[2])

        heatmap_per_freq_a = []
        for freq_index in range(num_freqs):
            freq_activations = activations_a[0, freq_index]
            freq_gradients = gradients_a[0, freq_index]
            alpha_k = freq_gradients.mean(dim=-1, keepdim=True)
            cam_1d = torch.relu((alpha_k * freq_activations).sum(dim=0))
            heatmap_per_freq_a.append(cam_1d)

        heatmap_a = torch.stack(heatmap_per_freq_a, dim=0)
        # Normalize
        hmin, hmax = heatmap_a.min(), heatmap_a.max()
        if hmax > hmin:
            heatmap_a = (heatmap_a - hmin) / (hmax - hmin)
        else:
            heatmap_a = torch.zeros_like(heatmap_a)

        # Clear gradients from first backward
        model.zero_grad()

        # Second backward for speaker B
        target_b = speaker_complex[..., speaker_idx_b].abs().sum()
        target_b.backward()
        gradients_b = cache["gradients"]
        if activations is None or gradients_b is None:
            raise RuntimeError("The Grad-CAM hooks did not capture activations and gradients for speaker B")

        activations_b = activations.reshape(batch_size, num_freqs, activations.shape[1], activations.shape[2])
        gradients_b = gradients_b.reshape(batch_size, num_freqs, gradients_b.shape[1], gradients_b.shape[2])

        heatmap_per_freq_b = []
        for freq_index in range(num_freqs):
            freq_activations = activations_b[0, freq_index]
            freq_gradients = gradients_b[0, freq_index]
            alpha_k = freq_gradients.mean(dim=-1, keepdim=True)
            cam_1d = torch.relu((alpha_k * freq_activations).sum(dim=0))
            heatmap_per_freq_b.append(cam_1d)

        heatmap_b = torch.stack(heatmap_per_freq_b, dim=0)
        hmin, hmax = heatmap_b.min(), heatmap_b.max()
        if hmax > hmin:
            heatmap_b = (heatmap_b - hmin) / (hmax - hmin)
        else:
            heatmap_b = torch.zeros_like(heatmap_b)

        return heatmap_a.detach(), heatmap_b.detach(), speaker_complex.detach()
    finally:
        forward_handle.remove()
        backward_handle.remove()


def magnitude_db(spectrogram: Tensor) -> np.ndarray:
    magnitude = spectrogram.abs().detach().cpu().numpy()
    if librosa is not None:
        return librosa.amplitude_to_db(magnitude, ref=np.max)
    return 20.0 * np.log10(np.maximum(magnitude, 1e-8))


def plot_results_5(
    mixture_spec_db: np.ndarray,
    speaker1_spec_db: np.ndarray,
    heatmap1: np.ndarray,
    speaker2_spec_db: np.ndarray,
    heatmap2: np.ndarray,
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    output_path: str,
    show: bool,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True, sharey=True, constrained_layout=True)

    plot_specs = [
        (axes[0], mixture_spec_db, "Input mixture spectrogram", "magma"),
        (axes[1], speaker1_spec_db, "Separated output spectrogram - Speaker 1", "magma"),
        (axes[2], heatmap1, "NBC2 Grad-CAM heatmap - Speaker 1", "inferno"),
        (axes[3], speaker2_spec_db, "Separated output spectrogram - Speaker 2", "magma"),
        (axes[4], heatmap2, "NBC2 Grad-CAM heatmap - Speaker 2", "inferno"),
    ]

    extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
    for ax, data, title, cmap in plot_specs:
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.set_title(title)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        fig.colorbar(im, ax=ax, pad=0.01)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    waveform, sample_rate, audio_path = load_audio(args.audio_path, args.sample_rate)
    waveform = adapt_channel_count(waveform, args.num_channels).to(device)
    effective_channels = waveform.shape[0]

    print(f"Using example WAV: {audio_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Waveform shape: {tuple(waveform.shape)}")

    num_freqs = args.n_fft // 2 + 1
    model = build_nbc2_small(effective_channels, num_freqs, args.n_speakers).to(device)
    if args.checkpoint is not None:
        load_checkpoint_into_model(model, args.checkpoint, device)
    model.eval()

    model_input, stft_complex, time_axis, freq_axis = stft_to_model_input(
        waveform=waveform,
        sample_rate=sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        ref_channel=args.ref_channel,
    )
    model_input = model_input.to(device)

    # Compute two Grad-CAM heatmaps (Speaker 1 and Speaker 2) from a single forward pass
    speaker_a_idx = 0
    speaker_b_idx = 1 if args.n_speakers > 1 else 0
    heatmap_a, heatmap_b, speaker_complex = compute_two_gradcam_heatmaps(model, model_input, speaker_a_idx, speaker_b_idx)

    mixture_spec_db = magnitude_db(stft_complex[0, :, :, args.plot_channel])
    speaker1_spec_db = magnitude_db(speaker_complex[0, :, :, speaker_a_idx])
    speaker2_spec_db = magnitude_db(speaker_complex[0, :, :, speaker_b_idx])

    output_path = Path(args.output_figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results_5(
        mixture_spec_db=mixture_spec_db,
        speaker1_spec_db=speaker1_spec_db,
        heatmap1=heatmap_a.detach().cpu().numpy(),
        speaker2_spec_db=speaker2_spec_db,
        heatmap2=heatmap_b.detach().cpu().numpy(),
        time_axis=time_axis.detach().cpu().numpy(),
        freq_axis=freq_axis.detach().cpu().numpy(),
        output_path=output_path.as_posix(),
        show=args.show,
    )

    if args.output_heatmap is not None:
        heatmap_file = Path(args.output_heatmap)
        heatmap_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"heatmap_a": heatmap_a, "heatmap_b": heatmap_b}, heatmap_file)

    print(f"Saved visualization to: {output_path}")
    if args.output_heatmap is not None:
        print(f"Saved heatmap tensor to: {args.output_heatmap}")


if __name__ == "__main__":
    main()
