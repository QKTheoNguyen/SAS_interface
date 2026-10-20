from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio.functional as AF

from sas_worker import ModelAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
SVA_ROOT = REPO_ROOT / "third_party" / "StreamVoiceAnon"
REFERENCES_DIR = Path(__file__).parent / "references"
HF_REPO = "Plachta/StreamVoiceAnon"
CONFIG = "configs/config_firefly_arvcasr_8192_delay0_8.yaml"
CHECKPOINT = "pretrained_checkpoints/dual_ar_delay_0_8.pth"

# The AR model's KV cache holds 2048 positions: 33 for the speaker condition,
# then 2 per codec frame (reference + delay + source). 1 frame = 2048 samples at 44.1 kHz.
MAX_SEQ_LEN = 2048
SPK_CONDITION_LEN = 33
SAMPLES_PER_FRAME = 2048


def link_checkpoints() -> None:
    """Point SVA's expected ``pretrained_checkpoints/`` at the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(HF_REPO)
    link = SVA_ROOT / "pretrained_checkpoints"
    if link.is_symlink() and Path(os.readlink(link)) != Path(snapshot):
        link.unlink()
    if not link.exists():
        link.symlink_to(snapshot, target_is_directory=True)


class Adapter(ModelAdapter):
    sample_rate = 44100

    def prefetch(self) -> None:
        link_checkpoints()

    def load(self, device: torch.device) -> None:
        link_checkpoints()
        # SVA resolves its configs and checkpoints relative to its repo root.
        # This worker serves only SVA, so changing the working directory is safe.
        os.chdir(SVA_ROOT)
        sys.path.insert(0, str(SVA_ROOT))
        from evaluations.infer_arvc import InferenceWrapper

        if device.type != "cuda":
            raise RuntimeError("StreamVoiceAnon needs a CUDA GPU (its KV cache and autocast are fp16 on CUDA).")
        self.wrapper = InferenceWrapper(CONFIG, CHECKPOINT)
        self.references = {p.stem: p for p in sorted(REFERENCES_DIR.glob("*.flac"))}

    def choices(self, param: str) -> list[str]:
        if param != "reference_speaker":
            raise KeyError(param)
        return list(self.references)

    def _pool_reference(self, name: str) -> torch.Tensor:
        if name == "random":
            name = random.choice(list(self.references))
        if name not in self.references:
            raise ValueError(f"Unknown pseudo-speaker {name!r}")
        audio, sr = sf.read(self.references[name], dtype="float32")
        return AF.resample(torch.from_numpy(audio)[None], sr, self.sample_rate)

    @torch.inference_mode()
    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        if task == "voice_conversion":
            if not targets:
                raise ValueError("Voice conversion needs at least one target voice clip.")
            references, alpha = targets, 1.0
        else:
            references = targets or [self._pool_reference(str(params.get("reference_speaker", "random")))]
            alpha = float(params.get("alpha", 0.5))
        delay = int(params.get("delay", 2))
        max_ref = int(float(params.get("max_reference_seconds", 10)) * self.sample_rate)

        reference = torch.cat(references, dim=-1)[:, :max_ref]
        frames = (reference.size(-1) + source.size(-1)) // SAMPLES_PER_FRAME + delay + 2
        if SPK_CONDITION_LEN + 2 * frames > MAX_SEQ_LEN:
            budget = ((MAX_SEQ_LEN - SPK_CONDITION_LEN) // 2 - delay - 2) * SAMPLES_PER_FRAME / self.sample_rate
            raise ValueError(
                f"Source + reference is {(reference.size(-1) + source.size(-1)) / self.sample_rate:.1f} s; "
                f"this model's context fits about {budget:.0f} s. Shorten the source or the reference length (Advanced)."
            )
        return self._generate(source, reference, alpha, delay)

    def _generate(self, source: torch.Tensor, reference: torch.Tensor, alpha: float, delay: int) -> torch.Tensor:
        """Tensor version of ``InferenceWrapper.infer`` (which only accepts file paths)."""
        w = self.wrapper
        src = source.to(self.device)
        ref = reference.to(self.device)

        ref_16k = AF.resample(ref, self.sample_rate, w.RESAMPLE_FREQ)
        ref_lens_16k = w.create_wave_lens_tensor(ref_16k)
        style_vectors = w.calculate_style_vec(ref_16k, ref_lens_16k)
        timbre_latents = w.calculate_timbre_latent(ref_16k, ref_lens_16k)

        ref_audio_codes, _, _, _ = w.wav2target_fn(ref, w.create_wave_lens_tensor(ref))
        src_content_codes, _ = w.speech_tokenizer.encode(src, w.create_wave_lens_tensor(src))
        ref_content_codes, _ = w.speech_tokenizer.encode(ref, w.create_wave_lens_tensor(ref))

        w.model.set_delay(delay=delay)
        style_vectors = w.apply_noise_mixing(style_vectors, alpha)
        timbre_latents = w.apply_noise_mixing(timbre_latents, alpha)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            vc_codes = w.model.generate(
                ref_content_codes=ref_content_codes.squeeze(0),
                ref_audio_codes=ref_audio_codes,
                src_content_codes=src_content_codes.squeeze(0),
                style_vectors=style_vectors,
                timbre_latents=timbre_latents,
            )
        return w.code2wav_fn(vc_codes).squeeze().float().cpu()
