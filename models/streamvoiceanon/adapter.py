from __future__ import annotations

import contextlib
import io
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as AF

from sas_worker import ModelAdapter, StreamSession
from sas_worker.audio import read_wav

log = logging.getLogger("sas_worker.streamvoiceanon")
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

# Streaming settings from StreamVoiceAnon's real-time GUI (evaluations/real-time-gui.py).
STREAM_DEFAULTS = {"delay": 2, "block_frames": 1}
STREAM_CACHES = dict(encode_window_frames=64, decode_window_frames=64, max_seq_frames=768, buffer_frames=32)
STREAM_MAX_PROMPT_FRAMES = 64


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
        # compile_ar is safe for both modes (the per-token decode always has the same shape).
        # The encoder and vocoder are compiled separately below and only used while streaming,
        # because their non-streaming inputs change length every request (= a recompile each time).
        self.wrapper = InferenceWrapper(CONFIG, CHECKPOINT, compile_ar=True)
        self.references = {p.stem: p for p in sorted(REFERENCES_DIR.glob("*.flac"))}

        w = self.wrapper
        compile_kwargs = dict(fullgraph=True, backend="inductor", mode="reduce-overhead")
        self.eager_head = w.firefly.head
        self.compiled_head = torch.compile(w.firefly.head, **compile_kwargs)
        w.compiled_speech_tokenizer_encode = torch.compile(w.speech_tokenizer.encode, **compile_kwargs)
        try:
            self._warm_up()
        except Exception as exc:
            # Typically no C compiler or Triton on the node. Keep the model usable, just slower.
            log.exception("torch.compile failed; falling back to eager mode")
            self.warning = (
                f"Running without torch.compile ({type(exc).__name__}: {str(exc).splitlines()[0]}). "
                "Streaming will likely be slower than real time; see README_perso.md for the compiler setup."
            )
            w.model.compiled_fn = None
            w.compiled_speech_tokenizer_encode = w.speech_tokenizer.encode
            self.compiled_head = self.eager_head
            torch._dynamo.reset()
            self._warm_up()

    def _warm_up(self) -> None:
        """Run a short silent stream so torch.compile happens now (1-3 min), not on the first user stream.
        Window shapes do not depend on delay or block size, so one warm-up covers every setting."""
        session = self.open_stream("anonymization", [], {"reference_speaker": next(iter(self.references))})
        try:
            silence = torch.zeros(1, session.chunk_samples)
            for _ in range(STREAM_DEFAULTS["delay"] + 4):
                session.process(silence)
        finally:
            session.close()

    def open_stream(self, task: str, targets: list[torch.Tensor], params: dict[str, Any]) -> StreamSession:
        references, alpha = self._references_and_alpha(task, targets, params)
        # prefill_prompt caches the whole reference (not its max_prompt_frames copy), so crop it
        # to keep room in the 2048-position KV cache before the first prompt refill.
        max_ref = int(float(params.get("max_reference_seconds", 10)) * self.sample_rate)
        return SVAStream(
            self,
            [torch.cat(references, dim=-1)[:, :max_ref]],
            alpha=alpha,
            delay=int(params.get("delay", STREAM_DEFAULTS["delay"])),
            block_frames=int(params.get("block_frames", STREAM_DEFAULTS["block_frames"])),
        )

    def _references_and_alpha(self, task: str, targets: list[torch.Tensor], params: dict[str, Any]):
        if task == "voice_conversion":
            if not targets:
                raise ValueError("Voice conversion needs at least one target voice clip.")
            return targets, 1.0
        references = targets or [self._pool_reference(str(params.get("reference_speaker", "random")))]
        return references, float(params.get("alpha", 0.5))

    def choices(self, param: str) -> list[str]:
        if param != "reference_speaker":
            raise KeyError(param)
        return list(self.references)

    def _pool_reference(self, name: str) -> torch.Tensor:
        if name == "random":
            name = random.choice(list(self.references))
        if name not in self.references:
            raise ValueError(f"Unknown pseudo-speaker {name!r}")
        return read_wav(self.references[name].read_bytes(), self.sample_rate)

    @torch.inference_mode()
    def convert(self, task: str, source: torch.Tensor, targets: list[torch.Tensor], params: dict[str, Any]) -> torch.Tensor:
        references, alpha = self._references_and_alpha(task, targets, params)
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


class SVAStream(StreamSession):
    """Live session on top of ``InferenceWrapper.prefill_prompt`` / ``process_one_chunk``."""

    def __init__(self, adapter: Adapter, references: list[torch.Tensor], alpha: float, delay: int, block_frames: int):
        self.adapter = adapter
        self.w = adapter.wrapper
        self.input_sample_rate = self.output_sample_rate = Adapter.sample_rate
        self.chunk_samples = block_frames * SAMPLES_PER_FRAME
        self.algorithmic_latency_ms = delay * SAMPLES_PER_FRAME / Adapter.sample_rate * 1000

        self.w.firefly.head = adapter.compiled_head
        with quiet(), torch.inference_mode():
            self.w.prefill_prompt(
                [r.to(adapter.device) for r in references],
                max_prompt_frames=STREAM_MAX_PROMPT_FRAMES,
                delay=delay,
                alpha=alpha,
            )
            self.w.setup_stream_caches(decode_chunk_frames=block_frames, **STREAM_CACHES)

    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        with quiet(), torch.inference_mode():
            return self.w.process_one_chunk(chunk.to(self.adapter.device)).float().cpu()

    def close(self) -> None:
        self.w.firefly.head = self.adapter.eager_head


@contextlib.contextmanager
def quiet():
    """process_one_chunk prints three timing lines per chunk; keep worker logs readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield
