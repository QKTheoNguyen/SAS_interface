"""Speaker similarity over time in the metrics worker (sas_worker.metrics).

Runs on CPU with a mock speaker encoder and synthetic tones: no GPU, no model download.
Run from the repo root:

    cd workers/metrics && uv run --with pytest pytest tests -q
"""

from __future__ import annotations

import io
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from sas_worker import metrics
from sas_worker.metrics import SAMPLE_RATE, MetricModel, rolling_speaker_similarity, time_aligned, validate_rolling

SR = SAMPLE_RATE
ENCODER_ID = "MockSpeakerEncoder"


class MockSpeakerEncoder:
    """Pooled 'speaker embedding' = energy in 32 FFT bands (0-4 kHz): equal tones -> cosine 1,
    tones far apart -> cosine ~0. ``fail_on`` makes the n-th call (0-based) raise."""

    sample_rate = SR
    output = "pooled"

    def __init__(self, fail_on: int | None = None):
        self.calls = 0
        self.fail_on = fail_on

    def encode(self, audio: torch.Tensor, sample_rate: int):
        assert sample_rate == SR
        call, self.calls = self.calls, self.calls + 1
        if call == self.fail_on:
            raise RuntimeError("mock inference failure")
        spectrum = torch.fft.rfft(audio.reshape(-1).double()).abs()
        bins = spectrum[: len(spectrum) // 2]  # 0 .. SR/4
        return SimpleNamespace(values=torch.stack([b.sum() for b in bins.tensor_split(32)]).float()[None])


def tones(*segments: tuple[float, float]) -> torch.Tensor:
    """Concatenated sine tones, each (frequency Hz, seconds); frequency 0 = digital silence."""
    parts = []
    for freq, seconds in segments:
        t = torch.arange(round(seconds * SR)) / SR
        parts.append(torch.zeros_like(t) if freq == 0 else 0.5 * torch.sin(2 * math.pi * freq * t))
    return torch.cat(parts)[None]


def run(src, out, window=4.0, hop=2.0, offset=0.0, encoder=None, encoder_id=ENCODER_ID):
    return rolling_speaker_similarity(encoder or MockSpeakerEncoder(), encoder_id, src, out, window, hop, offset)


# ---------------------------------------------------------------------------
# Rolling evaluator integration
# ---------------------------------------------------------------------------


def test_identical_audio_scores_one_on_the_default_schedule():
    src = tones((220, 12))
    r = run(src, src.clone())
    spans = [(w["window_start_seconds"], w["window_end_seconds"]) for w in r["windows"]]
    assert spans == [(0, 4), (2, 6), (4, 8), (6, 10), (8, 12)]
    assert all(w["status"] == "ok" for w in r["windows"])
    assert all(w["speaker_similarity"] == pytest.approx(1.0, abs=1e-6) for w in r["windows"])
    assert r["aligned"] and r["note"] is None
    assert r["counts"] == {"ok": 5, "unavailable": 0, "error": 0}
    assert r["mean_compute_ms"] is not None and r["mean_compute_ms"] >= 0


def test_converted_voice_scores_lower_where_it_changes():
    src = tones((220, 8))
    out = tones((220, 4), (1800, 4))  # "anonymized" only in the second half
    scores = [w["speaker_similarity"] for w in run(src, out)["windows"]]
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[2] < 0.1  # 4-8 s: fully converted
    assert scores[2] < scores[1] < scores[0]


def test_output_delay_is_compensated_and_cut_tail_is_unavailable():
    # Stream recording: output = source delayed by 0.5 s, cut at the source length.
    src = tones(*[(f, 0.5) for f in (200, 700, 1300, 1900) * 4])  # 8 s, pitch changes every 0.5 s
    delay = 0.5
    out = torch.cat([torch.zeros(1, int(delay * SR)), src[:, : -int(delay * SR)]], dim=1)
    compensated = run(src, out, window=2.0, hop=1.0, offset=delay)["windows"]
    naive = run(src, out, window=2.0, hop=1.0, offset=0.0)["windows"]
    for w in compensated:
        if w["window_end_seconds"] + delay <= 8:
            assert w["status"] == "ok"
            assert w["speaker_similarity"] == pytest.approx(1.0, abs=1e-5)
        else:  # its delayed output was never recorded
            assert w["status"] == "unavailable" and w["speaker_similarity"] is None
    # Without compensation the same windows compare shifted content.
    assert naive[0]["speaker_similarity"] < compensated[0]["speaker_similarity"] - 0.05


def test_mismatched_length_is_unaligned_and_never_scored():
    src = tones((220, 8))
    out = tones((220, 12))  # e.g. a model that changes the speaking rate
    encoder = MockSpeakerEncoder()
    r = run(src, out, encoder=encoder)
    assert not r["aligned"]
    assert "Alignment unverified" in r["note"]
    assert r["windows"] and all(w["status"] == "unavailable" and w["speaker_similarity"] is None for w in r["windows"])
    assert encoder.calls == 0


@pytest.mark.parametrize(
    ("source_s", "output_s", "offset_s", "aligned"),
    [
        (10.0, 10.0, 0.0, True),
        (10.0, 10.05, 0.0, True),  # within 100 ms
        (10.0, 10.3, 0.0, False),
        (10.0, 10.3, 0.3, True),  # source + model delay
        (10.0, 10.0, 0.3, True),  # stream recording: delayed tail cut
        (60.0, 61.0, 0.0, True),  # within 2 %
        (60.0, 62.0, 0.0, False),
    ],
)
def test_alignment_tolerance(source_s, output_s, offset_s, aligned):
    assert time_aligned(source_s, output_s, offset_s) is aligned


def test_short_recording_has_no_window_and_says_why():
    src = tones((220, 3))
    r = run(src, src.clone())
    assert r["windows"] == []
    assert "shorter than the 4 s window" in r["note"]
    assert r["counts"] == {"ok": 0, "unavailable": 0, "error": 0}
    assert r["mean_compute_ms"] is None


def test_silent_window_is_an_error_not_a_score():
    src = tones((220, 4), (0, 4), (220, 4))  # 4-8 s digital silence
    windows = run(src, src.clone())["windows"]
    by_span = {(w["window_start_seconds"], w["window_end_seconds"]): w for w in windows}
    silent = by_span[(4, 8)]
    assert silent["status"] == "error" and silent["speaker_similarity"] is None
    assert "Silent" in silent["error_message"]
    assert by_span[(2, 6)]["status"] == "ok"


def test_inference_failure_is_isolated_to_its_window():
    src = tones((220, 8))
    windows = run(src, src.clone(), encoder=MockSpeakerEncoder(fail_on=2))["windows"]  # 2nd window, source side
    assert [w["status"] for w in windows] == ["ok", "error", "ok"]
    assert windows[1]["speaker_similarity"] is None
    assert "mock inference failure" in windows[1]["error_message"]


def test_result_is_json_serializable():
    src = tones((220, 6))
    r = run(src, src.clone())
    assert json.loads(json.dumps(r))["windows"][0]["window_index"] == 0


@pytest.mark.parametrize(
    ("encoder_id", "window", "hop", "offset", "message"),
    [
        (ENCODER_ID, 0.0, 1.0, 0.0, "greater than 0"),
        (ENCODER_ID, 2.0, -1.0, 0.0, "greater than 0"),
        (ENCODER_ID, 2.0, 3.0, 0.0, "Hop must not be longer"),
        (ENCODER_ID, 1 / 3, 0.1, 0.0, "whole number of samples"),
        (ENCODER_ID, 4.0, 2.0, float("nan"), "finite"),
        ("CosyVoiceCAMPPlusSpeakerEncoder", 12.0, 2.0, 0.0, "up to 10 s"),
    ],
)
def test_invalid_settings_are_rejected(encoder_id, window, hop, offset, message):
    with pytest.raises(ValueError, match=message):
        validate_rolling(encoder_id, window, hop, offset)


def test_campplus_accepts_ten_second_windows():
    validate_rolling("CosyVoiceCAMPPlusSpeakerEncoder", 10.0, 2.0, 0.0)


# ---------------------------------------------------------------------------
# /metrics/compute endpoint (models replaced by the mock; no lifespan, so nothing loads)
# ---------------------------------------------------------------------------


def wav_bytes(wav: torch.Tensor, sample_rate: int = SR) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav.reshape(-1).numpy(), sample_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch):
    model = MetricModel(ENCODER_ID, "speaker", MockSpeakerEncoder, status="ready", obj=MockSpeakerEncoder())
    monkeypatch.setattr(metrics, "discover", lambda: [model])
    return TestClient(metrics.create_metrics_app(torch.device("cpu")))


def post(client, src, out, out_rate=SR, **form):
    files = {"source": ("s.wav", wav_bytes(src), "audio/wav"), "output": ("o.wav", wav_bytes(out, out_rate), "audio/wav")}
    return client.post("/metrics/compute", files=files, data={"speaker_encoder": ENCODER_ID, **form})


def test_endpoint_disabled_by_default_and_global_score_unchanged(client):
    src, out = tones((220, 4), (440, 4)), tones((220, 4), (1800, 4))
    off = post(client, src, out).json()
    on = post(client, src, out, rolling="true").json()
    assert "speaker_similarity_over_time" not in off
    assert on["speaker_similarity"]["value"] == pytest.approx(off["speaker_similarity"]["value"], abs=1e-6)
    # Full-recording cosine, not an average of window scores.
    encoder = MockSpeakerEncoder()
    a = metrics.speaker_embedding(encoder, src)
    b = metrics.speaker_embedding(encoder, out)
    assert off["speaker_similarity"]["value"] == pytest.approx(float(F.cosine_similarity(a, b, dim=-1)), abs=1e-4)
    rolling = on["speaker_similarity_over_time"]
    assert rolling["window_s"] == 4.0 and rolling["hop_s"] == 2.0 and len(rolling["windows"]) == 3


def test_endpoint_passes_window_hop_and_offset(client):
    src = tones((220, 6))
    r = post(client, src, src, rolling="true", rolling_window_s="2", rolling_hop_s="1", offset_s="0.25").json()
    rolling = r["speaker_similarity_over_time"]
    assert (rolling["window_s"], rolling["hop_s"], rolling["offset_s"]) == (2.0, 1.0, 0.25)
    assert len(rolling["windows"]) == 5


def test_endpoint_resamples_a_different_output_rate(client):
    src = tones((220, 6))
    out = torch.from_numpy(np.interp(np.arange(6 * 24000) / 24000, np.arange(6 * SR) / SR, src.reshape(-1).numpy()))
    rolling = post(client, src, out.float()[None], out_rate=24000, rolling="true").json()["speaker_similarity_over_time"]
    assert rolling["aligned"]
    assert all(w["status"] == "ok" and w["speaker_similarity"] > 0.99 for w in rolling["windows"])


@pytest.mark.parametrize(
    "form",
    [
        {"rolling": "true", "rolling_window_s": "2", "rolling_hop_s": "3"},
        {"rolling": "true", "rolling_window_s": "0"},
        {"rolling": "true", "speaker_encoder": ""},
    ],
)
def test_endpoint_rejects_bad_rolling_requests(client, form):
    src = tones((220, 6))
    response = post(client, src, src, **form)
    assert response.status_code == 400
