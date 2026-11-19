"""Build the pseudo-speaker reference pool used by StreamVoiceAnon and TVTSyn anonymization.

For every speaker of a LibriSpeech split, writes a ~10 s candidate clip (several
utterances of one chapter, concatenated) and measures two eGeMAPS features:

- pitch: median F0 (semitones from 27.5 Hz), voiced frames
- brightness: minus the Hammarberg index (dB), voiced frames. The index is the
  0-2 kHz peak over the 2-5 kHz peak, so a lower index means more high-frequency
  energy; flipping the sign makes "up" brighter.

Candidates with unreliable measurements are dropped first: opensmile and
pyworld (harvest) must agree on the median F0 within ``--max-f0-disagreement``
semitones, and that F0 must lie in a plausible range for the speaker's gender
(``PLAUSIBLE_F0_HZ``). Outside it, both trackers usually locked onto mains hum
or noise rather than the voice.

The remaining features are z-scored within each gender, then farthest-point sampling keeps
``--per-gender`` speakers spread over that 2D plane. The kept clips are written as
``<gender>_<speaker>.flac`` next to ``speakers.json`` (coordinates for the UI map).
Default split is train-clean-100: its speakers never overlap with the dev/test
speakers used in VoicePrivacy evaluations.

opensmile lives in the metrics worker environment:

    uv run --project workers/metrics python scripts/build_reference_pool.py \
        --librispeech data/raw/LibriSpeech
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
GAP_SECONDS = 0.25
FADE_SECONDS = 0.05
PLAUSIBLE_F0_HZ = {"F": (140.0, 300.0), "M": (70.0, 200.0)}
PITCH = "F0semitoneFrom27.5Hz_sma3nz_percentile50.0"
HAMMARBERG = "hammarbergIndexV_sma3nz_amean"


def read_speakers(librispeech: Path, split: str) -> dict[str, str]:
    """Speaker id -> gender ("F" / "M") for the speakers of ``split`` present on disk."""
    genders = {}
    for line in (librispeech / "SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";"):
            continue
        fields = [f.strip() for f in line.split("|")]
        if len(fields) >= 3 and fields[2] == split and fields[1] in ("F", "M"):
            if (librispeech / split / fields[0]).is_dir():
                genders[fields[0]] = fields[1]
    return genders


def build_clip(speaker_dir: Path, seconds: float) -> tuple[np.ndarray, int]:
    """First utterances of the speaker's first chapter, cropped to ``seconds`` with a short fade-out."""
    chapter = sorted(p for p in speaker_dir.iterdir() if p.is_dir())[0]
    pieces, total, sample_rate = [], 0.0, 16000
    for flac in sorted(chapter.glob("*.flac")):
        audio, sample_rate = sf.read(flac, dtype="float32")
        pieces += [audio, np.zeros(int(GAP_SECONDS * sample_rate), dtype="float32")]
        total += len(audio) / sample_rate
        if total >= seconds:
            break
    audio = np.concatenate(pieces[:-1])[: int(seconds * sample_rate)]
    fade = min(len(audio), int(FADE_SECONDS * sample_rate))
    audio[len(audio) - fade:] *= np.linspace(1.0, 0.0, fade, dtype="float32")
    return audio, sample_rate


def harvest_stats(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Median F0 in semitones from 27.5 Hz, from pyworld's harvest."""
    import pyworld as pw

    f0, _ = pw.harvest(audio.astype(np.float64), sample_rate, f0_floor=50, f0_ceil=500, frame_period=10)
    voiced = f0[f0 > 0]
    return float(12 * np.log2(np.median(voiced) / 27.5)) if len(voiced) else float("nan")


def measure(job: tuple[Path, str, str, float]) -> dict:
    speaker_dir, speaker, gender, seconds = job
    audio, sample_rate = build_clip(speaker_dir, seconds)
    features = smile().process_signal(audio, sample_rate).iloc[0]
    harvest_st = harvest_stats(audio, sample_rate)
    return {
        "speaker": speaker, "gender": gender, "audio": audio, "sample_rate": sample_rate,
        "pitch_st": float(features[PITCH]), "brightness_db": -float(features[HAMMARBERG]),
        "harvest_st": harvest_st,
    }


@functools.cache
def smile():
    import opensmile

    return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)


def farthest_points(points: np.ndarray, k: int) -> list[int]:
    """Greedy farthest-point sampling, starting from the point closest to the centre."""
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(0), axis=1)))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < min(k, len(points)):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--librispeech", type=Path, required=True, help="Directory containing SPEAKERS.TXT and the split")
    parser.add_argument("--split", default="train-clean-100")
    parser.add_argument("--per-gender", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--max-f0-disagreement", type=float, default=2.0, help="Semitones, opensmile vs harvest")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models/streamvoiceanon/references")
    args = parser.parse_args()

    genders = read_speakers(args.librispeech, args.split)
    if not genders:
        raise SystemExit(f"No {args.split} speakers found under {args.librispeech}")

    jobs = [(args.librispeech / args.split / spk, spk, g, args.seconds) for spk, g in sorted(genders.items())]
    candidates = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        for i, c in enumerate(pool.map(measure, jobs), 1):
            reasons = []
            if not abs(c["pitch_st"] - c["harvest_st"]) <= args.max_f0_disagreement:
                reasons.append(f"F0 trackers disagree ({c['pitch_st'] - c['harvest_st']:+.1f} st)")
            lo, hi = PLAUSIBLE_F0_HZ[c["gender"]]
            f0_hz = 27.5 * 2 ** (c["pitch_st"] / 12)
            if not lo <= f0_hz <= hi:
                reasons.append(f"median F0 {f0_hz:.0f} Hz outside {lo:.0f}-{hi:.0f} Hz")
            if reasons:
                print(f"\rskip {c['gender']}_{c['speaker']}: {', '.join(reasons)}")
            else:
                candidates.append(c)
            print(f"\r[{i}/{len(jobs)}] measured", end="", flush=True)
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    for old in [*args.out.glob("*.flac"), args.out / "speakers.json"]:
        old.unlink(missing_ok=True)

    kept = []
    for gender in ("F", "M"):
        group = [c for c in candidates if c["gender"] == gender]
        raw = np.array([[c["pitch_st"], c["brightness_db"]] for c in group])
        z = (raw - raw.mean(0)) / raw.std(0)
        for idx in farthest_points(z, args.per_gender):
            c = group[idx]
            name = f"{gender}_{c['speaker']}"
            sf.write(args.out / f"{name}.flac", c["audio"], c["sample_rate"])
            kept.append({
                "id": name,
                "gender": gender,
                "x": round(float(z[idx, 0]), 3),
                "y": round(float(z[idx, 1]), 3),
                "f0_hz": round(27.5 * 2 ** (c["pitch_st"] / 12), 1),
                "brightness_db": round(c["brightness_db"], 2),
                "seconds": round(len(c["audio"]) / c["sample_rate"], 1),
            })
        print(f"{gender}: kept {sum(k['gender'] == gender for k in kept)} of {len(group)} speakers")

    kept.sort(key=lambda k: k["id"])
    meta = {
        "source": f"LibriSpeech {args.split}",
        "axes": {
            "x": "Median F0, z-scored within gender",
            "y": "Brightness (minus Hammarberg index), z-scored within gender",
        },
        "speakers": kept,
    }
    (args.out / "speakers.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"Wrote {len(kept)} clips and speakers.json to {args.out}")


if __name__ == "__main__":
    main()
