"""Build the pseudo-speaker reference pool used by StreamVoiceAnon anonymization.

Picks gender-balanced speakers from a LibriSpeech split and writes one ~10 s
16 kHz FLAC per speaker (several utterances of one chapter, concatenated).
Default split is train-clean-360 so pool speakers never overlap with the
dev/test speakers used in VoicePrivacy evaluations.

    uv run --project gateway python scripts/build_reference_pool.py \
        --librispeech /path/to/LibriSpeech --split train-clean-360
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
GAP_SECONDS = 0.25


def read_speakers(librispeech: Path, split: str) -> dict[str, list[str]]:
    by_gender: dict[str, list[str]] = {"F": [], "M": []}
    for line in (librispeech / "SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";"):
            continue
        fields = [f.strip() for f in line.split("|")]
        if len(fields) >= 3 and fields[2] == split and fields[1] in by_gender:
            by_gender[fields[1]].append(fields[0])
    return by_gender


def build_clip(speaker_dir: Path, seconds: float) -> tuple[np.ndarray, int]:
    chapter = sorted(p for p in speaker_dir.iterdir() if p.is_dir())[0]
    pieces, total, sample_rate = [], 0.0, 16000
    for flac in sorted(chapter.glob("*.flac")):
        audio, sample_rate = sf.read(flac, dtype="float32")
        pieces += [audio, np.zeros(int(GAP_SECONDS * sample_rate), dtype="float32")]
        total += len(audio) / sample_rate
        if total >= seconds:
            break
    return np.concatenate(pieces[:-1]), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--librispeech", type=Path, required=True, help="Directory containing SPEAKERS.TXT and the split")
    parser.add_argument("--split", default="train-clean-360")
    parser.add_argument("--per-gender", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models/streamvoiceanon/references")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for gender, speakers in read_speakers(args.librispeech, args.split).items():
        available = [s for s in speakers if (args.librispeech / args.split / s).is_dir()]
        for speaker in sorted(rng.sample(available, args.per_gender)):
            audio, sample_rate = build_clip(args.librispeech / args.split / speaker, args.seconds)
            path = args.out / f"{gender}_{speaker}.flac"
            sf.write(path, audio, sample_rate)
            print(f"{path.name}: {len(audio) / sample_rate:.1f} s")


if __name__ == "__main__":
    main()
