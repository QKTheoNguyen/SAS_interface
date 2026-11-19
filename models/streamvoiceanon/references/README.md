# Pseudo-speaker reference pool

Default reference voices for StreamVoiceAnon and TVTSyn anonymization (used when no reference clip is uploaded).
The clips are not in git: build them locally (needs LibriSpeech train-clean-100, ~6.3 GB from https://www.openslr.org/12):

    uv run --project workers/metrics python scripts/build_reference_pool.py --librispeech data/raw/LibriSpeech

- Source: LibriSpeech `train-clean-100` (Panayotov et al., 2015), licensed CC BY 4.0
- 100 speakers (50 F / 50 M), picked by farthest-point sampling over pitch × brightness (z-scored within gender),
  after dropping speakers with unreliable F0 (see the script docstring)
- File name: `<gender>_<LibriSpeech speaker id>.flac`, 16 kHz mono, 10 s (consecutive utterances of one chapter)
- `speakers.json`: map coordinates and raw features, read by the gateway for the UI speaker map

`train-clean-100` speakers do not overlap with the LibriSpeech dev/test speakers used in VoicePrivacy Challenge evaluations.
