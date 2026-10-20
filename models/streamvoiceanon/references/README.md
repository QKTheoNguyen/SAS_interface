# Pseudo-speaker reference pool

Default reference voices for StreamVoiceAnon anonymization (used when no reference clip is uploaded).

- Source: LibriSpeech `train-clean-360` (Panayotov et al., 2015), licensed CC BY 4.0, https://www.openslr.org/12
- File name: `<gender>_<LibriSpeech speaker id>.flac`, 16 kHz mono, ~10–17 s (consecutive utterances of one chapter)
- Rebuilt with `scripts/build_reference_pool.py` (seed 0)

`train-clean-360` speakers do not overlap with the LibriSpeech dev/test speakers used in VoicePrivacy Challenge evaluations.
