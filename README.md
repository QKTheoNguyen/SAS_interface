# SAS Interface

A local web app for trying out speaker anonymization (SAS) and voice conversion (VC) models.
You pick a task and a mode, record or upload some speech, adjust the model's settings, and get back audio you can play or download.

## Models

| Task | Model | Non-streaming | Streaming | Worker |
|---|---|---|---|---|
| Anonymization | ASR-BN (SA-toolkit, via [quick-convert](https://github.com/benluks/quick-convert)) | ✅ | — | `quickconvert` |
| Anonymization | [StreamVoiceAnon](https://github.com/Plachtaa/StreamVoiceAnon) | ✅ | ✅ | `streamvoiceanon` (CUDA only) |
| Voice conversion | StreamVoiceAnon | ✅ | ✅ | `streamvoiceanon` (CUDA only) |
| Voice conversion | [kNN-VC](https://github.com/bshall/knn-vc) | ✅ | — | `quickconvert` |

For anonymization, StreamVoiceAnon picks from a bundled pool of 10 LibriSpeech `train-clean-360` pseudo-speakers
([models/streamvoiceanon/references/](models/streamvoiceanon/references/README.md)) unless you upload your own reference.
Its context holds about 45 s of source and reference audio combined.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and git. A CUDA GPU is recommended. CPU works, just slowly.

```bash
git clone --recursive <this repo> SAS_interface
cd SAS_interface
./setup.sh      # environments + checkpoints (~5 GB per worker)
./run.sh        # then open http://localhost:8000
```

Browsers only allow the microphone on `http://localhost` or HTTPS, so open the page there if you want to record.
If the app runs on a remote machine, forward the port with `ssh -L 8000:localhost:8000 <host>`.

## Metrics (anonymization)

After an anonymization (non-streaming, or once a stream is stopped), the **Metrics** panel under the result compares source and output:

| Metric | Models | Reading |
|---|---|---|
| Speaker similarity: cosine between speaker embeddings | every `SpeakerEncoder` exported by `quick_convert.components.speaker.speaker_encoders` (CAMPPlus by default, ESPnet WavLM-ECAPA) | lower = more private |
| Emotion distance (MED): Euclidean distance between SER emotion vectors | every `SERSystem` in `quick_convert.systems.ser.__all__` (Odyssey WavLM) | lower = emotion better preserved |

The `metrics` worker discovers these classes from quick-convert at start-up, so new encoders or SER models added
there appear in the dropdowns automatically. A model whose quick-convert extra is not installed is listed as
unavailable with the missing package; add the extra to [`workers/metrics/pyproject.toml`](workers/metrics/pyproject.toml).
Installing that environment builds `pyworld` (an ESPnet dependency), which needs a C++ compiler.

## Checking an install

While `./run.sh` is up, run `uv run --project gateway python scripts/smoke_test.py`. It converts a bundled clip with
every model and task, using the manifest defaults, and writes the outputs to `logs/smoke/`.

## Architecture

```
browser ──HTTP──▶ gateway (FastAPI, serves the UI) ──HTTP──▶ worker "quickconvert"    (torch 2.6: ASR-BN, kNN-VC)
                                                    ├──────▶ worker "streamvoiceanon" (torch 2.4: StreamVoiceAnon)
                                                    └──────▶ worker "metrics"         (speaker encoders + SER)
```

The gateway (`gateway/`) serves the plain HTML/JS UI and routes requests to workers. It has no ML dependencies.

Each worker (`workers/<name>/`) is its own process with its own `uv` environment, covering a group of models.
They're split up because the model repos pin incompatible PyTorch versions. Code shared between workers lives in `workers/common/`.

Each model (`models/<name>/`) has a `manifest.yaml` that describes it and an `adapter.py` that runs it. The UI is built from the manifests.

`config.yaml` lists which workers are enabled and where they listen. If a worker URL points to another host, such as a GPU node,
the gateway uses it as is and doesn't start that worker locally.

The browser converts every recording and upload to mono WAV before sending it, so the server doesn't need ffmpeg.

## Adding a model

1. Create `models/<name>/manifest.yaml`:

   ```yaml
   name: mymodel
   label: My model
   worker: quickconvert          # which worker environment runs it
   modes: [non-streaming]
   tasks:
     anonymization:
       target_voice: none        # none | optional | required
       presets:                  # optional; fills params when picked in the UI
         - name: Strong
           values: {strength: 0.9}
           default: true
       params:
         - name: strength
           label: Strength
           type: float           # choice | float | int | bool
           min: 0
           max: 1
           default: 0.5
           advanced: false       # true puts it under "Advanced"
   ```

   A `choice` parameter takes either a fixed `choices: [...]` list or `choices_from: worker`, which gets the options from the adapter's `choices()` method.

2. Create `models/<name>/adapter.py` with a class named `Adapter` that subclasses `sas_worker.ModelAdapter` and implements `load(device)` and `convert(task, source, targets, params)`.
   Audio comes in as a mono `[1, T]` float tensor at the adapter's `sample_rate`. See [`models/asrbn/adapter.py`](models/asrbn/adapter.py).

3. If the model needs a new environment, add `workers/<worker>/pyproject.toml` (see [`workers/quickconvert/`](workers/quickconvert/pyproject.toml)) and a URL for it in `config.yaml`.

4. Run `./setup.sh` again so it downloads the new checkpoints.

## Worker API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Device and per-model status (`pending`, `loading`, `ready`, `error`) |
| GET | `/models/{name}/choices/{param}` | Values for a `choices_from: worker` parameter |
| POST | `/models/{name}/convert` | Multipart: `task`, `params` (JSON), `source` (WAV), `targets` (WAV, repeatable). Returns WAV; header `X-Compute-Seconds` |
| WebSocket | `/models/{name}/stream` | Live conversion, see below |

The gateway exposes the same routes under `/api/models/...`, plus `/api/models` (manifests and status) and `/api/config`.

### Streaming protocol

1. Client sends text `{"type": "start", "task", "params", "n_targets"}`, then `n_targets` binary messages (WAV/FLAC files).
2. Worker replies text `{"type": "ready", "sample_rate", "output_sample_rate", "chunk_samples", "algorithmic_latency_ms"}`.
3. Client sends binary frames: `uint32 seq` + float32 PCM at `sample_rate` (little-endian). The worker cuts them into `chunk_samples` chunks.
4. For each chunk the worker sends binary: `uint32 seq` (last input frame consumed) + `float32 compute_ms` + float32 PCM at `output_sample_rate`.
5. Client sends text `{"type": "stop"}`; the worker answers `{"type": "end"}` after the last chunk and closes. Errors arrive as `{"type": "error", "detail"}`.

While a stream is open, that model refuses other requests (503), because streaming state lives in the model.
All model calls of a worker run on one thread, so compiled CUDA graphs are replayed from the thread that recorded them.

To make a model streamable, list `streaming` in its manifest `modes` and implement `open_stream()` returning a
`sas_worker.StreamSession` (`process(chunk)` per chunk, `close()` at the end). Parameters can be limited to one mode
with `modes: [streaming]` in their manifest entry.

## License

The code in this repository is released under the [MIT License](LICENSE).

Third-party parts keep their own licenses:

- `third_party/StreamVoiceAnon`: Apache 2.0 (git submodule, not redistributed here).
- `third_party/quick-convert`: see that repository (git submodule, not redistributed here).
- Model checkpoints are downloaded from their authors at setup time and follow their licenses.
- `models/streamvoiceanon/references/*.flac`: excerpts of LibriSpeech, CC BY 4.0 ([attribution](models/streamvoiceanon/references/README.md)).
