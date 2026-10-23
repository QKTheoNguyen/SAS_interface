import { fileToWav, formatSeconds, startRecording } from "./audio.js";
import { LiveStream } from "./stream.js";

const $ = (id) => document.getElementById(id);

const TASK_LABELS = { anonymization: "anonymization", voice_conversion: "voice conversion" };

const state = {
  config: { max_input_seconds: 60 },
  models: [],
  task: "anonymization",
  mode: "non-streaming",
  modelName: null,
  shownStatus: null, // status of the selected model when its settings were last rendered
  source: null, // {blob, duration, name}
  targets: [], // [{input, value}]
  params: {},
  choicesCache: {},
  streamSource: "mic", // "mic" | "file"
  streamFile: null,
  stream: null, // active LiveStream
  lastResult: null, // {original, output, task} of the result shown, for metrics
  metricsPoll: null,
  statsTimer: null,
};

// ---------------------------------------------------------------------------
// Audio input widget: record or upload one clip.
// ---------------------------------------------------------------------------

let audioInputCount = 0;

/**
 * Record or upload one clip. Like the streaming source, a full-width toggle picks
 * the method ("Record" / "Upload file") and only that method's control is shown.
 */
function createAudioInput({ onChange, onRemove } = {}) {
  const root = document.createElement("div");
  root.className = "clip";
  const radioName = `clip-source-${++audioInputCount}`;
  const fileInput = Object.assign(document.createElement("input"), { type: "file", accept: "audio/*", hidden: true });
  let method = "record"; // "record" | "upload"
  let value = null;
  let recorder = null;
  let stopping = false;
  let timer = null;
  let message = "";

  const set = (v) => {
    value = v;
    onChange?.(v);
    render();
  };

  const accept = (clip) => {
    if (clip.duration > state.config.max_input_seconds) {
      message = `Clip is ${formatSeconds(clip.duration)}; the limit is ${state.config.max_input_seconds} s.`;
      set(null);
    } else {
      message = "";
      set(clip);
    }
  };

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    fileInput.value = "";
    if (!file) return;
    try {
      accept({ ...(await fileToWav(file)), name: file.name });
    } catch (err) {
      message = err.message;
      render();
    }
  });

  async function toggleRecording() {
    if (recorder) {
      if (stopping) return;
      stopping = true;
      clearInterval(timer);
      const clip = await recorder.stop();
      recorder = null;
      stopping = false;
      accept({ ...clip, name: "recording" });
      return;
    }
    try {
      recorder = await startRecording();
      message = "";
      timer = setInterval(render, 200);
      render();
    } catch (err) {
      recorder = null;
      message = err.message;
      render();
    }
  }

  function button(text, onClick, cls = "secondary") {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = text;
    b.addEventListener("click", onClick);
    return b;
  }

  function methodToggle() {
    const group = document.createElement("div");
    group.className = "segmented";
    for (const [id, label] of [["record", "Record"], ["upload", "Upload file"]]) {
      const input = Object.assign(document.createElement("input"), {
        type: "radio", name: radioName, value: id, checked: method === id, disabled: !!recorder,
      });
      input.addEventListener("change", () => {
        method = id;
        render();
      });
      const lab = document.createElement("label");
      lab.append(input, label);
      group.append(lab);
    }
    return group;
  }

  function render() {
    root.replaceChildren(fileInput, methodToggle());
    const row = document.createElement("div");
    row.className = "row";
    if (recorder) {
      row.append(button(`■ Stop (${formatSeconds(recorder.elapsed())})`, toggleRecording, "recording"));
      if (recorder.elapsed() > state.config.max_input_seconds) toggleRecording();
    } else if (method === "record") {
      row.append(button(value ? "● Record again" : "● Start recording", toggleRecording));
    } else {
      row.append(button(value ? "Choose another file" : "Choose file", () => fileInput.click()));
    }
    if (!recorder) {
      const meta = document.createElement("span");
      meta.className = "muted";
      meta.textContent = value
        ? `${value.name} · ${formatSeconds(value.duration)}`
        : method === "upload" ? "No file chosen" : "";
      if (meta.textContent) row.append(meta);
      if (onRemove) {
        const remove = button("Remove clip", onRemove);
        remove.classList.add("push-right");
        row.append(remove);
      }
    }
    root.append(row);
    if (value && !recorder) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = URL.createObjectURL(value.blob);
      root.append(audio);
    }
    if (message) {
      const p = document.createElement("p");
      p.className = "status error";
      p.textContent = message;
      root.append(p);
    }
  }

  render();
  return { root, get value() { return value; } };
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

const compatibleModels = () =>
  state.models.filter((m) => m.tasks[state.task] && m.modes.includes(state.mode));

const selectedModel = () => state.models.find((m) => m.name === state.modelName);

function renderModels() {
  const models = compatibleModels();
  const select = $("model-select");
  if (!models.some((m) => m.name === state.modelName)) state.modelName = models[0]?.name ?? null;

  select.replaceChildren(
    ...models.map((m) => new Option(`${m.label} (${m.status})`, m.name, false, m.name === state.modelName)),
  );
  select.hidden = models.length <= 1;

  if (!models.length) {
    $("model-info").innerHTML = "";
    const p = document.createElement("p");
    p.className = "notice";
    p.textContent = `No model supports ${TASK_LABELS[state.task]} in ${state.mode} mode yet.`;
    $("model-info").append(p);
    $("model-dependent").hidden = true;
    return;
  }
  $("model-dependent").hidden = false;
  renumberSteps();
  renderModelInfo();
}

function renderModelInfo() {
  const m = selectedModel();
  const info = $("model-info");
  info.replaceChildren();

  const title = document.createElement("div");
  title.className = "row";
  const name = document.createElement("strong");
  name.textContent = m.label;
  const badge = document.createElement("span");
  badge.className = `badge ${m.status}`;
  badge.textContent = m.status + (m.device && m.status === "ready" ? ` · ${m.device}` : "");
  if (m.error) badge.title = m.error;
  title.append(name, badge);
  info.append(title);

  if (m.description) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = m.description;
    info.append(p);
  }
  if (m.links?.length) {
    const p = document.createElement("p");
    p.className = "row";
    for (const link of m.links) {
      const a = Object.assign(document.createElement("a"), { href: link.url, textContent: link.label, target: "_blank" });
      p.append(a);
    }
    info.append(p);
  }
  if (m.error) {
    const p = document.createElement("p");
    p.className = "status error";
    p.textContent = m.error;
    info.append(p);
  }
  if (m.warning) {
    const p = document.createElement("p");
    p.className = "warning";
    p.textContent = m.warning;
    info.append(p);
  }
  if (m.status === "loading" && m.modes.includes("streaming")) {
    const p = document.createElement("p");
    p.className = "notice";
    p.textContent = "Loading includes compiling the streaming kernels: ~7 min the first time, ~1 min once cached.";
    info.append(p);
  }
  updateRunButton();
}

// ---------------------------------------------------------------------------
// Settings (generated from the manifest)
// ---------------------------------------------------------------------------

async function choicesFor(model, param) {
  const options = [...(param.extra_choices || []), ...(param.choices || [])];
  if (param.choices_from !== "worker" || model.status !== "ready") return options;
  const key = `${model.name}/${param.name}`;
  if (!state.choicesCache[key]) {
    state.choicesCache[key] = fetchJson(`/api/models/${model.name}/choices/${param.name}`).then((r) => r.choices);
  }
  try {
    return [...options, ...(await state.choicesCache[key])];
  } catch (err) {
    delete state.choicesCache[key];
    return options;
  }
}

async function renderParam(model, param) {
  const wrap = document.createElement("div");
  wrap.className = "param";
  const id = `param-${param.name}`;
  const label = Object.assign(document.createElement("label"), { htmlFor: id, textContent: param.label || param.name });
  wrap.append(label);
  const value = state.params[param.name] ?? param.default;
  state.params[param.name] = value;
  const update = (v) => {
    state.params[param.name] = v;
    // Editing a value the preset controls turns the preset into "Custom".
    const preset = $("param-preset");
    const values = selectedPreset()?.values;
    if (preset && values && param.name in values && values[param.name] !== v) preset.value = "";
  };

  if (param.type === "choice") {
    const select = document.createElement("select");
    select.id = id;
    for (const c of await choicesFor(model, param)) select.add(new Option(c, c, false, String(c) === String(value)));
    select.addEventListener("change", () => update(select.value));
    wrap.append(select);
  } else if (param.type === "float" || param.type === "int") {
    const box = document.createElement("div");
    box.className = "range";
    const attrs = { min: param.min, max: param.max, step: param.step ?? (param.type === "int" ? 1 : 0.01), value };
    const range = Object.assign(document.createElement("input"), { type: "range", ...attrs });
    const number = Object.assign(document.createElement("input"), { type: "number", id, ...attrs });
    const parse = (v) => (param.type === "int" ? parseInt(v, 10) : parseFloat(v));
    range.addEventListener("input", () => update((number.value = range.value, parse(range.value))));
    number.addEventListener("input", () => update((range.value = number.value, parse(number.value))));
    box.append(range, number);
    wrap.append(box);
  } else if (param.type === "bool") {
    const box = Object.assign(document.createElement("input"), { type: "checkbox", id, checked: !!value });
    box.addEventListener("change", () => update(box.checked));
    label.prepend(box, " ");
  }

  if (param.help) {
    const help = document.createElement("div");
    help.className = "help";
    help.textContent = param.help;
    wrap.append(help);
  }
  return wrap;
}

function selectedPreset() {
  const presets = selectedModel()?.tasks[state.task]?.presets || [];
  return presets.find((p) => p.name === $("param-preset")?.value);
}

function renderPresetSelect(presets, current) {
  const wrap = document.createElement("div");
  wrap.className = "param";
  const label = Object.assign(document.createElement("label"), { htmlFor: "param-preset", textContent: "Preset" });
  const select = document.createElement("select");
  select.id = "param-preset";
  for (const p of presets) select.add(new Option(p.name, p.name, false, p.name === current));
  select.add(new Option("Custom", "", false, !current));
  const help = document.createElement("div");
  help.className = "help";
  // Read the option from this select: it is not in the page yet on first render.
  const describe = () =>
    (help.textContent = presets.find((p) => p.name === select.value)?.description || "Values edited by hand.");
  select.addEventListener("change", () => {
    const preset = selectedPreset();
    if (preset) renderSettings({ keepParams: true, preset: preset.name, apply: preset.values });
    else describe();
  });
  describe();
  wrap.append(label, select, help);
  return wrap;
}

async function renderSettings({ keepParams = false, preset, apply } = {}) {
  const m = selectedModel();
  state.shownStatus = m?.status;
  if (!keepParams) state.params = {};
  if (!m) return;
  const spec = m.tasks[state.task];
  const params = (spec.params || []).filter((p) => !p.modes || p.modes.includes(state.mode));
  const presets = spec.presets || [];
  const presetName = preset ?? presets.find((p) => p.default)?.name ?? presets[0]?.name;
  Object.assign(state.params, apply ?? (keepParams ? {} : presets.find((p) => p.name === presetName)?.values));

  const basic = await Promise.all(params.filter((p) => !p.advanced).map((p) => renderParam(m, p)));
  const advanced = await Promise.all(params.filter((p) => p.advanced).map((p) => renderParam(m, p)));
  if (presets.length) basic.unshift(renderPresetSelect(presets, presetName));
  $("params").replaceChildren(...basic);
  $("advanced-params").replaceChildren(...advanced);
  $("advanced").hidden = !advanced.length;
  $("params-card").hidden = !params.length;

  const targetVoice = spec.target_voice || "none";
  $("target-card").hidden = targetVoice === "none";
  $("target-required").textContent = targetVoice === "required" ? "(required)" : "(optional)";
  $("target-title").textContent = spec.target_voice_label || "Target voice";
  $("target-help").textContent = spec.target_voice_help || "";
  // Show an empty clip right away (record or upload), also when the voice is optional.
  if (targetVoice !== "none" && !state.targets.length) addTarget();
  renumberSteps();
}

/** Number the visible cards 1, 2, 3… in page order. */
function renumberSteps() {
  let n = 0;
  for (const step of document.querySelectorAll(".step")) {
    if (!step.closest("[hidden]")) step.textContent = String(++n);
  }
}

// ---------------------------------------------------------------------------
// Target voice clips
// ---------------------------------------------------------------------------

function renderTargets() {
  const box = $("targets");
  box.replaceChildren(...state.targets.map((t) => t.input.root));
  const spec = selectedModel()?.tasks[state.task];
  const minTotal = spec?.min_total_target_seconds;
  const total = state.targets.reduce((s, t) => s + (t.input.value?.duration || 0), 0);
  if (minTotal && total > 0 && total < minTotal) {
    const p = document.createElement("p");
    p.className = "warning";
    p.textContent = `Target audio totals ${formatSeconds(total)}. This model works best with at least ${minTotal} s.`;
    box.append(p);
  }
  updateRunButton();
}

function addTarget() {
  const entry = {};
  entry.input = createAudioInput({
    onChange: renderTargets,
    onRemove: () => {
      state.targets = state.targets.filter((t) => t !== entry);
      renderTargets();
    },
  });
  state.targets.push(entry);
  renderTargets();
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

function updateRunButton() {
  const m = selectedModel();
  const spec = m?.tasks[state.task];
  const needsTarget = spec?.target_voice === "required";
  const hasTarget = state.targets.some((t) => t.input.value);
  const streaming = state.mode === "streaming";
  $("source-input").hidden = streaming;
  $("stream-source").hidden = !streaming;
  $("stream-file-row").hidden = state.streamSource !== "file";
  $("run").textContent = state.stream ? "■ Stop streaming" : streaming ? "● Start streaming" : "Run";
  $("run").classList.toggle("recording", !!state.stream);
  updateMetricsButton();
  if (state.stream) {
    $("run").disabled = state.stream.stopping;
    return;
  }
  const blockers = [];
  if (!m || m.status !== "ready") blockers.push(`model is ${m?.status ?? "missing"}`);
  if (!streaming && !state.source) blockers.push("add source speech");
  if (streaming && state.streamSource === "file" && !state.streamFile) blockers.push("choose a file");
  if (needsTarget && !hasTarget) blockers.push("add a target voice clip");
  $("run").disabled = blockers.length > 0;
  $("run").title = blockers.join(", ");
}

function showResult({ original, output, model, meta }) {
  const url = URL.createObjectURL(output);
  $("original-audio").src = URL.createObjectURL(original);
  $("result-audio").src = url;
  $("download").href = url;
  $("download").download = `${model.name}_${state.task}_${state.mode}_${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
  $("result-meta").textContent = meta;
  $("result-card").hidden = false;
  state.lastResult = { original, output, task: state.task };
  $("metrics-results").replaceChildren();
  $("f0-plot").hidden = true;
  $("spectrogram-plot").hidden = true;
  $("metrics-status").textContent = "";
  $("metrics-panel").hidden = !(state.config.metrics && state.task === "anonymization");
  updateMetricsButton();
  renumberSteps();
  $("result-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------------------
// Metrics (anonymization only)
// ---------------------------------------------------------------------------

function fillMetricSelect(select, models) {
  const previous = select.value;
  const options = [new Option("None (skip)", "")];
  for (const m of models) {
    const suffix = m.status === "ready" ? "" : m.status === "unavailable" ? " (unavailable)" : " (loading…)";
    const option = new Option(m.label + suffix, m.id);
    option.disabled = m.status !== "ready";
    if (m.error) option.title = m.error;
    options.push(option);
  }
  select.replaceChildren(...options);
  const ready = models.filter((m) => m.status === "ready");
  const pick = ready.find((m) => m.id === previous) || ready.find((m) => m.default) || ready[0];
  select.value = previous === "" && select.dataset.touched ? "" : pick?.id ?? "";
}

async function refreshMetricOptions() {
  try {
    const options = await fetchJson("/api/metrics/options");
    fillMetricSelect($("metric-speaker"), options.speaker_encoders);
    fillMetricSelect($("metric-ser"), options.ser_models);
    fillMetricSelect($("metric-f0"), options.f0_trackers);
    fillMetricSelect($("metric-mos"), options.mos_models);
    const all = [...options.speaker_encoders, ...options.ser_models, ...options.f0_trackers, ...options.mos_models];
    const pending = all.some((m) => ["pending", "loading"].includes(m.status));
    clearTimeout(state.metricsPoll);
    if (pending && $("metrics-panel").open) state.metricsPoll = setTimeout(refreshMetricOptions, 3000);
  } catch (err) {
    $("metrics-status").className = "status error";
    $("metrics-status").textContent = err.message;
  }
  updateMetricsButton();
}

function updateMetricsButton() {
  const chosen =
    $("metric-speaker").value || $("metric-ser").value || $("metric-f0").value || $("metric-mos").value || $("metric-spectrogram").checked;
  // Metrics need a finished result: a live stream must be stopped first.
  $("compute-metrics").disabled = !state.lastResult || !!state.stream || !chosen;
  $("compute-metrics").title = state.stream ? "Stop the stream first" : chosen ? "" : "Pick at least one model or the spectrograms";
}

function metricTile(label, value, sub, headline) {
  const div = document.createElement("div");
  if (headline) div.append(Object.assign(document.createElement("div"), { className: "headline", textContent: headline }));
  const dt = Object.assign(document.createElement("dt"), { textContent: label });
  const dd = Object.assign(document.createElement("dd"), { textContent: value });
  const small = Object.assign(document.createElement("div"), { className: "sub", textContent: sub });
  div.append(dt, dd, small);
  div.title = sub;
  return div;
}

async function computeMetrics() {
  const status = $("metrics-status");
  const form = new FormData();
  form.append("source", state.lastResult.original, "source.wav");
  form.append("output", state.lastResult.output, "output.wav");
  form.append("speaker_encoder", $("metric-speaker").value);
  form.append("ser_model", $("metric-ser").value);
  form.append("f0_tracker", $("metric-f0").value);
  form.append("mos_model", $("metric-mos").value);
  form.append("spectrogram", $("metric-spectrogram").checked);
  status.className = "status";
  status.textContent = "Computing…";
  $("compute-metrics").disabled = true;
  try {
    const res = await fetch("/api/metrics/compute", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `${res.status} ${res.statusText}`);
    const r = await res.json();
    const tiles = [];
    if (r.speaker_similarity) {
      const tile = metricTile("Speaker similarity (cosine)", r.speaker_similarity.value.toFixed(3), "↓ lower = more private");
      tile.title = r.speaker_similarity.model;
      tiles.push(tile);
    }
    if (r.emotion_distance) {
      const e = r.emotion_distance;
      const change = e.source_emotion ? `${e.source_emotion} → ${e.output_emotion}` : "";
      const tile = metricTile("Emotion distance (MED)", e.value.toFixed(3), "↓ lower = emotion better preserved", change);
      tile.title = e.model;
      tiles.push(tile);
    }
    if (r.mos) {
      const tile = metricTile("Naturalness (UTMOS)", r.mos.output.toFixed(2), `source ${r.mos.source.toFixed(2)} · ↑ higher = more natural (1–5)`);
      tile.title = r.mos.model;
      tiles.push(tile);
    }
    $("metrics-results").replaceChildren(...tiles);
    $("f0-plot").hidden = !r.f0;
    if (r.f0) renderF0Plot($("f0-plot"), r.f0);
    $("spectrogram-plot").hidden = !r.spectrogram;
    if (r.spectrogram) renderSpectrograms($("spectrogram-plot"), r.spectrogram, r.f0);
    status.textContent = `Computed in ${formatSeconds(r.seconds)}.`;
  } catch (err) {
    status.className = "status error";
    status.textContent = err.message;
  } finally {
    updateMetricsButton();
  }
}

// ~5 round-numbered ticks covering [lo, hi].
function niceTicks(lo, hi, count = 5) {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(6));
  return ticks;
}

// Index of the frame time closest to t (frame times are increasing).
function nearestFrame(times, t) {
  let lo = 0;
  let hi = times.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < t) lo = mid;
    else hi = mid;
  }
  return Math.abs(times[hi] - t) < Math.abs(times[lo] - t) ? hi : lo;
}

/**
 * Overlaid source/output F0 contours (Hz); unvoiced frames (null) are gaps.
 * The output can be shifted in time to compensate for the model's delay.
 */
function renderF0Plot(figure, f0) {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const W = 640, H = 240, L = 44, R = 10, T = 8, B = 28;
  const series = [
    { key: "source", label: "Source", shift: 0, ...f0.source },
    { key: "output", label: "Output", shift: 0, ...f0.output },
  ];
  const voiced = series.flatMap((s) => s.hz.filter((v) => v !== null));
  const tMax = Math.max(...series.map((s) => s.t.at(-1) ?? 0), 0.01);
  let yMin = voiced.length ? Math.min(...voiced) : 50;
  let yMax = voiced.length ? Math.max(...voiced) : 500;
  const pad = Math.max((yMax - yMin) * 0.1, 5);
  yMin = Math.max(0, yMin - pad);
  yMax += pad;
  const x = (t) => L + (t / tMax) * (W - L - R);
  const y = (hz) => T + (1 - (hz - yMin) / (yMax - yMin)) * (H - T - B);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `F0 contours of source and output (${f0.model})` });
  for (const v of niceTicks(yMin, yMax)) {
    svg.append(el("line", { class: "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }));
    svg.append(el("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, String(v)));
  }
  for (const v of niceTicks(0, tMax, 6)) {
    svg.append(el("text", { x: x(v), y: H - B + 16, "text-anchor": "middle" }, `${v}`));
  }
  svg.append(el("text", { x: W - R, y: H - 4, "text-anchor": "end" }, "time (s)"));
  svg.append(el("line", { class: "axis", x1: L, x2: W - R, y1: H - B, y2: H - B }));
  const clip = el("clipPath", { id: "f0-clip" });
  clip.append(el("rect", { x: L, y: 0, width: W - L - R, height: H - B }));
  svg.append(clip);
  const draw = (s) => {
    let d = "";
    let pen = false;
    s.t.forEach((t, i) => {
      if (s.hz[i] === null) return void (pen = false);
      d += `${pen ? "L" : "M"}${x(t + s.shift).toFixed(1)} ${y(s.hz[i]).toFixed(1)}`;
      pen = true;
    });
    s.path.setAttribute("d", d);
  };
  for (const s of series) {
    s.path = el("path", { class: `series ${s.key}`, "clip-path": "url(#f0-clip)" });
    draw(s);
    svg.append(s.path);
  }

  // Hover: vertical guide, a dot per voiced series, values in the legend.
  const guide = el("line", { class: "guide", y1: T, y2: H - B, visibility: "hidden" });
  const dots = series.map((s) => el("circle", { class: `dot ${s.key}`, r: 3, visibility: "hidden" }));
  svg.append(guide, ...dots);
  const legend = Object.assign(document.createElement("div"), { className: "legend" });
  const readouts = series.map((s) => {
    const item = document.createElement("span");
    item.append(Object.assign(document.createElement("span"), { className: `swatch ${s.key}` }), s.label);
    const value = Object.assign(document.createElement("span"), { className: "muted" });
    item.append(value);
    legend.append(item);
    return value;
  });
  const timeReadout = Object.assign(document.createElement("span"), { className: "muted" });
  legend.append(timeReadout);
  const clear = () => {
    guide.setAttribute("visibility", "hidden");
    dots.forEach((dot) => dot.setAttribute("visibility", "hidden"));
    readouts.forEach((r) => (r.textContent = ""));
    timeReadout.textContent = "";
  };
  svg.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const t = (((event.clientX - box.left) / box.width) * W - L) / (W - L - R) * tMax;
    if (t < 0 || t > tMax) return clear();
    guide.setAttribute("x1", x(t));
    guide.setAttribute("x2", x(t));
    guide.setAttribute("visibility", "visible");
    timeReadout.textContent = `t = ${t.toFixed(2)} s`;
    series.forEach((s, i) => {
      const local = t - s.shift;
      const hz = local < 0 || local > s.t.at(-1) ? null : s.hz[nearestFrame(s.t, local)];
      readouts[i].textContent = hz === null ? " — unvoiced" : ` ${Math.round(hz)} Hz`;
      if (hz !== null) {
        dots[i].setAttribute("cx", x(t));
        dots[i].setAttribute("cy", y(hz));
      }
      dots[i].setAttribute("visibility", hz === null ? "hidden" : "visible");
    });
  });
  svg.addEventListener("pointerleave", clear);

  const caption = Object.assign(document.createElement("figcaption"), { textContent: `F0 contour (Hz) · ${f0.model}` });
  // Manual alignment: move the output contour earlier/later (Shift+click for bigger steps).
  const output = series[1];
  const shiftReadout = Object.assign(document.createElement("span"), { className: "muted" });
  shiftReadout.setAttribute("role", "status");
  const showShift = () => (shiftReadout.textContent = `Output shift: ${output.shift >= 0 ? "+" : "−"}${Math.round(Math.abs(output.shift) * 1000)} ms`);
  const shiftButton = (text, direction) => {
    const button = Object.assign(document.createElement("button"), { type: "button", className: "secondary", textContent: text });
    button.title = `Move the output contour ${direction < 0 ? "earlier" : "later"} by 10 ms (Shift: 50 ms)`;
    button.addEventListener("click", (event) => {
      output.shift = Math.round((output.shift + direction * (event.shiftKey ? 0.05 : 0.01)) * 1000) / 1000;
      draw(output);
      showShift();
      clear();
    });
    return button;
  };
  showShift();
  const controls = Object.assign(document.createElement("div"), { className: "shift" });
  controls.append(shiftButton("◀ Backward", -1), shiftReadout, shiftButton("Forward ▶", 1));

  figure.replaceChildren(caption, svg, legend, controls);
}

// Magma colormap stops, quiet → loud (reads on both light and dark pages).
const MAGMA = ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55064", "#fb8761", "#fec287", "#fcfdbf"];
const F0_ON_SPECTROGRAM = "#5ad8ff";

// 256-entry RGB lookup table interpolated between colour stops.
function colorLut(stops) {
  const rgb = stops.map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));
  const lut = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const p = (v / 255) * (rgb.length - 1);
    const i = Math.min(Math.floor(p), rgb.length - 2);
    for (let c = 0; c < 3; c++) lut[v * 3 + c] = rgb[i][c] + (rgb[i + 1][c] - rgb[i][c]) * (p - i);
  }
  return lut;
}

/**
 * Source (top) and output (bottom) spectrograms on a shared time axis. When F0 was
 * computed, each panel overlays its own signal's contour on a right-hand Hz axis
 * (Praat convention: F0 has its own scale, not the spectrogram's).
 */
function renderSpectrograms(figure, spec, f0) {
  const W = 640, H = 200, L = 44, R = f0 ? 44 : 10, T = 8, B = 28;
  const style = getComputedStyle(figure);
  const muted = style.getPropertyValue("--muted").trim();
  const font = `11px ${style.fontFamily}`;
  const panels = [
    { key: "source", label: "Source", spec: spec.source, f0: f0?.source },
    { key: "output", label: "Output", spec: spec.output, f0: f0?.output },
  ];
  const tMax = Math.max(...panels.map((p) => p.spec.frames * p.spec.hop_s), ...panels.map((p) => p.f0?.t.at(-1) ?? 0), 0.01);
  const voiced = panels.flatMap((p) => p.f0?.hz.filter((v) => v !== null) ?? []);
  let yMin = voiced.length ? Math.min(...voiced) : 50;
  let yMax = voiced.length ? Math.max(...voiced) : 500;
  const pad = Math.max((yMax - yMin) * 0.1, 5);
  yMin = Math.max(0, yMin - pad);
  yMax += pad;
  const lut = colorLut(MAGMA);
  const x = (t) => L + (t / tMax) * (W - L - R);
  const yHz = (hz) => T + (1 - hz / panels[0].spec.fmax) * (H - T - B);
  const yF0 = (hz) => T + (1 - (hz - yMin) / (yMax - yMin)) * (H - T - B);

  const draw = (panel) => {
    const { spec: s } = panel;
    const bytes = Uint8Array.from(atob(s.data), (c) => c.charCodeAt(0));
    const image = new ImageData(s.frames, s.bins);
    for (let f = 0; f < s.frames; f++) {
      for (let b = 0; b < s.bins; b++) {
        const v = bytes[f * s.bins + b];
        const px = ((s.bins - 1 - b) * s.frames + f) * 4;
        image.data[px] = lut[v * 3];
        image.data[px + 1] = lut[v * 3 + 1];
        image.data[px + 2] = lut[v * 3 + 2];
        image.data[px + 3] = 255;
      }
    }
    const heatmap = Object.assign(document.createElement("canvas"), { width: s.frames, height: s.bins });
    heatmap.getContext("2d").putImageData(image, 0, 0);

    const dpr = window.devicePixelRatio || 1;
    const canvas = Object.assign(document.createElement("canvas"), { width: W * dpr, height: H * dpr });
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${panel.label} spectrogram${panel.f0 ? " with F0 contour" : ""}`);
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.drawImage(heatmap, L, T, x(s.frames * s.hop_s) - L, H - T - B);

    ctx.font = font;
    ctx.fillStyle = muted;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const v of niceTicks(0, s.fmax / 1000)) ctx.fillText(String(v), L - 6, yHz(v * 1000));
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.fillText("kHz", 4, T + 4);
    ctx.textAlign = "center";
    for (const v of niceTicks(0, tMax, 6)) ctx.fillText(String(v), x(v), H - B + 16);
    ctx.textAlign = "right";
    ctx.fillText("time (s)", W - R, H - 4);

    if (panel.f0) {
      ctx.fillStyle = F0_ON_SPECTROGRAM;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      for (const v of niceTicks(yMin, yMax)) ctx.fillText(String(v), W - R + 6, yF0(v));
      ctx.textBaseline = "alphabetic";
      ctx.fillText("F0 Hz", W - R + 2, T + 4);
      ctx.save();
      ctx.beginPath();
      ctx.rect(L, T, W - L - R, H - T - B);
      ctx.clip();
      ctx.beginPath();
      let pen = false;
      panel.f0.t.forEach((t, i) => {
        const hz = panel.f0.hz[i];
        if (hz === null) return void (pen = false);
        ctx[pen ? "lineTo" : "moveTo"](x(t), yF0(hz));
        pen = true;
      });
      ctx.lineJoin = "round";
      ctx.strokeStyle = "rgba(0, 0, 0, 0.75)";
      ctx.lineWidth = 3.5;
      ctx.stroke();
      ctx.strokeStyle = F0_ON_SPECTROGRAM;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }

    // Hover readout: time, frequency under the cursor, and F0 of this signal.
    const readout = Object.assign(document.createElement("span"), { className: "muted" });
    canvas.addEventListener("pointermove", (event) => {
      const box = canvas.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * W;
      const py = ((event.clientY - box.top) / box.height) * H;
      if (px < L || px > W - R || py < T || py > H - B) return void (readout.textContent = "");
      const t = ((px - L) / (W - L - R)) * tMax;
      const hz = (1 - (py - T) / (H - T - B)) * s.fmax;
      let text = `t = ${t.toFixed(2)} s · ${Math.round(hz)} Hz`;
      if (panel.f0 && t <= panel.f0.t.at(-1)) {
        const value = panel.f0.hz[nearestFrame(panel.f0.t, t)];
        text += value === null ? " · F0 unvoiced" : ` · F0 ${Math.round(value)} Hz`;
      }
      readout.textContent = text;
    });
    canvas.addEventListener("pointerleave", () => (readout.textContent = ""));

    const head = Object.assign(document.createElement("div"), { className: "spec-head" });
    head.append(Object.assign(document.createElement("span"), { textContent: panel.label }), readout);
    return [head, canvas];
  };

  const { fmax, range_db: rangeDb } = panels[0].spec;
  const caption = Object.assign(document.createElement("figcaption"), {
    textContent: `Spectrogram (0–${fmax / 1000} kHz, 25 ms window, ${rangeDb} dB range)${f0 ? ` · F0: ${f0.model}` : ""}`,
  });
  figure.replaceChildren(caption, ...panels.flatMap(draw));
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

function renderLatency(stats) {
  const ms = (v) => `${Math.round(v)} ms`;
  const items = [
    ["End-to-end (estimate)", ms(stats.totalMs), "total"],
    ["Model delay", ms(stats.algorithmicMs)],
    ["Chunk capture", ms(stats.chunkMs)],
    ["Round trip", ms(stats.roundTripMs)],
    ["Compute / chunk", `${stats.computeMs.toFixed(1)} ms`],
    ["RTF", stats.rtf.toFixed(2), stats.rtf > 1 ? "bad" : ""],
    ["Playback buffer", ms(stats.bufferMs)],
    ["Audio device", ms(stats.deviceMs)],
    ["Underruns", String(stats.underruns), stats.underruns ? "bad" : ""],
    ["Streamed", formatSeconds(stats.sentSeconds)],
  ];
  $("latency-stats").replaceChildren(
    ...items.map(([label, value, cls]) => {
      const div = document.createElement("div");
      const dt = Object.assign(document.createElement("dt"), { textContent: label });
      const dd = Object.assign(document.createElement("dd"), { textContent: value, className: cls || "" });
      div.append(dt, dd);
      return div;
    }),
  );
  $("latency-card").hidden = false;
}

async function startStream() {
  const m = selectedModel();
  const status = $("run-status");
  status.className = "status";
  status.textContent = "Connecting…";
  const stream = new LiveStream({
    model: m.name,
    task: state.task,
    params: state.params,
    targets: state.targets.filter((t) => t.input.value).map((t) => t.input.value.blob),
    file: state.streamSource === "file" ? state.streamFile : null,
    monitor: $("stream-monitor").checked,
    maxSeconds: state.config.max_stream_seconds,
    onEnd: (result) => endStream(m, result),
  });
  state.stream = stream;
  updateRunButton();
  try {
    await stream.start();
    status.textContent = stream.opts.file ? `Streaming ${stream.opts.file.name}…` : "Streaming from the microphone…";
    state.statsTimer = setInterval(() => renderLatency(stream.stats), 250);
  } catch (err) {
    stream.finish(err);
  }
  updateRunButton();
}

function stopStream() {
  state.stream?.stop();
  $("run-status").textContent = "Stopping (flushing the model's delay)…";
  updateRunButton();
}

function endStream(model, result) {
  clearInterval(state.statsTimer);
  state.stream = null;
  const status = $("run-status");
  status.className = result.error ? "status error" : "status";
  status.textContent = result.error ? result.error.message : "";
  if (result.stats) {
    renderLatency(result.stats);
    const s = result.stats;
    showResult({
      original: result.input,
      output: result.output,
      model,
      meta: `${model.label} · streamed ${formatSeconds(s.sentSeconds)} · end-to-end ≈ ${Math.round(s.totalMs)} ms · RTF ${s.rtf.toFixed(2)}`,
    });
  }
  updateRunButton();
}

async function run() {
  if (state.mode === "streaming") return state.stream ? stopStream() : startStream();
  const m = selectedModel();
  const status = $("run-status");
  const form = new FormData();
  form.append("task", state.task);
  form.append("params", JSON.stringify(state.params));
  form.append("source", state.source.blob, "source.wav");
  state.targets.filter((t) => t.input.value).forEach((t, i) => form.append("targets", t.input.value.blob, `target${i}.wav`));

  $("run").disabled = true;
  status.className = "status";
  const started = performance.now();
  const tick = setInterval(() => (status.textContent = `Running ${m.label}… ${formatSeconds((performance.now() - started) / 1000)}`), 200);
  try {
    const res = await fetch(`/api/models/${m.name}/convert`, { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `${res.status} ${res.statusText}`);
    const compute = res.headers.get("X-Compute-Seconds");
    showResult({
      original: state.source.blob,
      output: await res.blob(),
      model: m,
      meta: `${m.label} · model time ${compute ? formatSeconds(+compute) : "?"} · total ${formatSeconds((performance.now() - started) / 1000)}`,
    });
    status.textContent = "";
  } catch (err) {
    status.className = "status error";
    status.textContent = err.message;
  } finally {
    clearInterval(tick);
    updateRunButton();
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

async function onSelectionChange() {
  state.stream?.stop();
  $("latency-card").hidden = true;
  renderModels();
  state.targets = [];
  renderTargets();
  await renderSettings();
  updateRunButton();
}

async function refreshModels() {
  try {
    state.models = await fetchJson("/api/models");
  } catch (err) {
    $("model-info").textContent = `Gateway error: ${err.message}`;
    return;
  }
  renderModels();
  // Re-render settings only when the selected model's status changed (e.g. loading -> ready),
  // so polling never wipes what the user typed.
  if (selectedModel()?.status !== state.shownStatus) await renderSettings();
}

async function init() {
  state.config = await fetchJson("/api/config").catch(() => state.config);
  const sourceInput = createAudioInput({ onChange: (v) => ((state.source = v), updateRunButton()) });
  $("source-input").append(sourceInput.root);

  document.querySelectorAll("input[name=task]").forEach((el) =>
    el.addEventListener("change", () => ((state.task = el.value), onSelectionChange())),
  );
  document.querySelectorAll("input[name=mode]").forEach((el) =>
    el.addEventListener("change", () => ((state.mode = el.value), onSelectionChange())),
  );
  $("model-select").addEventListener("change", (e) => ((state.modelName = e.target.value), onSelectionChange()));
  $("add-target").addEventListener("click", addTarget);
  $("run").addEventListener("click", run);
  document.querySelectorAll("input[name=stream-source]").forEach((el) =>
    el.addEventListener("change", () => ((state.streamSource = el.value), updateRunButton())),
  );
  $("metrics-panel").addEventListener("toggle", () => $("metrics-panel").open && refreshMetricOptions());
  for (const id of ["metric-speaker", "metric-ser", "metric-f0", "metric-mos"]) {
    $(id).addEventListener("change", () => (($(id).dataset.touched = "1"), updateMetricsButton()));
  }
  $("metric-spectrogram").addEventListener("change", updateMetricsButton);
  $("compute-metrics").addEventListener("click", computeMetrics);
  $("stream-file-button").addEventListener("click", () => $("stream-file").click());
  $("stream-file").addEventListener("change", () => {
    state.streamFile = $("stream-file").files[0] || null;
    $("stream-file-name").textContent = state.streamFile?.name || "No file chosen";
    updateRunButton();
  });

  await refreshModels();
  // Poll so "loading" models flip to "ready" without a page reload.
  setInterval(() => {
    if (state.models.length === 0 || state.models.some((m) => m.status !== "ready")) refreshModels();
  }, 3000);
}

init();
