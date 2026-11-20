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
  piiPoll: null,
  piiUrl: null, // object URL of the obfuscated output
  piiLiveUrl: null, // object URL of the live-masked stream recording
  piiOptions: null, // last /api/pii/options answer
  piiLive: { lines: new Map(), segments: [] }, // live transcript of the current / last stream (line -> words)
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

// ---------------------------------------------------------------------------
// Speaker map: pseudo-speakers placed by pitch (x) and brightness (y), from the
// model's speakers.json (scripts/build_reference_pool.py). One gender at a time,
// because coordinates are z-scored within each gender.
// ---------------------------------------------------------------------------

const PREVIEW_SECONDS = 3;
const preview = { audio: new Audio(), timer: null };
const SVG_NS = "http://www.w3.org/2000/svg";

function playPreview(url) {
  clearTimeout(preview.timer);
  preview.audio.src = url;
  preview.audio.play().catch(() => {});
  preview.timer = setTimeout(() => preview.audio.pause(), PREVIEW_SECONDS * 1000);
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

async function createSpeakerMap(model, param, value, update) {
  const key = `${model.name}/speakers`;
  state.choicesCache[key] ??= fetchJson(`/api/models/${model.name}/speakers`);
  let meta;
  try {
    meta = await state.choicesCache[key];
  } catch (err) {
    delete state.choicesCache[key];
    const p = Object.assign(document.createElement("p"), { className: "warning", textContent: err.message });
    return p;
  }

  const speakers = meta.speakers;
  const root = document.createElement("div");
  root.className = "speaker-map";
  let selected = value;
  let gender = speakers.find((s) => s.id === selected)?.gender || "F";

  // Shared scale for both genders so the plot does not jump when toggling.
  const W = 400, H = 280, PAD = 22;
  const xs = speakers.map((s) => s.x), ys = speakers.map((s) => s.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const px = (x) => PAD + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * PAD);
  const py = (y) => H - PAD - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * PAD);

  const toolbar = document.createElement("div");
  toolbar.className = "row";
  const toggle = document.createElement("div");
  toggle.className = "segmented";
  const radioName = `speaker-gender-${param.name}`;
  for (const [id, text] of [["F", "Female"], ["M", "Male"]]) {
    const input = Object.assign(document.createElement("input"), { type: "radio", name: radioName, value: id, checked: gender === id });
    input.addEventListener("change", () => ((gender = id), draw()));
    const lab = document.createElement("label");
    lab.append(input, text);
    toggle.append(lab);
  }
  const random = Object.assign(document.createElement("button"), { type: "button", className: "secondary", textContent: "Random" });
  random.addEventListener("click", () => select("random"));
  toolbar.append(toggle, random);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Pseudo-speaker map" });
  const status = document.createElement("div");
  status.className = "speaker-status";

  function select(id, play = false) {
    selected = id;
    update(id);
    if (play) playPreview(`/api/models/${model.name}/speakers/${encodeURIComponent(id)}.flac`);
    draw();
  }

  function draw() {
    const axes = [
      svgEl("line", { x1: px(0), x2: px(0), y1: PAD / 2, y2: H - PAD / 2, class: "axis" }),
      svgEl("line", { x1: PAD / 2, x2: W - PAD / 2, y1: py(0), y2: py(0), class: "axis" }),
    ];
    const labels = [
      [4, H - 6, "start", "← lower pitch"],
      [W - 4, H - 6, "end", "higher pitch →"],
      [4, 12, "start", "↑ brighter"],
      [4, H - 20, "start", "↓ darker"],
    ].map(([x, y, anchor, text]) => Object.assign(svgEl("text", { x, y, "text-anchor": anchor, class: "axis-label" }), { textContent: text }));
    const dots = speakers.filter((s) => s.gender === gender).map((s) => {
      const dot = svgEl("circle", {
        cx: px(s.x), cy: py(s.y), r: s.id === selected ? 8 : 5.5, tabindex: 0, role: "button",
        class: s.id === selected ? "dot selected" : "dot", "aria-label": `${s.id}, ${s.f0_hz} Hz`,
      });
      dot.append(Object.assign(svgEl("title"), { textContent: `${s.id} · ${s.f0_hz} Hz` }));
      dot.addEventListener("click", () => select(s.id, true));
      dot.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") (e.preventDefault(), select(s.id, true));
      });
      return dot;
    });
    svg.replaceChildren(...axes, ...labels, ...dots);

    random.classList.toggle("active", selected === "random");
    const s = speakers.find((sp) => sp.id === selected);
    status.textContent = s
      ? `Selected: ${s.id} · median F0 ${s.f0_hz} Hz${s.gender !== gender ? ` (${s.gender === "F" ? "female" : "male"} map)` : ""}`
      : "Selected: random voice, a new one for each run.";
  }

  draw();
  root.append(toolbar, svg, status);
  return root;
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
  } else if (param.type === "speaker_map") {
    label.removeAttribute("for");
    wrap.append(await createSpeakerMap(model, param, value, update));
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
  const live = livePanelVisible();
  $("live-pii-panel").hidden = !live;
  $("advanced").hidden = !advanced.length;
  $("params-card").hidden = !params.length && !live;
  if (live) refreshPiiOptions();

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
  updatePiiButton();
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

function showResult({ original, output, model, meta, offsetS = 0 }) {
  const url = URL.createObjectURL(output);
  $("original-audio").src = URL.createObjectURL(original);
  $("result-audio").src = url;
  $("download").href = url;
  $("download").download = `${model.name}_${state.task}_${state.mode}_${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
  $("result-meta").textContent = meta;
  $("result-card").hidden = false;
  // offsetS: how much later the output is than the source (stream recordings: the model delay).
  state.lastResult = { original, output, task: state.task, mode: state.mode, offsetS };
  $("metrics-results").replaceChildren();
  $("rolling-plot").hidden = true;
  $("f0-plot").hidden = true;
  $("spectrogram-plot").hidden = true;
  $("metrics-status").textContent = "";
  $("metrics-panel").hidden = !(state.config.metrics && state.task === "anonymization");
  updateMetricsButton();
  resetPii();
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

// Speaker similarity over time: window/hop limits mirrored from workers/common/sas_worker/metrics.py.
const ROLLING_SAMPLE_RATE = 16000;
const ROLLING_MAX_WINDOW_S = { CosyVoiceCAMPPlusSpeakerEncoder: 10 };

/** {on, window, hop, error}: `on` only when the box is ticked and a speaker encoder is picked. */
function rollingSettings() {
  const encoder = $("metric-speaker").value;
  const on = !!encoder && $("metric-rolling").checked;
  const preset = $("rolling-preset").value;
  const [win, hop] = preset === "custom" ? [Number($("rolling-window").value), Number($("rolling-hop").value)] : preset.split("/").map(Number);
  let error = "";
  const wholeSamples = (s) => Math.abs(s * ROLLING_SAMPLE_RATE - Math.round(s * ROLLING_SAMPLE_RATE)) < 1e-6;
  if (!(win > 0) || !(hop > 0)) error = "Window and hop must be greater than 0 s.";
  else if (hop > win) error = "Hop must not be longer than the window.";
  else if (!wholeSamples(win) || !wholeSamples(hop)) error = `Window and hop must be whole samples at ${ROLLING_SAMPLE_RATE / 1000} kHz.`;
  else if (win > (ROLLING_MAX_WINDOW_S[encoder] ?? Infinity)) error = `This encoder supports windows up to ${ROLLING_MAX_WINDOW_S[encoder]} s.`;
  return { on, window: win, hop, error };
}

function updateRollingControls() {
  const hasEncoder = !!$("metric-speaker").value;
  $("metric-rolling").disabled = !hasEncoder;
  $("rolling-custom").hidden = $("rolling-preset").value !== "custom";
  $("rolling-preset").disabled = !hasEncoder || !$("metric-rolling").checked;
  const { on, error } = rollingSettings();
  const help = $("rolling-help");
  help.className = on && error ? "help warning" : "help";
  help.textContent = !hasEncoder
    ? "Pick a speaker encoder to plot speaker similarity over time."
    : on && error
      ? error
      : "Uses the speaker encoder above, on successive windows of source and output.";
}

function updateMetricsButton() {
  updateRollingControls();
  const rolling = rollingSettings();
  const chosen =
    $("metric-speaker").value || $("metric-ser").value || $("metric-f0").value || $("metric-mos").value || $("metric-spectrogram").checked;
  const invalid = rolling.on && rolling.error;
  // Metrics need a finished result: a live stream must be stopped first.
  $("compute-metrics").disabled = !state.lastResult || !!state.stream || !chosen || !!invalid;
  $("compute-metrics").title = state.stream
    ? "Stop the stream first"
    : !chosen
      ? "Pick at least one model or the spectrograms"
      : invalid
        ? rolling.error
        : "";
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
  const rolling = rollingSettings();
  form.append("rolling", rolling.on);
  if (rolling.on) {
    form.append("rolling_window_s", rolling.window);
    form.append("rolling_hop_s", rolling.hop);
    form.append("offset_s", state.lastResult.offsetS || 0);
  }
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
    $("rolling-plot").hidden = !r.speaker_similarity_over_time;
    if (r.speaker_similarity_over_time) renderRollingPlot($("rolling-plot"), r.speaker_similarity_over_time, r.speaker_similarity);
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

// ---------------------------------------------------------------------------
// Post processing: PII obfuscation (anonymization, non-streaming only)
// ---------------------------------------------------------------------------

/** Show / hide section 8 and clear its results. In streaming mode it is shown before Start
 *  (live controls); in non-streaming mode once there is a result. */
/** Show / hide section 8 (once there is a result) and clear its results. */
function resetPii() {
  const show = !!state.config.pii && state.task === "anonymization" && !!state.lastResult;
  $("pii-card").hidden = !show;
  $("pii-result").hidden = true;
  $("pii-live-result").hidden = true;
  $("pii-status").textContent = "";
  for (const key of ["piiUrl", "piiLiveUrl"]) {
    if (state[key]) URL.revokeObjectURL(state[key]);
    state[key] = null;
  }
  if (show) refreshPiiOptions();
  else updatePiiButton();
}

/** Live PII panel (Settings > Advanced): streaming anonymization with a metrics worker. */
function livePanelVisible() {
  return !!state.config.pii && state.task === "anonymization" && state.mode === "streaming";
}

/** Fill a model dropdown. With `streaming`, models without open_stream() are listed but disabled. */
function fillPiiSelect(select, models, { streaming = false } = {}) {
  const previous = select.value;
  const usable = (m) => m.status === "ready" && (!streaming || m.streaming);
  const options = models.map((m) => {
    const suffix =
      m.status === "unavailable" ? " (unavailable)" : m.status !== "ready" ? " (loading…)" : !usable(m) ? " (no streaming mode)" : "";
    const option = new Option(m.label + suffix, m.id);
    option.disabled = !usable(m);
    if (m.error) option.title = m.error;
    return option;
  });
  if (!options.length) options.push(Object.assign(new Option("None available", ""), { disabled: true }));
  select.replaceChildren(...options);
  const ok = models.filter(usable);
  select.value = (ok.find((m) => m.id === previous) || ok.find((m) => m.default) || ok[0])?.id ?? "";
}

async function refreshPiiOptions() {
  const status = $("pii-status");
  try {
    const options = await fetchJson("/api/pii/options");
    state.piiOptions = options;
    fillPiiSelect($("pii-asr"), options.asr_models);
    fillPiiSelect($("pii-detector"), options.pii_models);
    fillPiiSelect($("live-pii-asr"), options.asr_models, { streaming: true });
    fillPiiSelect($("live-pii-detector"), options.pii_models, { streaming: true });
    status.className = "status";
    status.textContent = options.asr_models.length && options.pii_models.length ? "" : "No ASR/PII models available.";
    const pending = [...options.asr_models, ...options.pii_models].some((m) => ["pending", "loading"].includes(m.status));
    clearTimeout(state.piiPoll);
    if (pending && !($("pii-card").hidden && $("live-pii-panel").hidden)) state.piiPoll = setTimeout(refreshPiiOptions, 3000);
  } catch (err) {
    for (const id of ["pii-asr", "pii-detector", "live-pii-asr", "live-pii-detector"]) fillPiiSelect($(id), []);
    state.piiOptions = null;
    status.className = "status error";
    status.textContent = err.message;
  }
  updatePiiButton();
}

/** The selected ASR and PII models both implement open_stream(). */
function piiLiveSupported() {
  const o = state.piiOptions;
  const asr = o?.asr_models.find((m) => m.id === $("live-pii-asr").value);
  const pii = o?.pii_models.find((m) => m.id === $("live-pii-detector").value);
  return !!(asr?.streaming && pii?.streaming);
}

function updatePiiButton() {
  const live = !!state.stream; // settings are locked while streaming
  const models = $("pii-asr").value && $("pii-detector").value;
  const ok = state.lastResult && models && !live;
  $("run-pii").disabled = !ok;
  $("run-pii").title = live ? "Stop the stream first" : ok ? "" : "Needs a result and a ready ASR model and PII detector";
  for (const id of ["pii-asr", "pii-detector", "pii-mask", "pii-padding", "pii-padding-range"]) $(id).disabled = live;

  // Live panel (Settings > Advanced)
  const supported = piiLiveSupported();
  if (!supported) setLivePii(false);
  const on = livePiiOn();
  $("live-pii-enable").disabled = live || !supported;
  for (const id of ["live-pii-asr", "live-pii-detector", "live-pii-mask", "live-pii-padding", "live-pii-padding-range", "live-pii-holdback-auto"]) {
    $(id).disabled = live;
  }
  const manual = !live && !$("live-pii-holdback-auto").checked;
  $("live-pii-holdback").disabled = $("live-pii-holdback-range").disabled = !manual;
  $("live-pii-help").textContent = !state.piiOptions
    ? "No ASR/PII models available."
    : supported
      ? "The output is held back long enough for the detector to decide, then masked before you hear it."
      : "Needs an ASR model and a PII detector that both have a streaming mode.";
  $("live-pii-state").textContent = on ? "· on" : "· off";
}

/** Live obfuscation toggle button: its state is aria-pressed. */
function livePiiOn() {
  return $("live-pii-enable").getAttribute("aria-pressed") === "true";
}

function setLivePii(on) {
  const button = $("live-pii-enable");
  button.setAttribute("aria-pressed", String(on));
  button.classList.toggle("active", on);
  button.textContent = `Live obfuscation: ${on ? "on" : "off"}`;
}

/** Live PII options for LiveStream, or null when live obfuscation is off. */
function livePiiOptions() {
  if (!livePanelVisible() || !livePiiOn() || !piiLiveSupported()) return null;
  return {
    asr_model: $("live-pii-asr").value,
    pii_model: $("live-pii-detector").value,
    mask: $("live-pii-mask").value,
    padding_ms: +$("live-pii-padding").value,
    holdback_ms: $("live-pii-holdback-auto").checked ? null : +$("live-pii-holdback").value,
  };
}

/** Words of the live transcript in order. A revised line (same `line` id) replaced its earlier words. */
function liveWords() {
  return [...state.piiLive.lines.values()].flat();
}

function onLivePii(event) {
  const live = state.piiLive;
  if (event.type === "pii_words") {
    const batch = new Map();
    for (const w of event.words) {
      const key = w.line ?? `batch-${live.lines.size}`; // untagged words are final: their own group
      if (!batch.has(key)) batch.set(key, []);
      if (w.word) batch.get(key).push(w);
    }
    for (const [key, words] of batch) live.lines.set(key, words);
  }
  else if (event.type === "pii_segments") live.segments.push(...event.segments);
  else if (event.type === "pii_error") {
    $("pii-live-warning").hidden = false;
    $("pii-live-warning").textContent = `⚠ ${event.detail}`;
  }
  $("pii-card").hidden = false;
  $("pii-live-result").hidden = false;
  renumberSteps();
  renderTranscript($("pii-live-transcript"), liveWords(), live.segments);
}

/** After a stream with live PII: the masked recording, leak report and final transcript. */
function showLivePii(result, model) {
  if (!result.masked) return;
  const box = $("pii-live-result");
  box.hidden = false;
  if (state.piiLiveUrl) URL.revokeObjectURL(state.piiLiveUrl);
  state.piiLiveUrl = URL.createObjectURL(result.masked);
  $("pii-live-audio").src = state.piiLiveUrl;
  $("pii-live-download").href = state.piiLiveUrl;
  $("pii-live-download").download = ($("download").download || `${model.name}.wav`).replace(/\.wav$/, `_pii-live-${$("live-pii-mask").value}.wav`);
  $("pii-live-player").hidden = false;
  $("pii-live-row").hidden = false;
  const sum = result.pii?.summary;
  const n = state.piiLive.segments.length;
  $("pii-live-meta").textContent = sum
    ? `${n} PII segment${n === 1 ? "" : "s"} · holdback ${Math.round(sum.holdback_ms)} ms · ` +
      (sum.leaked_segments ? `Leaked: ${Math.round(sum.leaked_ms)} ms in ${sum.leaked_segments} segment${sum.leaked_segments === 1 ? "" : "s"}` : "no leak")
    : "";
  const warning = $("pii-live-warning");
  if (result.pii?.error) {
    warning.hidden = false;
    warning.textContent = `⚠ ${result.pii.error}`;
  } else if (sum?.leaked_segments) {
    warning.hidden = false;
    warning.textContent = "Some PII was detected after it had already been played: increase the holdback.";
  }
  renderTranscript($("pii-live-transcript"), liveWords(), state.piiLive.segments);
}

/** Transcript with the words inside a PII segment wrapped in <mark>, one mark per segment. */
function renderTranscript(box, words, segments) {
  const segmentOf = (w) => {
    const mid = (w.start_s + w.end_s) / 2;
    return segments.findIndex((s) => mid >= s.start_s && mid <= s.end_s);
  };
  const nodes = [];
  let mark = null;
  let markSeg = -1;
  for (const w of words) {
    const seg = segmentOf(w);
    if (seg >= 0 && seg === markSeg) {
      mark.insertBefore(document.createTextNode(` ${w.word}`), mark.lastChild);
      continue;
    }
    if (nodes.length) nodes.push(document.createTextNode(" "));
    if (seg >= 0) {
      const s = segments[seg];
      mark = document.createElement("mark");
      mark.title = `${s.label} · ${formatSeconds(s.start_s)}–${formatSeconds(s.end_s)} · confidence ${s.confidence.toFixed(2)}`;
      mark.append(w.word, Object.assign(document.createElement("span"), { className: "pii-label", textContent: s.label }));
      nodes.push(mark);
    } else {
      nodes.push(document.createTextNode(w.word));
    }
    markSeg = seg;
  }
  box.replaceChildren(...nodes);
}

async function runPii() {
  const status = $("pii-status");
  const form = new FormData();
  form.append("source", state.lastResult.original, "source.wav");
  form.append("output", state.lastResult.output, "output.wav");
  form.append("asr_model", $("pii-asr").value);
  form.append("pii_model", $("pii-detector").value);
  form.append("mask", $("pii-mask").value);
  form.append("padding_ms", $("pii-padding").value);
  form.append("offset_s", state.lastResult.offsetS || 0);
  status.className = "status";
  status.textContent = "Detecting…";
  $("run-pii").disabled = true;
  try {
    const res = await fetch("/api/pii/obfuscate", { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `${res.status} ${res.statusText}`);
    const r = await res.json();
    const bytes = Uint8Array.from(atob(r.audio_b64), (c) => c.charCodeAt(0));
    if (state.piiUrl) URL.revokeObjectURL(state.piiUrl);
    state.piiUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
    $("pii-audio").src = state.piiUrl;
    $("pii-download").href = state.piiUrl;
    $("pii-download").download = ($("download").download || "output.wav").replace(/\.wav$/, `_pii-${$("pii-mask").value}.wav`);
    const n = r.segments.length;
    $("pii-meta").textContent = `${n} PII segment${n === 1 ? "" : "s"} masked`;
    const drift = r.output_seconds - r.source_seconds;
    $("pii-warning").hidden = Math.abs(drift) < 0.05;
    $("pii-warning").textContent =
      `Output is ${formatSeconds(Math.abs(drift))} ${drift > 0 ? "longer" : "shorter"} than the source: ` +
      "masks may be slightly off, increase the padding if PII is still audible.";
    renderTranscript($("pii-transcript"), r.words, r.segments);
    $("pii-result").hidden = false;
    status.textContent = `Done in ${formatSeconds(r.seconds)}.`;
  } catch (err) {
    status.className = "status error";
    status.textContent = err.message;
  } finally {
    updatePiiButton();
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
 * Speaker similarity over time (quick-convert RollingSpeakerSimilarity, one point per window).
 * `rolling`: the `speaker_similarity_over_time` response of /api/metrics/compute.
 * `overall`: the full-recording `speaker_similarity`, drawn as a dashed reference, never derived
 * from the windows. Windows without a score (unavailable / error) break the line and are drawn as
 * hollow markers in a separate "n/a" band below the axis, so they never read as a value.
 */
function renderRollingPlot(figure, rolling, overall) {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const div = (className, textContent = "") => Object.assign(document.createElement("div"), { className, textContent });
  const ms = (v) => `${Math.round(v)} ms`;
  // Results may come back in any order; plot by window index.
  const windows = [...rolling.windows].sort((a, b) => a.window_index - b.window_index);
  const { ok, unavailable, error } = rolling.counts;

  const caption = Object.assign(document.createElement("figcaption"), { textContent: "Speaker similarity over time" });
  const delay = rolling.offset_s ? ` · output delay ${ms(rolling.offset_s * 1000)} compensated` : "";
  const meta = div(
    "rolling-meta muted",
    `${rolling.model} · window ${rolling.window_s} s / hop ${rolling.hop_s} s${delay} · ` +
      `${ok} scored, ${unavailable} unavailable, ${error} failed` +
      (rolling.mean_compute_ms !== null ? ` · ${ms(rolling.mean_compute_ms)} per window` : ""),
  );
  const explain = div(
    "help muted",
    "Cosine similarity between the speaker embeddings of the same time window in the source and the output " +
      "(−1 to 1; lower = the voice changed more). It is not an EER, a linkability score or a privacy percentage. " +
      "Silence, short windows and misalignment make single windows less reliable. The overall score is computed " +
      "on the full recordings, not averaged from these windows.",
  );
  const parts = [caption, meta];
  if (rolling.note) parts.push(div("notice", rolling.note));
  if (windows.length === 0) {
    figure.replaceChildren(...parts, explain);
    return;
  }

  const W = 640, H = 250, L = 44, R = 10, T = 10, B = 28, NA = 20;
  const plotBottom = H - B - NA; // y = -1
  const naY = H - B - NA / 2; // band for windows without a score
  const tMax = Math.max(rolling.source_seconds, windows.at(-1).window_end_seconds, 0.01);
  const x = (t) => L + (t / tMax) * (W - L - R);
  const y = (v) => T + ((1 - v) / 2) * (plotBottom - T);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Speaker similarity over time (${rolling.model})` });
  for (const v of [-1, -0.5, 0, 0.5, 1]) {
    svg.append(el("line", { class: v === 0 ? "zero" : "grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }));
    svg.append(el("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, String(v)));
  }
  for (const v of niceTicks(0, tMax, 6)) {
    svg.append(el("text", { x: x(v), y: H - B + 16, "text-anchor": "middle" }, `${v}`));
  }
  svg.append(el("text", { x: W - R, y: H - 4, "text-anchor": "end" }, "time (s), point at window end"));
  svg.append(el("line", { class: "axis", x1: L, x2: W - R, y1: H - B, y2: H - B }));
  if (ok < windows.length) svg.append(el("text", { x: L - 6, y: naY + 4, "text-anchor": "end" }, "n/a"));

  const shade = el("rect", { class: "shade", y: T, height: H - B - T, visibility: "hidden" });
  svg.append(shade);

  if (overall) {
    svg.append(el("line", { class: "overall", x1: L, x2: W - R, y1: y(overall.value), y2: y(overall.value) }));
    const above = overall.value < 0.85;
    svg.append(
      el("text", { class: "overall-label", x: W - R - 2, y: y(overall.value) + (above ? -5 : 13), "text-anchor": "end" },
        `overall (full recording) ${overall.value.toFixed(3)}`),
    );
  }

  let d = "";
  let pen = false;
  for (const w of windows) {
    if (w.status !== "ok") {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x(w.window_end_seconds).toFixed(1)} ${y(w.speaker_similarity).toFixed(1)}`;
    pen = true;
  }
  svg.append(el("path", { class: "series", d }));
  const markers = windows.map((w) => {
    const marker =
      w.status === "ok"
        ? el("circle", { class: "dot", cx: x(w.window_end_seconds), cy: y(w.speaker_similarity), r: 3 })
        : el("circle", { class: `missing ${w.status}`, cx: x(w.window_end_seconds), cy: naY, r: 3.5 });
    svg.append(marker);
    return marker;
  });

  // Hover: shade the hovered window's span, read its values out under the plot.
  const readout = div("legend");
  const idle = "Hover the plot to inspect a window.";
  readout.textContent = idle;
  let active = null;
  const clear = () => {
    shade.setAttribute("visibility", "hidden");
    active?.classList.remove("active");
    active = null;
    readout.textContent = idle;
  };
  svg.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const t = ((((event.clientX - box.left) / box.width) * W - L) / (W - L - R)) * tMax;
    if (t < 0 || t > tMax) return clear();
    let i = 0;
    windows.forEach((w, j) => {
      if (Math.abs(w.window_end_seconds - t) < Math.abs(windows[i].window_end_seconds - t)) i = j;
    });
    const w = windows[i];
    shade.setAttribute("x", x(w.window_start_seconds));
    shade.setAttribute("width", x(w.window_end_seconds) - x(w.window_start_seconds));
    shade.setAttribute("visibility", "visible");
    active?.classList.remove("active");
    active = markers[i];
    active.classList.add("active");
    const span = `Window ${w.window_index + 1}: ${w.window_start_seconds.toFixed(1)}–${w.window_end_seconds.toFixed(1)} s`;
    readout.textContent =
      w.status === "ok"
        ? `${span} · similarity ${w.speaker_similarity.toFixed(3)} · computed in ${ms(w.compute_time_ms)}`
        : `${span} · ${w.status === "error" ? "failed" : "unavailable"}: ${w.error_message || "no score"}`;
  });
  svg.addEventListener("pointerleave", clear);

  figure.replaceChildren(...parts, svg, readout, explain);
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
    ...(stats.piiHoldbackMs ? [["PII holdback", ms(stats.piiHoldbackMs)]] : []),
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
  state.piiLive = { lines: new Map(), segments: [] };
  $("pii-live-result").hidden = true;
  $("pii-live-player").hidden = true;
  $("pii-live-row").hidden = true;
  $("pii-live-warning").hidden = true;
  $("pii-live-transcript").replaceChildren();
  const stream = new LiveStream({
    model: m.name,
    task: state.task,
    params: state.params,
    targets: state.targets.filter((t) => t.input.value).map((t) => t.input.value.blob),
    file: state.streamSource === "file" ? state.streamFile : null,
    monitor: $("stream-monitor").checked,
    maxSeconds: state.config.max_stream_seconds,
    pii: livePiiOptions(),
    onPii: onLivePii,
    onEnd: (result) => endStream(m, result),
  });
  state.stream = stream;
  updateRunButton();
  try {
    await stream.start();
    status.textContent = stream.opts.file ? `Streaming ${stream.opts.file.name}…` : "Streaming from the microphone…";
    // Auto holdback: show the value the PII service picked (detector latency + padding).
    if (stream.stats.piiHoldbackMs && $("live-pii-holdback-auto").checked) {
      const v = Math.min(3000, Math.round(stream.stats.piiHoldbackMs / 50) * 50);
      $("live-pii-holdback").value = $("live-pii-holdback-range").value = v;
    }
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
      offsetS: s.algorithmicMs / 1000,
    });
    showLivePii(result, model);
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
  resetPii();
  renumberSteps();
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
  for (const id of ["metric-rolling", "rolling-preset"]) $(id).addEventListener("change", updateMetricsButton);
  for (const id of ["rolling-window", "rolling-hop"]) $(id).addEventListener("input", updateMetricsButton);
  $("compute-metrics").addEventListener("click", computeMetrics);
  for (const id of ["pii-asr", "pii-detector"]) $(id).addEventListener("change", updatePiiButton);
  $("pii-padding-range").addEventListener("input", () => ($("pii-padding").value = $("pii-padding-range").value));
  $("pii-padding").addEventListener("input", () => ($("pii-padding-range").value = $("pii-padding").value));
  $("run-pii").addEventListener("click", runPii);
  $("live-pii-enable").addEventListener("click", () => (setLivePii(!livePiiOn()), updatePiiButton()));
  for (const id of ["live-pii-asr", "live-pii-detector", "live-pii-holdback-auto"]) {
    $(id).addEventListener("change", updatePiiButton);
  }
  $("live-pii-padding-range").addEventListener("input", () => ($("live-pii-padding").value = $("live-pii-padding-range").value));
  $("live-pii-holdback-range").addEventListener("input", () => ($("live-pii-holdback").value = $("live-pii-holdback-range").value));
  $("live-pii-holdback").addEventListener("input", () => ($("live-pii-holdback-range").value = $("live-pii-holdback").value));
  $("live-pii-padding").addEventListener("input", () => ($("live-pii-padding-range").value = $("live-pii-padding").value));
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
