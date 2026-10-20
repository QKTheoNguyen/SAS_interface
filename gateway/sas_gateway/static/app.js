import { fileToWav, formatSeconds, startRecording } from "./audio.js";

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
};

// ---------------------------------------------------------------------------
// Audio input widget: record or upload one clip.
// ---------------------------------------------------------------------------

function createAudioInput({ onChange, onRemove } = {}) {
  const root = document.createElement("div");
  root.className = "clip";
  const fileInput = Object.assign(document.createElement("input"), { type: "file", accept: "audio/*", hidden: true });
  let value = null;
  let recorder = null;
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
      clearInterval(timer);
      const clip = await recorder.stop();
      recorder = null;
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

  function render() {
    root.replaceChildren(fileInput);
    const row = document.createElement("div");
    row.className = "row";
    if (recorder) {
      row.append(button(`■ Stop (${formatSeconds(recorder.elapsed())})`, toggleRecording, "recording"));
      if (recorder.elapsed() > state.config.max_input_seconds) toggleRecording();
    } else {
      row.append(button(value ? "● Re-record" : "● Record", toggleRecording));
      row.append(button(value ? "Replace file" : "Upload file", () => fileInput.click()));
      if (onRemove) row.append(button("Remove", onRemove));
      if (value) {
        const meta = document.createElement("span");
        meta.className = "muted";
        meta.textContent = `${value.name} · ${formatSeconds(value.duration)}`;
        row.append(meta);
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
  if (state.mode === "streaming") {
    const p = document.createElement("p");
    p.className = "notice";
    p.textContent = "Streaming is not wired up yet (phase 3).";
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
  const describe = () => (help.textContent = selectedPreset()?.description || "Values edited by hand.");
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
  const params = spec.params || [];
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
  $("target-card").querySelector(".step").textContent = params.length ? "6" : "5";
  if (targetVoice === "required" && !state.targets.length) addTarget();
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
  const blockers = [];
  if (!m || m.status !== "ready") blockers.push(`model is ${m?.status ?? "missing"}`);
  if (state.mode === "streaming") blockers.push("streaming is not available yet");
  if (!state.source) blockers.push("add source speech");
  if (needsTarget && !hasTarget) blockers.push("add a target voice clip");
  $("run").disabled = blockers.length > 0;
  $("run").title = blockers.join(", ");
}

async function run() {
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
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    $("original-audio").src = URL.createObjectURL(state.source.blob);
    $("result-audio").src = url;
    $("download").href = url;
    $("download").download = `${m.name}_${state.task}_${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
    const compute = res.headers.get("X-Compute-Seconds");
    $("result-meta").textContent = `${m.label} · model time ${compute ? formatSeconds(+compute) : "?"} · total ${formatSeconds((performance.now() - started) / 1000)}`;
    $("result-card").hidden = false;
    status.textContent = "";
    $("result-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
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

  await refreshModels();
  // Poll so "loading" models flip to "ready" without a page reload.
  setInterval(() => {
    if (state.models.length === 0 || state.models.some((m) => m.status !== "ready")) refreshModels();
  }, 3000);
}

init();
