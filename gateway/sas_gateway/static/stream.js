// Live streaming client. Protocol: see workers/common/sas_worker/server.py (stream endpoint).
import { encodeWav } from "./audio.js";

const JITTER_SECONDS = 0.05; // playback cushion added after an underrun
const EMA = 0.1;

function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

/** Decode a file and resample it to `sampleRate` mono with the browser's resampler. */
async function decodeFileAt(file, sampleRate) {
  const ctx = new AudioContext();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(await file.arrayBuffer());
  } catch {
    throw new Error(`Cannot decode "${file.name}". Try WAV, FLAC, MP3, OGG or WebM.`);
  } finally {
    ctx.close();
  }
  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * sampleRate), sampleRate);
  const node = offline.createBufferSource();
  node.buffer = decoded;
  node.connect(offline.destination);
  node.start();
  return (await offline.startRendering()).getChannelData(0);
}

export class LiveStream {
  /**
   * @param {object} opts
   * @param {string} opts.model, opts.task
   * @param {object} opts.params
   * @param {Blob[]} opts.targets   target / reference clips (WAV)
   * @param {File|null} opts.file   stream this file at real-time pace instead of the mic
   * @param {boolean} opts.monitor  play the output while streaming
   * @param {number} opts.maxSeconds
   * @param {(result) => void} opts.onEnd   called once with {input, output, stats} or {error}
   */
  constructor(opts) {
    this.opts = opts;
    this.seq = 0;
    this.sendTimes = new Map();
    this.inputChunks = [];
    this.outputChunks = [];
    this.playAt = 0;
    this.stopping = false;
    this.finished = false;
    this.stats = {
      algorithmicMs: 0, chunkMs: 0, computeMs: 0, computeAvgMs: 0, roundTripMs: 0,
      bufferMs: 0, deviceMs: 0, underruns: 0, sentSeconds: 0, rtf: 0, totalMs: 0,
    };
  }

  async start() {
    const { model, task, params, targets } = this.opts;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${scheme}://${location.host}/api/models/${model}/stream`);
    this.ws.binaryType = "arraybuffer";
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = () => reject(new Error("Cannot open the stream connection to the gateway."));
    });

    this.ws.send(JSON.stringify({ type: "start", task, params, n_targets: targets.length }));
    for (const blob of targets) this.ws.send(await blob.arrayBuffer());

    const ready = await new Promise((resolve, reject) => {
      this.ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "ready") resolve(msg);
        else reject(new Error(msg.detail || "Stream refused"));
      };
      this.ws.onclose = () => reject(new Error("The worker closed the stream before it started."));
    });
    this.ready = ready;
    this.stats.algorithmicMs = ready.algorithmic_latency_ms;
    this.stats.chunkMs = (ready.chunk_samples / ready.sample_rate) * 1000;
    this.ws.onmessage = (e) => this.onMessage(e.data);
    this.ws.onclose = () => this.finish();

    // Running the context at the model's rate avoids resampling the mic; some browsers
    // refuse to connect a mic to a context at a non-native rate, so fall back if needed.
    try {
      this.ctx = new AudioContext({ sampleRate: ready.sample_rate, latencyHint: "interactive" });
    } catch {
      this.ctx = new AudioContext({ latencyHint: "interactive" });
    }
    this.stats.deviceMs = ((this.ctx.baseLatency || 0) + (this.ctx.outputLatency || 0)) * 1000;
    if (this.opts.file) await this.startFile();
    else await this.startMic();
  }

  async startMic() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone unavailable. Open the page via http://localhost (see README_perso.md, step 3.3).");
    }
    this.mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    let source;
    try {
      source = this.ctx.createMediaStreamSource(this.mic);
    } catch {
      await this.ctx.close();
      this.ctx = new AudioContext({ latencyHint: "interactive" });
      source = this.ctx.createMediaStreamSource(this.mic);
    }
    await this.ctx.audioWorklet.addModule("stream-capture-worklet.js");
    this.capture = new AudioWorkletNode(this.ctx, "stream-capture", {
      processorOptions: { targetRate: this.ready.sample_rate, chunkSamples: this.ready.chunk_samples },
    });
    this.capture.port.onmessage = (e) => this.sendChunk(e.data);
    source.connect(this.capture);
    this.micSource = source;
  }

  async startFile() {
    const { sample_rate: rate, chunk_samples: n } = this.ready;
    const samples = await decodeFileAt(this.opts.file, rate);
    const started = performance.now();
    let next = 0;
    const tick = () => {
      if (this.stopping) return;
      const due = Math.floor((((performance.now() - started) / 1000) * rate) / n) + 1;
      while (next < due && next * n < samples.length) {
        const chunk = new Float32Array(n);
        chunk.set(samples.subarray(next * n, (next + 1) * n));
        this.sendChunk(chunk);
        next++;
      }
      if (next * n >= samples.length) this.stop();
      else this.timer = setTimeout(tick, 5);
    };
    tick();
  }

  sendChunk(chunk, record = true) {
    if (this.ws.readyState !== WebSocket.OPEN) return;
    this.seq++;
    this.sendTimes.set(this.seq, performance.now());
    if (record) {
      this.inputChunks.push(chunk);
      this.stats.sentSeconds += chunk.length / this.ready.sample_rate;
    }
    const frame = new ArrayBuffer(4 + chunk.byteLength);
    new DataView(frame).setUint32(0, this.seq, true);
    new Float32Array(frame, 4).set(chunk);
    this.ws.send(frame);
    if (record && this.stats.sentSeconds >= this.opts.maxSeconds) this.stop();
  }

  onMessage(data) {
    if (typeof data === "string") {
      const msg = JSON.parse(data);
      if (msg.type === "end") this.finish();
      else if (msg.type === "error") this.finish(new Error(msg.detail));
      return;
    }
    const view = new DataView(data);
    const seq = view.getUint32(0, true);
    const computeMs = view.getFloat32(4, true);
    const pcm = new Float32Array(data, 8);
    const now = performance.now();
    const s = this.stats;
    if (this.sendTimes.has(seq)) {
      const rt = now - this.sendTimes.get(seq);
      s.roundTripMs = s.roundTripMs ? s.roundTripMs + EMA * (rt - s.roundTripMs) : rt;
      this.sendTimes.delete(seq);
    }
    s.computeMs = computeMs;
    s.computeAvgMs = s.computeAvgMs ? s.computeAvgMs + EMA * (computeMs - s.computeAvgMs) : computeMs;
    s.rtf = s.computeAvgMs / s.chunkMs;
    this.outputChunks.push(pcm);
    if (this.opts.monitor) this.play(pcm);
    s.totalMs = s.chunkMs + s.roundTripMs + s.bufferMs + s.algorithmicMs + s.deviceMs;
  }

  play(pcm) {
    const ctx = this.ctx;
    if (!ctx || ctx.state === "closed") return;
    const buffer = ctx.createBuffer(1, pcm.length, this.ready.output_sample_rate);
    buffer.copyToChannel(pcm, 0);
    const node = ctx.createBufferSource();
    node.buffer = buffer;
    node.connect(ctx.destination);
    const now = ctx.currentTime;
    if (this.playAt < now + 0.005) {
      if (this.playAt > 0) this.stats.underruns++;
      this.playAt = now + JITTER_SECONDS;
    }
    node.start(this.playAt);
    this.stats.bufferMs = (this.playAt - now) * 1000;
    this.playAt += buffer.duration;
  }

  /** Stop capturing, flush the model's delay with silence, and wait for the worker's "end". */
  stop() {
    if (this.stopping) return;
    this.stopping = true;
    clearTimeout(this.timer);
    this.micSource?.disconnect();
    this.capture?.disconnect();
    this.mic?.getTracks().forEach((t) => t.stop());
    if (this.ready && this.ws.readyState === WebSocket.OPEN) {
      const flush = Math.ceil(this.stats.algorithmicMs / this.stats.chunkMs) + 2;
      for (let i = 0; i < flush; i++) this.sendChunk(new Float32Array(this.ready.chunk_samples), false);
      this.ws.send(JSON.stringify({ type: "stop" }));
      this.stopTimeout = setTimeout(() => this.finish(), 30000);
    } else {
      this.finish();
    }
  }

  finish(error) {
    if (this.finished) return;
    this.finished = true;
    this.stopping = true;
    clearTimeout(this.timer);
    clearTimeout(this.stopTimeout);
    this.micSource?.disconnect();
    this.capture?.disconnect();
    this.mic?.getTracks().forEach((t) => t.stop());
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) this.ws.close();
    // Let queued output finish playing before closing the audio context.
    const remaining = this.ctx ? Math.max(0, this.playAt - this.ctx.currentTime) : 0;
    setTimeout(() => this.ctx?.close(), remaining * 1000 + 200);

    if (!this.ready || (error && !this.outputChunks.length)) {
      this.opts.onEnd({ error });
      return;
    }
    this.opts.onEnd({
      error,
      input: encodeWav(concat(this.inputChunks), this.ready.sample_rate),
      output: encodeWav(concat(this.outputChunks), this.ready.output_sample_rate),
      stats: { ...this.stats },
    });
  }
}
