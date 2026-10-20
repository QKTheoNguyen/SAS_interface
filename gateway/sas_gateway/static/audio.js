// Browser-side audio helpers. Everything sent to the server is mono 16-bit WAV,
// so the server never needs ffmpeg: the browser decodes mp3/ogg/webm/flac itself.

export function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function concat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

function toMono(audioBuffer) {
  const { numberOfChannels, length } = audioBuffer;
  if (numberOfChannels === 1) return audioBuffer.getChannelData(0);
  const out = new Float32Array(length);
  for (let ch = 0; ch < numberOfChannels; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) out[i] += data[i] / numberOfChannels;
  }
  return out;
}

/** Decode any browser-supported audio file into {blob (WAV), duration}. */
export async function fileToWav(file) {
  const ctx = new AudioContext();
  try {
    const decoded = await ctx.decodeAudioData(await file.arrayBuffer());
    return { blob: encodeWav(toMono(decoded), decoded.sampleRate), duration: decoded.duration };
  } catch {
    throw new Error(`Cannot decode "${file.name}". Try WAV, FLAC, MP3, OGG or WebM.`);
  } finally {
    ctx.close();
  }
}

/**
 * Start recording from the microphone. Returns {stop(): Promise<{blob, duration}>, elapsed(): seconds}.
 * Browser processing (echo cancellation, noise suppression, AGC) is disabled so the
 * models receive the raw voice.
 */
export async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone unavailable. Open the page via http://localhost (see README_perso.md, step 3.3).");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("recorder-worklet.js");
  const sourceNode = ctx.createMediaStreamSource(stream);
  const recorder = new AudioWorkletNode(ctx, "recorder-processor");
  const chunks = [];
  let samples = 0;
  recorder.port.onmessage = (e) => {
    chunks.push(e.data);
    samples += e.data.length;
  };
  sourceNode.connect(recorder);

  return {
    elapsed: () => samples / ctx.sampleRate,
    async stop() {
      sourceNode.disconnect();
      recorder.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      const sampleRate = ctx.sampleRate;
      await ctx.close();
      const data = concat(chunks);
      return { blob: encodeWav(data, sampleRate), duration: data.length / sampleRate };
    },
  };
}

export function formatSeconds(s) {
  return `${s.toFixed(1)} s`;
}
