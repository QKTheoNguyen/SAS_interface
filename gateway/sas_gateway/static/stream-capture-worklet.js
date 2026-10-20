// Live mic capture for streaming: resamples to the model's rate (linear interpolation,
// continuous across blocks) and posts fixed-size Float32Array chunks to the main thread.
// When the AudioContext already runs at the model's rate this is a plain pass-through.
class StreamCaptureProcessor extends AudioWorkletProcessor {
  constructor({ processorOptions: { targetRate, chunkSamples } }) {
    super();
    this.step = sampleRate / targetRate; // input samples per output sample
    this.t = 0; // next output position, in input samples relative to the current block (-1 = previous block's last sample)
    this.prev = 0;
    this.chunk = new Float32Array(chunkSamples);
    this.filled = 0;
  }

  push(value) {
    this.chunk[this.filled++] = value;
    if (this.filled === this.chunk.length) {
      this.port.postMessage(this.chunk.slice(0));
      this.filled = 0;
    }
  }

  process(inputs) {
    const x = inputs[0] && inputs[0][0];
    if (!x) return true;
    const last = x.length - 1;
    while (this.t <= last) {
      const i = Math.floor(this.t);
      const frac = this.t - i;
      const a = i < 0 ? this.prev : x[i];
      const b = i + 1 <= last ? x[i + 1] : a;
      this.push(a + frac * (b - a));
      this.t += this.step;
    }
    this.t -= x.length;
    this.prev = x[last];
    return true;
  }
}

registerProcessor("stream-capture", StreamCaptureProcessor);
