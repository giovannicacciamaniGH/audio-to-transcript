'use strict';
/**
 * Speaker labels come back per request, so "Speaker A" in one chunk and "Speaker A" in
 * the next are not necessarily the same person. This re-identifies voices across the
 * whole recording: it takes a spectral fingerprint of each (chunk, label) pair straight
 * from the audio and clusters fingerprints that sound like the same voice.
 */

const FFT_SIZE = 512;
const BAND_COUNT = 24;
const MAX_ANALYSIS_SECONDS = 20;   // per speaker per chunk — plenty for a timbre average
const FALLBACK_THRESHOLD = 0.97;   // used only when the recording offers no calibration
const MARGIN = 0.01;

/** Iterative in-place radix-2 FFT. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k];
        const ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
const melToHz = (mel) => 700 * (10 ** (mel / 2595) - 1);

/** Mel-spaced band edges over the usable speech range. */
function bandEdges(sampleRate) {
  const low = hzToMel(80);
  const high = hzToMel(Math.min(7600, sampleRate / 2 - 100));
  const edges = [];
  for (let i = 0; i <= BAND_COUNT; i++) {
    const hz = melToHz(low + ((high - low) * i) / BAND_COUNT);
    edges.push(Math.max(1, Math.round((hz / sampleRate) * FFT_SIZE)));
  }
  return edges;
}

/**
 * Average log-band-energy over the loud frames of the given time ranges, then
 * mean-subtracted — which cancels overall gain and leaves the shape of the voice.
 */
function fingerprint(buffer, ranges) {
  const data = buffer.getChannelData(0);
  const rate = buffer.sampleRate;
  const edges = bandEdges(rate);
  const window = new Float32Array(FFT_SIZE);
  for (let i = 0; i < FFT_SIZE; i++) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / FFT_SIZE);

  const sum = new Float64Array(BAND_COUNT);
  const frames = [];
  let analysed = 0;

  for (const [from, to] of ranges) {
    if (analysed >= MAX_ANALYSIS_SECONDS) break;
    const start = Math.max(0, Math.floor(from * rate));
    const stop = Math.min(data.length, Math.floor(to * rate));
    for (let at = start; at + FFT_SIZE < stop; at += FFT_SIZE / 2) {
      const re = new Float64Array(FFT_SIZE);
      const im = new Float64Array(FFT_SIZE);
      let energy = 0;
      for (let i = 0; i < FFT_SIZE; i++) {
        const v = data[at + i];
        energy += v * v;
        re[i] = v * window[i];
      }
      if (Math.sqrt(energy / FFT_SIZE) < 0.01) continue;   // skip near-silence
      fft(re, im);
      const bands = new Float64Array(BAND_COUNT);
      for (let b = 0; b < BAND_COUNT; b++) {
        let acc = 0;
        for (let k = edges[b]; k < edges[b + 1]; k++) acc += re[k] * re[k] + im[k] * im[k];
        bands[b] = Math.log(acc / Math.max(1, edges[b + 1] - edges[b]) + 1e-10);
      }
      frames.push(bands);
      analysed += FFT_SIZE / 2 / rate;
      if (analysed >= MAX_ANALYSIS_SECONDS) break;
    }
  }

  if (frames.length < 8) return null;   // too little voiced audio to judge
  for (const bands of frames) for (let b = 0; b < BAND_COUNT; b++) sum[b] += bands[b];

  const vec = new Float64Array(BAND_COUNT);
  let mean = 0;
  for (let b = 0; b < BAND_COUNT; b++) { vec[b] = sum[b] / frames.length; mean += vec[b]; }
  mean /= BAND_COUNT;
  let norm = 0;
  for (let b = 0; b < BAND_COUNT; b++) { vec[b] -= mean; norm += vec[b] * vec[b]; }
  norm = Math.sqrt(norm) || 1;
  for (let b = 0; b < BAND_COUNT; b++) vec[b] /= norm;
  return vec;
}

const cosine = (a, b) => {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
};

/**
 * units: [{ start, end, speaker, offset }] — `offset` identifies which request the
 * label came from. Rewrites `speaker` so the same voice carries one label throughout.
 * Returns a report for logging/tests.
 */
function unifySpeakers(units, buffer, override = null) {
  const timed = units.filter((u) => u.speaker && u.start != null);
  if (!timed.length) return null;

  const groups = new Map();
  for (const unit of timed) {
    const key = `${unit.offset}|${unit.speaker}`;
    if (!groups.has(key)) {
      groups.set(key, { key, part: unit.offset, label: unit.speaker, first: unit.start, ranges: [] });
    }
    const group = groups.get(key);
    group.ranges.push([unit.start, unit.end]);
    group.first = Math.min(group.first, unit.start);
  }
  const parts = new Set([...groups.values()].map((g) => g.part));
  if (parts.size < 2) return null;   // one request: labels are already consistent

  const prints = [];
  for (const group of groups.values()) {
    const vec = fingerprint(buffer, group.ranges);
    if (vec) prints.push({ ...group, vec });
  }
  if (prints.length < 2) return null;

  // Calibrate on this recording: two labels inside ONE request are, by construction,
  // two different people — so the highest similarity among those pairs is what
  // "different voices, same microphone" looks like here. Anything above it is one voice.
  let sameRequestMax = null;
  for (let i = 0; i < prints.length; i++) {
    for (let j = i + 1; j < prints.length; j++) {
      if (prints[i].part !== prints[j].part) continue;
      const score = cosine(prints[i].vec, prints[j].vec);
      if (sameRequestMax == null || score > sameRequestMax) sameRequestMax = score;
    }
  }
  const threshold = override != null
    ? override
    : (sameRequestMax != null ? Math.min(0.995, sameRequestMax + MARGIN) : FALLBACK_THRESHOLD);

  // Agglomerative merge. Two labels from the SAME request are by definition different
  // people, so they must never land in the same cluster.
  let clusters = prints.map((p) => ({
    members: [p], parts: new Set([p.part]), vec: p.vec, first: p.first, weight: 1,
  }));

  for (;;) {
    let best = null;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        let clash = false;
        for (const part of clusters[j].parts) if (clusters[i].parts.has(part)) { clash = true; break; }
        if (clash) continue;
        const score = cosine(clusters[i].vec, clusters[j].vec);
        if (!best || score > best.score) best = { i, j, score };
      }
    }
    if (!best || best.score < threshold) break;

    const a = clusters[best.i];
    const b = clusters[best.j];
    const vec = new Float64Array(a.vec.length);
    let norm = 0;
    for (let k = 0; k < vec.length; k++) {
      vec[k] = (a.vec[k] * a.weight + b.vec[k] * b.weight) / (a.weight + b.weight);
      norm += vec[k] * vec[k];
    }
    norm = Math.sqrt(norm) || 1;
    for (let k = 0; k < vec.length; k++) vec[k] /= norm;

    clusters[best.i] = {
      members: [...a.members, ...b.members],
      parts: new Set([...a.parts, ...b.parts]),
      vec,
      first: Math.min(a.first, b.first),
      weight: a.weight + b.weight,
    };
    clusters = clusters.filter((_, n) => n !== best.j);
  }

  // Name the voices in the order they are first heard.
  clusters.sort((x, y) => x.first - y.first);
  const rename = new Map();
  clusters.forEach((cluster, n) => {
    const name = String.fromCharCode(65 + n);
    for (const member of cluster.members) rename.set(member.key, name);
  });

  for (const unit of units) {
    if (!unit.speaker) continue;
    const mapped = rename.get(`${unit.offset}|${unit.speaker}`);
    if (mapped) unit.speaker = mapped;
  }

  return {
    voices: clusters.length,
    prints: prints.length,
    threshold: +threshold.toFixed(3),
    calibratedFrom: sameRequestMax == null ? 'default' : +sameRequestMax.toFixed(3),
    mapping: [...rename.entries()].map(([key, name]) => `${key} → ${name}`),
  };
}

window.voiceprint = { unifySpeakers, fingerprint, cosine };
