'use strict';

const $ = (id) => document.getElementById(id);

const DIARIZE_MODEL = 'gpt-4o-transcribe-diarize';
const TARGET_RATE = 16000;   // what the speech models use internally
const BIN_HZ = 10;           // loudness envelope resolution: one bin per 100 ms
const DIRECT_LIMIT = 24 * 1024 * 1024;  // send the file untouched below this
// Chunk lengths come from the server so a deployment can shorten them to fit a hosting
// proxy's request timeout. The diarizing model needs roughly half the audio's duration to
// process, so long chunks hit OpenAI's own gateway timeout (a 10-minute chunk returns 500).
let CHUNK_SECONDS = 600;     // 10 min @16 kHz mono ≈ 19 MB — under the 25 MB request limit
let DIARIZE_CHUNK_SECONDS = 180;
const REQUEST_CONCURRENCY = 3;
const SPLIT_SLACK = 25;      // seconds either side of a target cut we may move to find silence
const MIN_UNCOVERED = 2.0;   // ignore shorter untranscribed stretches in the self-check
const UNCERTAIN = 0.5;       // token probability below which a word is worth checking

const state = {
  file: null, result: null, busy: false, audio: null, textModel: null,
  abort: null, startedAt: 0, statusText: '', ticker: null,
};

/* ---------------------------------------------------------------- helpers */
const fmtBytes = (n) => {
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
};

const fmtClock = (s) => {
  if (s == null || Number.isNaN(s)) return '';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
           : `${m}:${String(sec).padStart(2, '0')}`;
};

const stamp = (s, comma) => {
  const ms = Math.floor((s % 1) * 1000);
  const h = String(Math.floor(s / 3600)).padStart(2, '0');
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const sec = String(Math.floor(s % 60)).padStart(2, '0');
  return `${h}:${m}:${sec}${comma ? ',' : '.'}${String(ms).padStart(3, '0')}`;
};

// Hues far enough apart to stay distinct, in assignment order.
const HUES = [212, 145, 278, 28, 330, 188, 95, 352, 258, 48];

// Languages start further along the wheel, so a language chip and a speaker chip are
// never the same colour even before the filled/outlined distinction.
const LANG_OFFSET = 4;

function hueFor(map, key, offset = 0) {
  if (!map.has(key)) map.set(key, HUES[(map.size + offset) % HUES.length]);
  return map.get(key);
}

/** The hue a key would get, without claiming it — building a menu must not assign colours. */
function previewHue(map, key, offset = 0) {
  return map.has(key) ? map.get(key) : HUES[(map.size + offset) % HUES.length];
}

function chip(text, hue, kind) {
  const el = document.createElement('span');
  el.className = `badge ${kind}`;
  el.style.setProperty('--hue', hue);
  el.textContent = text;
  return el;
}

const langNames = (() => {
  try { return new Intl.DisplayNames([navigator.language || 'en'], { type: 'language' }); }
  catch { return null; }
})();
const langName = (code) => {
  if (!code || code === 'und') return 'unknown';
  try { return langNames?.of(code) || code; } catch { return code; }
};

function paintStatus() {
  const elapsed = state.startedAt ? Math.round((Date.now() - state.startedAt) / 1000) : 0;
  $('status').textContent = elapsed > 4
    ? `${state.statusText}  (${fmtClock(elapsed)} elapsed)`
    : state.statusText;
}

function setStatus(text, pct) {
  $('progress').classList.remove('hidden');
  state.statusText = text;
  paintStatus();
  if (pct != null) $('barFill').style.width = `${Math.max(2, Math.min(100, pct))}%`;
}

/** A long transcription is indistinguishable from a hang without a running clock. */
function startClock() {
  state.startedAt = Date.now();
  state.abort = new AbortController();
  clearInterval(state.ticker);
  state.ticker = setInterval(paintStatus, 1000);
  $('cancelBtn').disabled = false;
}

function stopClock() {
  clearInterval(state.ticker);
  state.ticker = null;
  state.startedAt = 0;
  state.abort = null;
  $('progress').classList.add('hidden');
}

function showError(message) {
  $('error').textContent = message;
  $('error').classList.remove('hidden');
}

/** Run tasks with a concurrency cap, keeping results in input order. */
async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  }));
  return out;
}

/** Cancelling must stop local work too, not just the requests in flight. */
function throwIfCancelled() {
  if (state.abort && state.abort.signal.aborted) throw new DOMException('Cancelled', 'AbortError');
}

function seek(seconds) {
  const player = $('player');
  if (player.src && seconds != null) { player.currentTime = seconds; player.play(); }
}

/* --------------------------------------------------------- audio analysis */
function encodeWav(channelData, sampleRate) {
  const n = channelData.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buffer);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); view.setUint32(4, 36 + n * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); str(36, 'data'); view.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * RMS loudness per 100 ms. Sampled with a stride rather than every sample, and yielding
 * to the browser between slices: scanning every sample of a long recording on the UI
 * thread freezes the tab for tens of seconds.
 */
async function computeEnvelope(buffer) {
  const data = buffer.getChannelData(0);          // channel 0 is enough to find speech
  const binSize = Math.max(1, Math.round(buffer.sampleRate / BIN_HZ));
  const stride = Math.max(1, Math.floor(binSize / 160));
  const bins = Math.ceil(buffer.length / binSize);
  const env = new Float32Array(bins);

  for (let b = 0; b < bins; b++) {
    const from = b * binSize;
    const to = Math.min(buffer.length, from + binSize);
    let sum = 0;
    let n = 0;
    for (let i = from; i < to; i += stride) { sum += data[i] * data[i]; n++; }
    env[b] = Math.sqrt(sum / Math.max(1, n));
    if ((b & 4095) === 4095) {
      throwIfCancelled();
      await new Promise((r) => setTimeout(r, 0));  // let the UI breathe
    }
  }
  return env;
}

async function ensureAudio(file) {
  if (state.audio && state.audio.file === file) return state.audio;

  const known = $('player').duration;
  const seconds = Number.isFinite(known) ? known : file.size / 32000;  // rough guess
  // Decoding at the file's own rate is an order of magnitude faster, but an hour of
  // 48 kHz stereo needs well over a gigabyte. Past ~15 minutes, decode through a 16 kHz
  // context instead: the browser resamples as it decodes, at a third of the memory.
  const heavy = seconds > 15 * 60;
  const ctx = heavy ? new OfflineAudioContext(1, 1, TARGET_RATE) : new AudioContext();

  let buffer;
  try {
    if (heavy) setStatus('Reading the audio for the self-check…');
    buffer = await ctx.decodeAudioData(await file.arrayBuffer());
  } catch (err) {
    throw new Error(
      `This file could not be decoded in the browser (${err.message || 'unsupported format'}). ` +
      'Convert it to mp3, m4a or wav and try again.',
    );
  } finally {
    if (!heavy) ctx.close();
  }

  state.audio = { file, buffer, env: await computeEnvelope(buffer), duration: buffer.duration };
  return state.audio;
}

/** Stretches of the recording that carry sound, independent of what any model returned. */
function speechRegions(env) {
  const sorted = Float64Array.from(env).sort();
  const noise = sorted[Math.floor(sorted.length * 0.2)] || 0;
  const loud = sorted[Math.floor(sorted.length * 0.95)] || 0;
  const threshold = Math.max(noise * 2.5, loud * 0.04, 0.003);
  const hangover = Math.round(0.5 * BIN_HZ);
  const minRun = Math.round(0.3 * BIN_HZ);

  const regions = [];
  let i = 0;
  while (i < env.length) {
    if (env[i] <= threshold) { i++; continue; }
    let j = i;
    let quiet = 0;
    while (j < env.length && quiet < hangover) {
      j++;
      quiet = env[j] > threshold ? 0 : quiet + 1;
    }
    const end = Math.min(env.length, j);
    if (end - i >= minRun) regions.push([i / BIN_HZ, end / BIN_HZ]);
    i = end;
  }
  return regions;
}

/** Speech intervals inside one chunk, relative to its start (empty if not decoded). */
function speechWithin(start, span) {
  if (!state.audio || !span) return [];
  const end = start + span;
  return speechRegions(state.audio.env)
    .filter(([a, b]) => b > start && a < end)
    .map(([a, b]) => [Math.max(a, start) - start, Math.min(b, end) - start]);
}

/** Move a cut to the quietest 300 ms nearby, so a chunk never breaks mid-word. */
function findQuietCut(env, target, duration) {
  const width = Math.max(1, Math.round(0.3 * BIN_HZ));
  const lo = Math.max(0, Math.floor((target - SPLIT_SLACK) * BIN_HZ));
  const hi = Math.min(env.length, Math.ceil((target + SPLIT_SLACK) * BIN_HZ));
  let best = target;
  let bestLevel = Infinity;
  for (let i = lo; i + width <= hi; i++) {
    let sum = 0;
    for (let j = i; j < i + width; j++) sum += env[j];
    const level = sum / width;
    if (level < bestLevel) { bestLevel = level; best = (i + width / 2) / BIN_HZ; }
  }
  return Math.min(duration, Math.max(0, best));
}

async function renderSlice(audioBuffer, start, duration) {
  const frames = Math.max(1, Math.ceil(duration * TARGET_RATE));
  const ctx = new OfflineAudioContext(1, frames, TARGET_RATE);
  const src = ctx.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(ctx.destination);
  src.start(0, start, duration);
  const rendered = await ctx.startRendering();
  return encodeWav(rendered.getChannelData(0), TARGET_RATE);
}

/**
 * Plans the requests. One request whenever the file fits — these models transcribe
 * code-switching audio far better with the whole recording in front of them. Oversized
 * files are cut on silence, and diarized ones are cut by duration as well, because that
 * model is too slow to be given long chunks.
 */
async function planChunks(file, opts) {
  // With speakers on, every chunk is also sent to the diarizing model, which times out on
  // long audio — so the shorter limit governs.
  const target = (opts.model.includes('diarize') || opts.speakers)
    ? DIARIZE_CHUNK_SECONDS
    : CHUNK_SECONDS;
  const known = $('player').duration;
  const overLimit = file.size > DIRECT_LIMIT;
  const overTime = Number.isFinite(known) && known > target * 1.25;

  if (!overLimit && !overTime) {
    // Speaker mapping needs the loudness envelope, so decode even a short file.
    if (opts.speakers) { try { await ensureAudio(file); } catch { /* mapping falls back to linear */ } }
    return [{ index: 0, blob: file, name: file.name, offset: 0, span: known || 0 }];
  }

  setStatus('Decoding the audio so it can be split on silence…', 4);
  const { buffer, env, duration } = await ensureAudio(file);

  const cuts = [0];
  while (cuts[cuts.length - 1] + target < duration) {
    const aim = cuts[cuts.length - 1] + target;
    const cut = findQuietCut(env, aim, duration);
    cuts.push(cut > cuts[cuts.length - 1] + 1 ? cut : aim);
  }
  cuts.push(duration);

  const base = file.name.replace(/\.[^.]+$/, '') || 'audio';
  const plans = [];
  for (let i = 0; i < cuts.length - 1; i++) {
    if (cuts[i + 1] - cuts[i] <= 0.05) continue;
    plans.push({
      index: plans.length,
      buffer,
      start: cuts[i],
      span: cuts[i + 1] - cuts[i],
      name: `${base}-part${plans.length + 1}.wav`,
      offset: cuts[i],
    });
  }
  return plans;
}

/* ------------------------------------------------------------- transcribing */
async function sendAudio(blob, name, opts) {
  const res = await fetch('/api/transcribe', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/octet-stream',
      'x-filename': encodeURIComponent(name),
      'x-model': opts.model,
      'x-language': encodeURIComponent(opts.language || ''),
      'x-prompt': encodeURIComponent(opts.prompt || ''),
    },
    body: blob,
    signal: state.abort ? state.abort.signal : undefined,
  });
  const data = await res.json().catch(() => ({ error: 'Unreadable response from the server.' }));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
  return data;
}

/**
 * The diarizing model loses a large share of the words (measured: 36 of 60 utterances,
 * versus 60 of 60 for gpt-4o-transcribe), so it is never used for the transcript itself.
 * It is asked only WHO speaks WHEN, and those turns are mapped onto the complete text.
 *
 * Position in the text maps to position in the *speech*, not the wall clock: silence is
 * excluded, so a long pause does not shift every later attribution.
 */
function mapTextToSpeech(speechIntervals, span) {
  const intervals = speechIntervals.length ? speechIntervals : [[0, span]];
  const total = intervals.reduce((sum, [a, b]) => sum + (b - a), 0) || span;
  return (fraction) => {
    let target = Math.max(0, Math.min(1, fraction)) * total;
    for (const [a, b] of intervals) {
      const length = b - a;
      if (target <= length) return a + target;
      target -= length;
    }
    return intervals[intervals.length - 1][1];
  };
}

function speakerAt(segments, time) {
  for (const seg of segments) if (time >= seg.start && time <= seg.end) return seg.speaker;
  let best = null;
  let bestGap = Infinity;
  for (const seg of segments) {
    const gap = time < seg.start ? seg.start - time : time - seg.end;
    if (gap < bestGap) { bestGap = gap; best = seg; }
  }
  return best ? best.speaker : null;
}

/** Give each unit an estimated time and the speaker talking at that moment. */
function attachSpeakers(units, segments, { start, span, speech }) {
  if (!units.length) return;
  const chars = units.reduce((n, u) => n + u.text.length, 0) || 1;
  const toTime = mapTextToSpeech(speech, span);
  let seen = 0;
  for (const unit of units) {
    const from = toTime(seen / chars);
    seen += unit.text.length;
    const to = toTime(seen / chars);
    unit.start = start + from;
    unit.end = start + Math.max(to, from + 0.2);
    unit.estimated = true;
    if (segments.length) unit.speaker = speakerAt(segments, (from + to) / 2);
  }
}

/** Split a chunk's plain text into units when the model returns no timestamps. */
const ABBREVIATION = /(?:\b(?:dr|mr|mrs|ms|prof|sig|st|vs|etc|no|fig|approx)|\b[a-z])\.$/i;

function sentenceUnits(text, offset) {
  // Models often omit the space after a full stop where the language switches
  // ("…your case.Buongiorno a tutti"), so split on the punctuation itself.
  const pieces = text
    .split(/\n+/)
    .flatMap((line) => line.split(/(?<=[.!?…])/))
    .map((t) => t.trim())
    .filter(Boolean);

  const out = [];
  for (const piece of pieces) {
    const prev = out[out.length - 1];
    const prevIsFragment = prev && (ABBREVIATION.test(prev) || prev.split(/\s+/).length < 3);
    if (prevIsFragment) out[out.length - 1] = `${prev} ${piece}`;
    else out.push(piece);
  }
  return out.map((t) => ({ text: t, start: null, end: null, speaker: null, offset }));
}

/**
 * The tokens concatenate back into the part's text, so walking them alongside the units
 * gives each unit the confidence of the least certain word in it.
 */
function attachConfidence(units, tokens) {
  if (!tokens || !tokens.length) return;
  const spans = [];
  let at = 0;
  for (const token of tokens) {
    spans.push([at, at + token.token.length]);
    at += token.token.length;
  }
  const stream = tokens.map((t) => t.token).join('');

  let cursor = 0;
  for (const unit of units) {
    const found = stream.indexOf(unit.text, cursor);
    if (found === -1) continue;
    const end = found + unit.text.length;
    cursor = end;
    const covering = tokens.filter((_, i) => spans[i][1] > found && spans[i][0] < end);
    if (!covering.length) continue;
    unit.minP = Math.min(...covering.map((t) => t.p));
    unit.uncertain = covering
      .filter((t) => t.p < UNCERTAIN)
      .map((t) => t.token.trim())
      .filter(Boolean);
  }
}

function rebuild(r) {
  r.units = r.parts.flatMap((p) => p.units).map((u, i) => ({ ...u, i }));
  r.segments = r.units.filter((u) => u.start != null);
  r.text = r.parts.map((p) => p.text).filter(Boolean).join('\n\n');
  return r;
}

/** Transcribe the loaded file with the given options, chunking only when necessary. */
async function runTranscription(opts, note) {
  const plans = await planChunks(state.file, opts);
  let done = 0;
  const report = () => setStatus(
    plans.length > 1
      ? `${note} — ${done} of ${plans.length} parts done`
      : `${note}… this can take a while for long recordings.`,
    20 + (done / plans.length) * 70,
  );
  report();

  const parts = await mapLimit(plans, REQUEST_CONCURRENCY, async (plan) => {
    throwIfCancelled();
    // Slices are rendered inside the worker so only a few exist in memory at once.
    const blob = plan.blob || await renderSlice(plan.buffer, plan.start, plan.span);

    // The transcript always comes from the verbatim model. When speakers are wanted the
    // diarizing model runs on the same audio, but only its turns are used.
    const [data, turns] = await Promise.all([
      sendAudio(blob, plan.name, opts),
      opts.speakers
        ? sendAudio(blob, plan.name, { ...opts, model: DIARIZE_MODEL }).catch((err) => {
            console.warn('[speakers] diarization failed for this part:', err.message);
            return null;
          })
        : null,
    ]);
    done++;
    report();

    const offset = plan.offset;
    const text = (data.text || '').trim();
    const units = data.segments.length
      ? data.segments.map((seg) => ({
          text: seg.text,
          start: offset + seg.start,
          end: offset + seg.end,
          speaker: seg.speaker,
          offset,
        }))
      : sentenceUnits(text, offset);
    attachConfidence(units, data.logprobs);

    if (turns && turns.segments.length && !data.segments.length) {
      attachSpeakers(units, turns.segments, {
        start: offset,
        span: plan.span || data.duration || turns.duration || 0,
        speech: speechWithin(offset, plan.span || turns.duration || 0),
      });
    }
    return { offset, text, units, language: data.language || null, duration: data.duration ?? plan.span };
  });

  const duration = parts.reduce((end, p) => Math.max(end, p.offset + (p.duration || 0)), 0);
  return { parts, duration };
}

async function transcribe() {
  if (state.busy || !state.file) return;
  state.busy = true;
  state.result = null;
  $('go').disabled = true;
  $('error').classList.add('hidden');
  $('result').classList.add('hidden');
  startClock();
  setStatus('Preparing audio…', 3);

  const opts = {
    model: $('model').value,
    language: $('language').value,
    prompt: $('prompt').value.trim(),
    speakers: $('speakers').checked,
  };

  try {
    const { parts, duration } = await runTranscription(opts, 'Transcribing');
    const result = rebuild({
      parts,
      duration,
      model: opts.model,
      source: state.file.name,
      language: parts.find((p) => p.language)?.language || null,
      paragraphs: null,
      structureError: null,
      check: null,
      crossCheck: null,
    });
    state.result = result;

    // Speaker labels are assigned per request, so "Speaker A" in part 1 and in part 5 are
    // not necessarily the same person. Re-identify the voices across the whole recording.
    if (result.parts.length > 1 && state.audio && window.voiceprint) {
      setStatus('Matching voices across the parts…', 91);
      try {
        const report = voiceprint.unifySpeakers(result.units, state.audio.buffer);
        if (report) {
          result.voices = report;
          console.info('[voices]', report.voices, 'distinct, threshold', report.threshold,
                       '(calibrated from', report.calibratedFrom + ')');
        }
      } catch (err) {
        console.warn('[voices] matching failed, keeping per-part labels:', err);
      }
    }
    orderSpeakers(result.units);   // whoever speaks first is Speaker A

    if ($('structure').checked && result.units.length) {
      setStatus('Splitting into paragraphs by language…', 93);
      try {
        const res = await fetch('/api/structure', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ units: result.units.map((u) => ({ i: u.i, text: u.text })) }),
          signal: state.abort ? state.abort.signal : undefined,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || `Paragraphing failed (${res.status}).`);
        if (data.units && data.units.length) adoptUnits(result, data.units);
        result.paragraphs = splitOnSpeaker(result, data.paragraphs);
      } catch (err) {
        result.structureError = err.message || String(err); // the transcript itself is fine
      }
    }

    setStatus('Done.', 100);
    stopClock();
    renderResult();
    runSelfCheck();
  } catch (err) {
    stopClock();
    if (err.name === 'AbortError') showError('Cancelled.');
    else showError(err.message || String(err));
  } finally {
    state.busy = false;
    $('go').disabled = !state.file;
  }
}

/**
 * The paragraphing pass may split a unit that contained two languages. Those pieces
 * inherit the original unit's speaker, and share its time span in proportion to their
 * length. r.segments is left untouched so subtitles keep the model's own timings.
 */
function adoptUnits(r, returned) {
  const groups = [];
  for (const unit of returned) {
    const last = groups[groups.length - 1];
    if (last && last.source === unit.source) last.pieces.push(unit);
    else groups.push({ source: unit.source, pieces: [unit] });
  }

  const units = [];
  for (const group of groups) {
    const origin = r.units[group.source];
    const span = origin && origin.start != null && origin.end != null
      ? origin.end - origin.start
      : null;
    const total = group.pieces.reduce((n, p) => n + p.text.length, 0) || 1;
    let cursor = origin ? origin.start : null;

    for (const piece of group.pieces) {
      const share = span == null ? null : span * (piece.text.length / total);
      units.push({
        i: units.length,
        text: piece.text,
        lang: piece.lang,
        start: cursor,
        end: share == null ? null : cursor + share,
        speaker: origin ? origin.speaker : null,
        offset: origin ? origin.offset : 0,
        minP: origin ? origin.minP : undefined,
        uncertain: origin ? origin.uncertain : undefined,
      });
      if (share != null) cursor += share;
    }
  }
  r.units = units;
}

/**
 * The first voice heard is always Speaker A, the second B, and so on. The API's own
 * letters are per request and arbitrary, so the labels are re-lettered here by the time
 * each voice first speaks.
 */
function orderSpeakers(units) {
  const firstHeard = new Map();
  for (const unit of units) {
    if (!unit.speaker || unit.start == null) continue;
    const seen = firstHeard.get(unit.speaker);
    if (seen == null || unit.start < seen) firstHeard.set(unit.speaker, unit.start);
  }
  if (!firstHeard.size) return null;

  const rename = new Map([...firstHeard.entries()]
    .sort((a, b) => a[1] - b[1])
    .map(([label], i) => [label, String.fromCharCode(65 + i)]));
  for (const unit of units) {
    if (unit.speaker && rename.has(unit.speaker)) unit.speaker = rename.get(unit.speaker);
  }
  return rename;
}

/** With a diarizing model, a change of speaker also starts a new paragraph. */
function splitOnSpeaker(r, paragraphs) {
  const out = [];
  for (const para of paragraphs) {
    let run = [];
    let speaker = null;
    for (const i of para.units) {
      const s = r.units[i]?.speaker || null;
      if (run.length && s !== speaker) {
        out.push({ lang: para.lang, units: run, speaker });
        run = [];
      }
      speaker = s;
      run.push(i);
    }
    if (run.length) out.push({ lang: para.lang, units: run, speaker });
  }
  return out.map((p, index) => ({ ...p, index }));
}

/* ----------------------------------------------------------------- rendering */
const unitText = (r, i) => (r.units[i] ? r.units[i].text : '');
const originalText = (r, p) => p.units.map((i) => unitText(r, i)).join(' ').trim();
const paragraphText = (r, p) => (p.editedText != null ? p.editedText : originalText(r, p));
const paragraphStart = (r, p) => {
  const first = r.units[p.units[0]];
  if (first && first.start != null) return first.start;
  return p.manualStart != null ? p.manualStart : null;   // a hand-added row
};
const paragraphEnd = (r, p) => {
  const last = r.units[p.units[p.units.length - 1]];
  return last && last.end != null ? last.end : null;
};

function renderParagraphs(r) {
  const box = $('viewText');
  box.innerHTML = '';
  const multilingual = new Set(r.paragraphs.map((p) => p.lang)).size > 1;
  const hasSpeakers = r.voiceHues.size > 0;
  let previousLang = null;

  r.paragraphs.forEach((para, position) => {
    box.append(insertStrip(r, position));
    const el = document.createElement('div');
    el.className = previousLang && para.lang !== previousLang ? 'para switch' : 'para';
    if (para.added) el.classList.add('added');

    // Language, speaker and time live in their own column to the left of the text.
    const meta = document.createElement('div');
    meta.className = 'paraMeta';
    if (multilingual || para.added) {
      const langChip = chip(langName(para.lang), hueFor(r.langHues, para.lang, LANG_OFFSET), 'lang');
      langChip.classList.add('editableChip');
      langChip.title = 'Click to change the language of this paragraph';
      langChip.onclick = (event) => {
        event.stopPropagation();
        showPicker(langChip, languageOptions(r, para.lang), (value) => {
          para.lang = value;
          renderResult();
        });
      };
      meta.append(langChip);
    }
    if (para.speaker || (para.added && hasSpeakers)) {
      if (!para.speaker) para.speaker = [...r.voiceHues.keys()][0];
      const voiceChip = chip(`Speaker ${para.speaker}`, hueFor(r.voiceHues, para.speaker), 'voice');
      voiceChip.classList.add('editableChip');
      voiceChip.title = 'Click to reassign this paragraph to another speaker';
      voiceChip.onclick = (event) => {
        event.stopPropagation();
        showPicker(voiceChip, speakerOptions(r, para.speaker), (value) => {
          para.speaker = value;
          // Keep the segment list and the exports in step with the correction.
          for (const i of para.units) if (r.units[i]) r.units[i].speaker = value;
          renderResult();
        });
      };
      meta.append(voiceChip);
    }
    const at = paragraphStart(r, para);
    if (at != null) {
      const t = document.createElement('time');
      t.textContent = fmtClock(at);
      t.onclick = () => seek(at);
      meta.append(t);
    }
    if (!meta.childElementCount) el.classList.add('bare');
    el.append(meta);

    const body = document.createElement('p');
    body.className = 'paraBody';
    body.textContent = paragraphText(r, para);
    // plaintext-only is not in every browser; falling back keeps the editor usable.
    try { body.contentEditable = 'plaintext-only'; } catch { body.contentEditable = 'true'; }
    if (body.contentEditable !== 'plaintext-only') {
      body.addEventListener('paste', (event) => {
        event.preventDefault();
        const text = (event.clipboardData || window.clipboardData).getData('text');
        document.execCommand('insertText', false, text);
      });
    }
    body.spellcheck = false;
    body.title = 'Click to correct the text';
    body.addEventListener('blur', () => {
      const text = body.textContent.trim();
      para.editedText = text === originalText(r, para) ? null : text;
      markEdited(r, para, el);
      $('resultMeta').textContent = metaLine(r);
    });
    el.append(body);
    box.append(el);

    const shaky = para.units
      .map((i) => r.units[i])
      .filter((u) => u && u.minP != null && u.minP < UNCERTAIN);
    if (shaky.length) {
      const words = [...new Set(shaky.flatMap((u) => u.uncertain || []))].slice(0, 8);
      const worst = Math.min(...shaky.map((u) => u.minP));
      const flag = chip(`⚠ check`, 40, 'lowconf');
      flag.title = words.length
        ? `The model was unsure of: ${words.join(', ')} (lowest confidence ${Math.round(worst * 100)}%). ` +
          'Click to listen.'
        : `Lowest word confidence in this paragraph: ${Math.round(worst * 100)}%. Click to listen.`;
      const at = paragraphStart(r, para);
      if (at != null) { flag.style.cursor = 'pointer'; flag.onclick = (e) => { e.stopPropagation(); seek(at); }; }
      meta.append(flag);
    }

    const mark = document.createElement('label');
    mark.className = 'interpMark';
    mark.title = 'Tick when this passage is the interpreter speaking, not the original speaker';
    const tick = document.createElement('input');
    tick.type = 'checkbox';
    tick.checked = para.interpreter === true;
    tick.onclick = (event) => event.stopPropagation();
    tick.onchange = () => {
      para.interpreter = tick.checked;
      el.classList.toggle('interp', tick.checked);
      $('resultMeta').textContent = metaLine(r);
      renderLegend(r);
      renderSegments(r);
    };
    mark.append(tick, document.createTextNode('interpreter'));
    meta.append(mark);
    if (para.interpreter) el.classList.add('interp');

    if (para.added) {
      const controls = document.createElement('div');
      controls.className = 'addedControls';

      const tag = document.createElement('span');
      tag.className = 'badge added';
      tag.title = 'This paragraph was added by hand — it is not in the audio transcript';
      tag.textContent = '+ added';

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'removeRow';
      remove.title = 'Remove this paragraph';
      remove.textContent = '×';
      remove.onclick = (event) => { event.stopPropagation(); removeParagraph(r, para.index); };

      controls.append(tag, remove);
      meta.append(controls);
    }

    markEdited(r, para, el);
    previousLang = para.lang;
  });

  box.append(insertStrip(r, r.paragraphs.length));
}

/* ------------------------------------------------------------------ editing */
let openPicker = null;

function closePicker() {
  if (openPicker) { openPicker.remove(); openPicker = null; }
}

/** Small menu anchored under a chip. options: [{ value, label, hue, kind, current }] */
function showPicker(anchor, options, onPick) {
  closePicker();
  const menu = document.createElement('div');
  menu.className = 'menuList picker';
  for (const option of options) {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = option.current ? 'current' : '';
    if (option.hue != null) item.append(chip(option.label, option.hue, option.kind));
    else item.append(document.createTextNode(option.label));
    item.onclick = (event) => { event.stopPropagation(); closePicker(); onPick(option.value); };
    menu.append(item);
  }
  const box = anchor.getBoundingClientRect();
  menu.style.top = `${box.bottom + window.scrollY + 4}px`;
  menu.style.left = `${box.left + window.scrollX}px`;
  document.body.append(menu);
  openPicker = menu;
}

/** The languages offered: those already in the transcript, plus the form's list. */
function languageOptions(r, current) {
  const seen = new Map();
  for (const para of r.paragraphs) seen.set(para.lang, langName(para.lang));
  for (const option of $('language').options) {
    if (option.value) seen.set(option.value, option.textContent.trim());
  }
  return [...seen].map(([value, label]) => ({
    value,
    label,
    kind: 'lang',
    hue: previewHue(r.langHues, value, LANG_OFFSET),
    current: value === current,
  }));
}

/** Existing voices plus one spare label, so a misattributed passage can be moved. */
function speakerOptions(r, current) {
  const letters = [...r.voiceHues.keys()].sort();
  const next = String.fromCharCode(65 + letters.length);
  return [...letters, next].map((value) => ({
    value,
    label: `Speaker ${value}`,
    kind: 'voice',
    hue: previewHue(r.voiceHues, value),
    current: value === current,
  }));
}

/** A paragraph whose text was changed by hand is marked, and stays marked in the exports. */
function markEdited(r, para, el) {
  if (para.added) return;
  const edited = para.editedText != null && para.editedText !== originalText(r, para);
  if (!edited) para.editedText = null;
  const meta = el.querySelector('.paraMeta');
  const existing = meta.querySelector('.edited');
  if (edited && !existing) {
    const tag = document.createElement('span');
    tag.className = 'badge edited';
    tag.title = 'This paragraph was edited by hand';
    tag.textContent = '✎ edited';
    meta.append(tag);
  } else if (!edited && existing) {
    existing.remove();
  }
}

/** The confidence of the least certain word in a paragraph (1 when unknown). */
function paragraphConfidence(r, para) {
  const values = para.units.map((i) => r.units[i]).filter((u) => u && u.minP != null).map((u) => u.minP);
  return values.length ? Math.min(...values) : 1;
}

function metaLine(r) {
  const shown = r.paragraphs && r.paragraphs.length
    ? r.paragraphs.map((p) => paragraphText(r, p)).join(' ')
    : r.text;
  const editedCount = (r.paragraphs || []).filter((p) => !p.added && p.editedText != null).length;
  const addedCount = (r.paragraphs || []).filter((p) => p.added).length;
  const bits = [];
  if (r.duration) bits.push(`${fmtClock(r.duration)} of audio`);
  bits.push(`${shown.split(/\s+/).filter(Boolean).length} words`);
  if (r.paragraphs) bits.push(`${r.paragraphs.length} paragraphs`);
  if (editedCount) bits.push(`${editedCount} edited`);
  if (addedCount) bits.push(`${addedCount} added`);
  const interpreted = (r.paragraphs || []).filter((p) => p.interpreter).length;
  if (interpreted) bits.push(`${interpreted} interpreted`);
  const shaky = (r.paragraphs || []).filter((p) => paragraphConfidence(r, p) < UNCERTAIN).length;
  if (shaky) bits.push(`${shaky} to check`);
  bits.push(`model: ${r.model}`);
  if (r.parts.length > 1) bits.push(`${r.parts.length} parts`);
  return bits.join(' · ');
}

/** Insert an empty paragraph at `at`, taking its language and speaker from its neighbour. */
function addParagraph(r, at) {
  const neighbour = r.paragraphs[at - 1] || r.paragraphs[at] || null;
  const previousEnd = neighbour ? paragraphStart(r, neighbour) : 0;
  r.paragraphs.splice(at, 0, {
    lang: neighbour ? neighbour.lang : (r.langHues.keys().next().value || 'en'),
    speaker: neighbour ? neighbour.speaker : ([...r.voiceHues.keys()][0] || null),
    units: [],
    editedText: '',
    added: true,
    manualStart: previousEnd,
  });
  r.paragraphs.forEach((para, i) => { para.index = i; });
  renderResult();
  // Put the cursor straight into the new row.
  const body = document.querySelectorAll('.para')[at]?.querySelector('.paraBody');
  if (body) { body.focus(); body.scrollIntoView({ block: 'center' }); }
}

function removeParagraph(r, at) {
  r.paragraphs.splice(at, 1);
  r.paragraphs.forEach((para, i) => { para.index = i; });
  renderResult();
}

/** A thin strip between paragraphs; clicking it opens a new row at that point. */
function insertStrip(r, at) {
  const strip = document.createElement('div');
  strip.className = 'insertStrip';
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = '+ add a paragraph here';
  button.onclick = (event) => { event.stopPropagation(); addParagraph(r, at); };
  strip.append(button);
  return strip;
}

/** The Timestamps view: one row per timed segment, tagged when marked as interpreting. */
function renderSegments(r) {
  const box = $('viewSegments');
  box.innerHTML = '';
  const interpretedUnits = new Set();
  for (const para of r.paragraphs || []) {
    if (para.interpreter) for (const i of para.units) interpretedUnits.add(i);
  }
  for (const seg of r.segments) {
    const row = document.createElement('div');
    row.className = 'seg';
    const t = document.createElement('time');
    t.textContent = `${fmtClock(seg.start)} – ${fmtClock(seg.end)}`;
    t.onclick = () => seek(seg.start);
    const body = document.createElement('p');
    body.style.margin = '0';
    if (seg.speaker) {
      const who = chip(`Speaker ${seg.speaker}`, hueFor(r.voiceHues, seg.speaker), 'voice');
      who.classList.add('inlineChip');
      body.append(who);
    }
    body.append(document.createTextNode(seg.text));
    if (interpretedUnits.has(seg.i)) {
      const tag = document.createElement('span');
      tag.className = 'badge interpTag';
      tag.textContent = 'interpreter';
      body.append(document.createTextNode(' '), tag);
    }
    row.append(t, body);
    box.append(row);
  }
}

/** A key to the colours: which voice and which language each hue stands for. */
function renderLegend(r) {
  const box = $('legend');
  box.innerHTML = '';
  // Paragraphing is optional, so there may be no paragraphs at all — speakers still exist.
  const paragraphs = r.paragraphs || [];
  let shown = 0;

  if (r.voiceHues.size) {
    const row = document.createElement('div');
    row.className = 'legendRow';
    const label = document.createElement('span');
    label.className = 'legendLabel';
    label.textContent = r.voiceHues.size === 1 ? 'Voice' : `${r.voiceHues.size} voices`;
    row.append(label);
    for (const [speaker, hue] of [...r.voiceHues].sort((a, b) => a[0].localeCompare(b[0]))) {
      const group = document.createElement('span');
      group.className = 'legendVoice';
      group.append(chip(`Speaker ${speaker}`, hue, 'voice'));

      // An interpreter usually interprets throughout, so offer it per voice as well.
      const paras = paragraphs.filter((p) => p.speaker === speaker);
      if (paras.length) {
        const mark = document.createElement('label');
        mark.className = 'interpMark';
        mark.title = `Mark everything Speaker ${speaker} says as interpreting`;
        const tick = document.createElement('input');
        tick.type = 'checkbox';
        tick.checked = paras.every((p) => p.interpreter);
        tick.indeterminate = !tick.checked && paras.some((p) => p.interpreter);
        tick.onchange = () => {
          for (const para of paras) para.interpreter = tick.checked;
          renderParagraphs(r);
          renderLegend(r);
          renderSegments(r);
          $('resultMeta').textContent = metaLine(r);
        };
        mark.append(tick, document.createTextNode('interpreter'));
        group.append(mark);
      }
      row.append(group);
    }
    box.append(row);
    shown++;
  }

  if (r.langHues.size > 1) {
    const row = document.createElement('div');
    row.className = 'legendRow';
    const label = document.createElement('span');
    label.className = 'legendLabel';
    label.textContent = `${r.langHues.size} languages`;
    row.append(label);
    for (const [lang, hue] of r.langHues) row.append(chip(langName(lang), hue, 'lang'));
    box.append(row);
    shown++;
  }

  box.classList.toggle('hidden', shown === 0);
}

function renderResult() {
  const r = state.result;
  if (!r) return;
  $('result').classList.remove('hidden');

  // Assign colours in order of first appearance, shared by every view.
  r.langHues = new Map();
  r.voiceHues = new Map();
  for (const para of r.paragraphs || []) hueFor(r.langHues, para.lang, LANG_OFFSET);
  for (const unit of r.units) if (unit.speaker) hueFor(r.voiceHues, unit.speaker);

  if (r.paragraphs && r.paragraphs.length) renderParagraphs(r);
  else $('viewText').textContent = r.text || '(no speech detected)';

  renderLegend(r);

  $('resultMeta').textContent = metaLine(r);

  if (r.structureError) showError(`Transcript is complete, but paragraphing failed: ${r.structureError}`);

  const hasTimes = r.segments.length > 0;
  document.querySelectorAll('.dl-ts').forEach((b) => b.classList.toggle('hidden', !hasTimes));
  document.querySelector('.tab[data-view="segments"]').classList.toggle('hidden', !hasTimes);

  renderSegments(r);

  renderCheck(r);
}

/* --------------------------------------------------------------- self-check */
/**
 * Audits the transcript against the audio itself rather than against the model's own
 * account of it: how much of the recording carries sound, how much text came back for
 * it, and — when the model gave timestamps — which sounding stretches produced nothing.
 */
function selfCheck(r, audio) {
  if (!audio) return null;
  const regions = speechRegions(audio.env);
  const speech = regions.reduce((sum, [a, b]) => sum + (b - a), 0);
  const words = r.text.split(/\s+/).filter(Boolean).length;
  const rate = speech > 0 ? words / speech : 0;

  let uncovered = null;
  if (r.segments.length) {
    uncovered = [];
    for (const [from, to] of regions) {
      let covered = 0;
      for (const seg of r.segments) {
        if (seg.end <= from || seg.start >= to) continue;
        covered += Math.min(to, seg.end) - Math.max(from, seg.start);
      }
      if (to - from >= MIN_UNCOVERED && covered < (to - from) * 0.25) uncovered.push([from, to]);
    }
  }

  return { duration: audio.duration, speech, words, rate, uncovered };
}

function renderCheck(r) {
  const box = $('coverage');
  const check = r.check;
  box.classList.remove('hidden', 'warn');
  box.innerHTML = '';

  if (!check) {
    box.textContent = 'Checking the transcript against the audio…';
    return;
  }

  const line = document.createElement('div');
  box.append(line);
  const spoken = `${fmtClock(check.speech)} of the ${fmtClock(check.duration)} recording carries sound`;
  const rate = `${check.words} words came back — ${Math.round(check.rate * 60)} per minute of it`;
  const slow = check.rate < 1.4;      // ≈ 85 wpm: low for continuous speech
  const gaps = check.uncovered || [];

  if (!slow && !gaps.length) {
    // Deliberately not "nothing is missing": a word-rate check only catches gross loss.
    line.innerHTML = `<strong>✓ No large omission detected.</strong> ${spoken}, and ${rate}` +
      (check.uncovered ? ', with no sounding stretch left untranscribed' : '') +
      '. That rules out big losses, not individual sentences — the cross-check below is the ' +
      'reliable test, and any word the model itself was unsure of is flagged in the transcript.';
    addCrossCheckUi(r, box);
    return;
  }

  box.classList.add('warn');
  const notes = [];
  if (slow) notes.push(`only ${Math.round(check.rate * 60)} words per minute of sound came back, ` +
    `which is low — speech may have been dropped`);
  if (gaps.length) notes.push(`${gaps.length} stretch${gaps.length === 1 ? '' : 'es'} carry sound ` +
    `but produced no text`);
  line.innerHTML = `<strong>⚠ ${notes.join('; ')}.</strong> ${spoken}. ` +
    `If this is a bilingual recording, whisper-1 is known to transcribe only one language per ` +
    `request — try gpt-4o-transcribe.`;

  addCrossCheckUi(r, box);

  if (gaps.length) {
    const list = document.createElement('ul');
    for (const [from, to] of gaps.slice(0, 10)) {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.className = 'gap';
      btn.type = 'button';
      btn.textContent = `${fmtClock(from)} – ${fmtClock(to)}`;
      btn.onclick = () => seek(from);
      li.append(btn, document.createTextNode(` · ${Math.round(to - from)}s`));
      list.append(li);
    }
    if (gaps.length > 10) {
      const li = document.createElement('li');
      li.textContent = `…and ${gaps.length - 10} more`;
      list.append(li);
    }
    box.append(list);
  }
}

/** A second model on the same audio is the only real check on what the first one heard. */
function addCrossCheckUi(r, box) {
  if (r.crossCheck) { renderCrossCheck(r, box); return; }
  const btn = document.createElement('button');
  btn.className = 'ghost crossBtn';
  btn.type = 'button';
  btn.textContent = `Cross-check with ${VERIFIERS[r.model] || 'gpt-4o-transcribe'}`;
  btn.onclick = () => { btn.disabled = true; crossCheck(); };
  box.append(btn);
}

/** Decoding is slow, so the audit lands a moment after the transcript is on screen. */
async function runSelfCheck() {
  const r = state.result;
  if (!r || !state.file) return;
  // Decoding a marathon recording costs real memory; the transcript is already shown.
  const known = $('player').duration;
  if (Number.isFinite(known) && known > 2 * 3600) {
    $('coverage').textContent =
      'Self-check skipped: decoding a recording this long in the browser would use too much memory. ' +
      'Use the cross-check button to compare against a second model instead.';
    $('coverage').classList.remove('hidden', 'warn');
    return;
  }
  try {
    const audio = await ensureAudio(state.file);
    if (state.result !== r) return;
    r.check = selfCheck(r, audio);
  } catch {
    r.check = null; // undecodable container — the transcript still stands
    $('coverage').classList.add('hidden');
    return;
  } finally {
    $('progress').classList.add('hidden'); // ensureAudio may have shown it again
  }
  renderCheck(r);
}

/* --------------------------------------------------------------- cross-check */
const VERIFIERS = {
  'gpt-4o-transcribe': 'gpt-4o-transcribe-diarize',
  'gpt-4o-mini-transcribe': 'gpt-4o-transcribe',
  'gpt-4o-transcribe-diarize': 'gpt-4o-transcribe',
  'whisper-1': 'gpt-4o-transcribe',
};

const contentWords = (text) => text
  .toLowerCase()
  .normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '')
  .replace(/[^a-z0-9\s]/g, ' ')
  .split(/\s+/)
  .filter((w) => w.length > 2);

/** Sentences the second model heard that this transcript has no trace of. */
function findMissing(primaryText, units) {
  const bag = new Set(contentWords(primaryText));
  return units.filter((u) => {
    const words = contentWords(u.text);
    if (words.length < 3) return false;
    const hits = words.filter((w) => bag.has(w)).length;
    return hits / words.length < 0.5;
  });
}

async function crossCheck() {
  const r = state.result;
  if (!r || state.busy) return;
  const verifier = VERIFIERS[r.model] || 'gpt-4o-transcribe';
  state.busy = true;
  $('go').disabled = true;
  startClock();
  try {
    const { parts } = await runTranscription(
      { model: verifier, language: $('language').value, prompt: $('prompt').value.trim() },
      `Cross-checking with ${verifier}`,
    );
    const units = parts.flatMap((part) => part.units);
    const text = parts.map((part) => part.text).join(' ');
    if (state.result !== r) return;
    r.crossCheck = {
      model: verifier,
      words: text.split(/\s+/).filter(Boolean).length,
      missing: findMissing(r.text, units),
      alsoMissedHere: findMissing(text, r.units),
    };
    stopClock();
    renderCheck(r);
  } catch (err) {
    stopClock();
    if (err.name !== 'AbortError') showError(`Cross-check failed: ${err.message || err}`);
  } finally {
    state.busy = false;
    $('go').disabled = !state.file;
  }
}

function renderCrossCheck(r, box) {
  const cc = r.crossCheck;
  const mine = r.text.split(/\s+/).filter(Boolean).length;
  const wrap = document.createElement('div');
  wrap.className = 'crossResult';

  const head = document.createElement('div');
  head.innerHTML = cc.missing.length
    ? `<strong>${cc.model} heard ${cc.missing.length} passage${cc.missing.length === 1 ? '' : 's'} ` +
      `that ${cc.missing.length === 1 ? 'is' : 'are'} not in this transcript</strong> ` +
      `(${cc.words} words there vs ${mine} here):`
    : `<strong>✓ Nothing missing.</strong> ${cc.model} returned ${cc.words} words against ${mine} ` +
      `here, and every passage it heard is present in this transcript.`;
  wrap.append(head);

  if (cc.missing.length) {
    const list = document.createElement('ul');
    for (const unit of cc.missing.slice(0, 12)) {
      const li = document.createElement('li');
      if (unit.start != null) {
        const btn = document.createElement('button');
        btn.className = 'gap';
        btn.type = 'button';
        btn.textContent = fmtClock(unit.start);
        btn.onclick = () => seek(unit.start);
        li.append(btn, document.createTextNode(' · '));
      }
      li.append(document.createTextNode(unit.text));
      list.append(li);
    }
    wrap.append(list);
    const hint = document.createElement('div');
    hint.textContent = `Re-run with ${cc.model} to capture these.`;
    wrap.append(hint);
  }
  box.append(wrap);
}

/* ------------------------------------------------------------------ exports */
function exportText(r) {
  if (!r.paragraphs || !r.paragraphs.length) return r.text;
  const multilingual = new Set(r.paragraphs.map((p) => p.lang)).size > 1;
  return r.paragraphs.filter((para) => paragraphText(r, para)).map((para) => {
    const tags = [];
    const at = paragraphStart(r, para);
    if (at != null) tags.push(fmtClock(at));
    if (multilingual) tags.push(langName(para.lang));
    if (para.speaker) tags.push(`Speaker ${para.speaker}`);
    if (para.interpreter) tags.push('interpreter');
    if (para.added) tags.push('added');
    else if (para.editedText != null) tags.push('edited');
    return (tags.length ? `[${tags.join(' · ')}]\n` : '') + paragraphText(r, para);
  }).join('\n\n');
}

/** Quote anything containing a comma, quote or newline; double up embedded quotes. */
const csvCell = (value) => {
  const text = value == null ? '' : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

const csvRows = (r) => {
  const rows = [['paragraph', 'start', 'end', 'start_seconds', 'language', 'speaker',
                 'interpreter', 'edited', 'confidence', 'text']];
  if (r.paragraphs && r.paragraphs.length) {
    for (const para of r.paragraphs) {
      const from = paragraphStart(r, para);
      const to = paragraphEnd(r, para);
      rows.push([
        para.index + 1,
        from == null ? '' : fmtClock(from),
        to == null ? '' : fmtClock(to),
        from == null ? '' : from.toFixed(2),
        langName(para.lang),
        para.speaker ? `Speaker ${para.speaker}` : '',
        para.interpreter ? 'yes' : '',
        para.added ? 'added' : (para.editedText != null ? 'yes' : ''),
        para.added ? '' : paragraphConfidence(r, para).toFixed(2),
        paragraphText(r, para),
      ]);
    }
  } else {
    r.units.forEach((u, i) => rows.push([
      i + 1,
      u.start == null ? '' : fmtClock(u.start),
      u.end == null ? '' : fmtClock(u.end),
      u.start == null ? '' : u.start.toFixed(2),
      r.language ? langName(r.language) : '',
      u.speaker ? `Speaker ${u.speaker}` : '',
      '',
      '',
      u.minP != null ? u.minP.toFixed(2) : '',
      u.text,
    ]));
  }
  return rows;
};

const MIME_TYPES = {
  csv: 'text/csv;charset=utf-8',
  json: 'application/json;charset=utf-8',
};

const builders = {
  // \ufeff so Excel opens accented text (è, à, ü) as UTF-8 rather than mojibake.
  csv: (r) => '\ufeff' + csvRows(r).map((row) => row.map(csvCell).join(',')).join('\r\n'),
  txt: (r) => exportText(r),
  json: (r) => JSON.stringify({
    source: r.source,
    model: r.model,
    paragraphModel: state.textModel || null,
    duration: r.duration,
    check: r.check,
    voiceMatching: r.voices || null,
    paragraphs: (r.paragraphs || []).map((p) => ({
      index: p.index,
      language: p.lang,
      speaker: p.speaker || null,
      start: paragraphStart(r, p),
      interpreter: p.interpreter === true,
      added: p.added === true,
      confidence: p.added ? null : +paragraphConfidence(r, p).toFixed(3),
      uncertainWords: [...new Set(p.units.map((i) => r.units[i])
        .filter((u) => u && u.uncertain && u.uncertain.length)
        .flatMap((u) => u.uncertain))],
      edited: !p.added && p.editedText != null,
      text: paragraphText(r, p),
      transcribedText: !p.added && p.editedText != null ? originalText(r, p) : undefined,
    })),
    segments: r.segments.map((s) => ({ start: s.start, end: s.end, speaker: s.speaker, text: s.text })),
    text: r.text,
  }, null, 2),
  srt: (r) => r.segments.map((s, i) =>
    `${i + 1}\n${stamp(s.start, true)} --> ${stamp(s.end, true)}\n${s.text}\n`).join('\n'),
  vtt: (r) => `WEBVTT\n\n${r.segments.map((s) =>
    `${stamp(s.start, false)} --> ${stamp(s.end, false)}\n${s.text}\n`).join('\n')}`,
};

function download(kind) {
  const r = state.result;
  if (!r) return;
  const blob = new Blob([builders[kind](r)], { type: MIME_TYPES[kind] || 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${r.source.replace(/\.[^.]+$/, '')}.${kind}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/* -------------------------------------------------------------------- wiring */
function acceptFile(file) {
  if (!file) return;
  state.file = file;
  state.result = null;
  state.audio = null;
  $('fileCard').classList.remove('hidden');
  $('result').classList.add('hidden');
  $('error').classList.add('hidden');
  $('fileName').textContent = file.name;
  $('fileInfo').textContent = fmtBytes(file.size);
  $('playerName').textContent = file.name;
  $('playerBar').classList.remove('hidden');
  document.body.classList.add('hasPlayer');
  const player = $('player');
  if (player.src) URL.revokeObjectURL(player.src);
  player.src = URL.createObjectURL(file);
  player.onloadedmetadata = () => {
    if (Number.isFinite(player.duration)) {
      $('fileInfo').textContent = `${fmtClock(player.duration)} · ${fmtBytes(file.size)}`;
    }
  };
  $('go').disabled = false;
}

const drop = $('drop');
drop.onclick = () => $('file').click();
drop.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('file').click(); } };
$('file').onchange = (e) => acceptFile(e.target.files[0]);
['dragenter', 'dragover'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => acceptFile(e.dataTransfer.files[0]));

$('clearBtn').onclick = () => {
  state.file = null;
  state.result = null;
  state.audio = null;
  $('file').value = '';
  $('player').removeAttribute('src');
  $('playerBar').classList.add('hidden');
  document.body.classList.remove('hasPlayer');
  $('fileCard').classList.add('hidden');
  $('result').classList.add('hidden');
  $('go').disabled = true;
};

$('cancelBtn').onclick = () => {
  if (state.abort) state.abort.abort();
  $('cancelBtn').disabled = true;
  setStatus('Cancelling…');
};

$('go').onclick = transcribe;
$('copyBtn').onclick = async () => {
  if (!state.result) return;
  const text = exportText(state.result);
  // navigator.clipboard needs a secure context: over plain http on a server it is undefined.
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
    else throw new Error('insecure context');
  } catch {
    const scratch = document.createElement('textarea');
    scratch.value = text;
    scratch.style.cssText = 'position:fixed;top:-1000px;opacity:0';
    document.body.append(scratch);
    scratch.select();
    try { document.execCommand('copy'); } catch { /* nothing else to try */ }
    scratch.remove();
  }
  $('copyBtn').textContent = 'Copied';
  setTimeout(() => { $('copyBtn').textContent = 'Copy'; }, 1400);
};
const downloadMenu = $('downloadMenu');
const closeMenu = () => {
  downloadMenu.classList.add('hidden');
  $('downloadBtn').setAttribute('aria-expanded', 'false');
};
$('downloadBtn').onclick = (e) => {
  e.stopPropagation();
  const open = downloadMenu.classList.toggle('hidden');
  $('downloadBtn').setAttribute('aria-expanded', String(!open));
};
document.addEventListener('click', () => { closeMenu(); closePicker(); });
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  closeMenu();
  closePicker();
  if (document.activeElement?.classList.contains('paraBody')) document.activeElement.blur();
});
document.querySelectorAll('[data-dl]').forEach((b) => {
  b.onclick = () => { download(b.dataset.dl); closeMenu(); };
});
document.querySelectorAll('.tab').forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
    const segs = tab.dataset.view === 'segments';
    $('viewSegments').classList.toggle('hidden', !segs);
    $('viewText').classList.toggle('hidden', segs);
  };
});

fetch('/api/health').then((r) => r.json()).then((h) => {
  state.textModel = h.textModel;
  if (h.chunkSeconds) CHUNK_SECONDS = h.chunkSeconds;
  if (h.diarizeChunkSeconds) DIARIZE_CHUNK_SECONDS = h.diarizeChunkSeconds;
  if (!h.hasKey) {
    $('keyWarning').innerHTML =
      'No OpenAI API key found. Create a <code>.env</code> file next to <code>server.js</code> containing ' +
      '<code>OPENAI_API_KEY=sk-…</code> and restart the server.';
    $('keyWarning').classList.remove('hidden');
  }
}).catch(() => {});
