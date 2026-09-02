'use strict';
/**
 * Model B — a local Whisper-WebUI instance (github.com/jhj0517/Whisper-WebUI).
 *
 * Its REST backend is task-based: POST the audio, get an identifier, poll until the task
 * completes. Everything here normalises that into the same shape the browser already gets
 * from the OpenAI path, so the front end does not care which engine produced a transcript.
 */

const POLL_INTERVAL = 2000;
const POLL_TIMEOUT = 30 * 60 * 1000;   // local CPU transcription can be slow
const PROBE_TIMEOUT = 1500;

const baseUrl = () =>
  (process.env.WHISPER_WEBUI_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

/** Is a Whisper-WebUI backend answering? Used to tell the UI whether Model B is available. */
async function probe() {
  try {
    const res = await fetch(`${baseUrl()}/docs`, {
      method: 'GET',
      signal: AbortSignal.timeout(PROBE_TIMEOUT),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** faster-whisper reports an average log-probability per segment; 0..1 is friendlier. */
function segmentConfidence(segment) {
  if (typeof segment.avg_logprob === 'number') return Math.min(1, Math.exp(segment.avg_logprob));
  if (Array.isArray(segment.words) && segment.words.length) {
    return Math.min(...segment.words.map((w) => (typeof w.probability === 'number' ? w.probability : 1)));
  }
  return null;
}

function normalise(segments) {
  const clean = (segments || [])
    .filter((s) => s && typeof s.text === 'string' && s.text.trim())
    .map((s) => ({
      start: typeof s.start === 'number' ? s.start : 0,
      end: typeof s.end === 'number' ? s.end : 0,
      text: s.text.trim(),
      speaker: s.speaker ? String(s.speaker).replace(/^SPEAKER[_\s]?/i, '') : null,
      confidence: segmentConfidence(s),
    }));

  return {
    text: clean.map((s) => s.text).join(' '),
    duration: clean.length ? clean[clean.length - 1].end : null,
    language: null,
    segments: clean,
    logprobs: [],
  };
}

/**
 * Transcribe one buffer. `language` is a two-letter code or '' for auto,
 * `diarize` asks Whisper-WebUI for speaker labels (needs HF_TOKEN configured there).
 */
async function transcribe(buffer, { filename, type, language, diarize }) {
  const base = baseUrl();

  const query = new URLSearchParams();
  if (language) query.set('lang', language);
  if (diarize) query.set('is_diarize', 'true');

  const form = new FormData();
  form.append('file', new File([buffer], filename, { type }));

  const queued = await fetch(`${base}/transcription/?${query}`, { method: 'POST', body: form });
  const queuedBody = await queued.text();
  if (!queued.ok) {
    throw Object.assign(
      new Error(`Whisper-WebUI rejected the upload (${queued.status}): ${queuedBody.slice(0, 200)}`),
      { statusCode: 502 },
    );
  }

  let identifier;
  try {
    identifier = JSON.parse(queuedBody).identifier;
  } catch {
    throw Object.assign(new Error('Whisper-WebUI returned an unreadable queue response.'), { statusCode: 502 });
  }
  if (!identifier) {
    throw Object.assign(new Error('Whisper-WebUI did not return a task identifier.'), { statusCode: 502 });
  }

  const deadline = Date.now() + POLL_TIMEOUT;
  for (;;) {
    if (Date.now() > deadline) {
      throw Object.assign(new Error('Whisper-WebUI did not finish within 30 minutes.'), { statusCode: 504 });
    }
    await new Promise((r) => setTimeout(r, POLL_INTERVAL));

    const res = await fetch(`${base}/task/${identifier}`);
    if (!res.ok) {
      throw Object.assign(new Error(`Whisper-WebUI task lookup failed (${res.status}).`), { statusCode: 502 });
    }
    const task = await res.json();
    const status = String(task.status || '').toUpperCase();

    if (status === 'COMPLETED') return normalise(task.result);
    if (status === 'FAILED') {
      throw Object.assign(
        new Error(`Whisper-WebUI failed: ${task.error || 'no reason given'}`),
        { statusCode: 502 },
      );
    }
    // QUEUED or IN_PROGRESS — keep waiting.
  }
}

module.exports = { transcribe, probe, baseUrl };
