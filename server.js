#!/usr/bin/env node
/**
 * Audio -> Transcript
 * Tiny zero-dependency Node server that serves the UI and proxies audio
 * to the OpenAI transcription API (the API key never reaches the browser).
 */
'use strict';

const http = require('node:http');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { structureTranscript } = require('./structure.js');

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const PORT = Number(process.env.PORT) || 5178;
// Loopback by default so a laptop install is not exposed. Hosts like Render need 0.0.0.0.
const ON_HOST = Boolean(process.env.RENDER || process.env.PORT && process.env.NODE_ENV === 'production');
const HOST = process.env.HOST || (ON_HOST ? '0.0.0.0' : '127.0.0.1');
const APP_PASSWORD = process.env.APP_PASSWORD || '';

// Chunk sizes are tunable because hosting proxies cut long requests (Render's is ~100 s).
// Two minutes, measured: on a 22-minute bilingual consultation with a known script,
// 11-minute parts captured 53 of 61 scripted utterances and 2-minute parts captured 58.
// Long parts quietly drop short interjections ("from what?", "have what?").
const CHUNK_SECONDS = Number(process.env.CHUNK_SECONDS) || 120;
const DIARIZE_CHUNK_SECONDS = Number(process.env.DIARIZE_CHUNK_SECONDS) || 180;
const OPENAI_URL = 'https://api.openai.com/v1/audio/transcriptions';
const MAX_UPLOAD = 26 * 1024 * 1024;        // OpenAI caps a single request at 25 MB
const MAX_JSON = 12 * 1024 * 1024;
const UPSTREAM_TIMEOUT = 8 * 60 * 1000;
const TEXT_MODEL = process.env.OPENAI_TEXT_MODEL || 'gpt-5.4-mini';

// Measured on a bilingual recording (see README): without this instruction gpt-4o-transcribe
// drops the non-dominant language on roughly a third of runs; with it, every run was complete.
// whisper-1 must NOT get it — an English instruction makes whisper transcribe only English —
// and the diarizing model rejects `prompt` outright.
const VERBATIM_PROMPT =
  'This recording may switch between languages, sometimes sentence by sentence. Transcribe ' +
  'every utterance verbatim, each one in the language it was actually spoken. Never translate, ' +
  'never summarise, and never skip a sentence.';

// --- minimal .env loader (no dotenv dependency) -----------------------------
function loadEnvFile() {
  const file = path.join(ROOT, '.env');
  if (!fs.existsSync(file)) return;
  for (const raw of fs.readFileSync(file, 'utf8').split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (!(key in process.env)) process.env[key] = value;
  }
}
loadEnvFile();

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.json': 'application/json; charset=utf-8',
};

const AUDIO_MIME = {
  mp3: 'audio/mpeg', mpeg: 'audio/mpeg', mpga: 'audio/mpeg',
  m4a: 'audio/mp4', mp4: 'audio/mp4', aac: 'audio/aac',
  wav: 'audio/wav', webm: 'audio/webm', ogg: 'audio/ogg',
  oga: 'audio/ogg', opus: 'audio/ogg', flac: 'audio/flac',
  mov: 'video/quicktime', avi: 'video/x-msvideo',
};

/** Shared-password gate. Constant-time compare so the password cannot be probed by timing. */
function authorized(req) {
  if (!APP_PASSWORD) return true;
  const header = req.headers.authorization || '';
  if (!header.startsWith('Basic ')) return false;
  const decoded = Buffer.from(header.slice(6), 'base64').toString('utf8');
  const supplied = Buffer.from(decoded.slice(decoded.indexOf(':') + 1));
  const expected = Buffer.from(APP_PASSWORD);
  return supplied.length === expected.length && crypto.timingSafeEqual(supplied, expected);
}

function json(res, code, body) {
  const payload = JSON.stringify(body);
  res.writeHead(code, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(payload),
    'Cache-Control': 'no-store',
  });
  res.end(payload);
}

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) {
        reject(Object.assign(new Error('Upload too large'), { statusCode: 413 }));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function header(req, name) {
  const v = req.headers[name];
  if (!v) return '';
  try { return decodeURIComponent(v); } catch { return String(v); }
}

async function handleTranscribe(req, res) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return json(res, 500, {
      error: 'OPENAI_API_KEY is not set. Put it in a .env file next to server.js and restart.',
    });
  }

  let buf;
  try {
    buf = await readBody(req, MAX_UPLOAD);
  } catch (err) {
    return json(res, err.statusCode || 400, {
      error: err.statusCode === 413
        ? 'That chunk is over the 25 MB per-request limit.'
        : 'Could not read the upload.',
    });
  }
  if (!buf.length) return json(res, 400, { error: 'Empty upload.' });

  const filename = header(req, 'x-filename') || 'audio.wav';
  const model = header(req, 'x-model') || 'whisper-1';
  const language = header(req, 'x-language');
  const prompt = header(req, 'x-prompt');
  const ext = path.extname(filename).slice(1).toLowerCase();
  const type = AUDIO_MIME[ext] || 'application/octet-stream';


  // Only whisper-1 and the diarizing model return timestamps, in different shapes.
  const isDiarize = model.includes('diarize');
  const responseFormat = isDiarize ? 'diarized_json' : (model === 'whisper-1' ? 'verbose_json' : 'json');

  const form = new FormData();
  form.append('file', new File([buf], filename, { type }));
  form.append('model', model);
  form.append('response_format', responseFormat);
  if (language) form.append('language', language);
  if (!isDiarize) {
    const instructions = [];
    if (model !== 'whisper-1') instructions.push(VERBATIM_PROMPT);
    if (prompt) instructions.push(model === 'whisper-1' ? prompt : `Expect terms such as: ${prompt}.`);
    if (instructions.length) form.append('prompt', instructions.join(' '));
  }
  if (isDiarize) form.append('chunking_strategy', 'auto');
  // Per-token confidence: it localises exactly where the model was guessing.
  if (responseFormat === 'json') form.append('include[]', 'logprobs');
  if (responseFormat === 'verbose_json') form.append('timestamp_granularities[]', 'segment');

  // Gateway 5xx and rate limits are common on long audio; retry before giving up.
  const RETRY_ON = new Set([429, 500, 502, 503, 504]);
  let upstream;
  let raw = '';
  for (let attempt = 0; attempt < 3; attempt++) {
    if (attempt) await new Promise((r) => setTimeout(r, attempt * 3000));
    try {
      upstream = await fetch(OPENAI_URL, {
        method: 'POST',
        headers: { Authorization: `Bearer ${apiKey}` },
        body: form,
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT),
      });
    } catch (err) {
      const timedOut = err.name === 'TimeoutError' || err.name === 'AbortError';
      if (attempt === 2 || !timedOut) {
        console.warn(`[transcribe] ${model} ${filename}: ${err.name} ${err.message}`);
        return json(res, 504, {
          error: timedOut
            ? `${model} did not answer within ${Math.round(UPSTREAM_TIMEOUT / 60000)} minutes. ` +
              'It is the slowest model — try a shorter recording, or gpt-4o-transcribe.'
            : `Could not reach OpenAI: ${err.message}`,
        });
      }
      continue;
    }

    raw = await upstream.text();
    if (upstream.ok) break;
    console.warn(`[transcribe] ${model} ${filename} (${buf.length} bytes): HTTP ${upstream.status} ${raw.slice(0, 200)}`);
    if (!RETRY_ON.has(upstream.status) || attempt === 2) {
      let message = raw;
      try { message = JSON.parse(raw).error?.message || raw; } catch { /* keep raw */ }
      // A bare gateway error says nothing useful on its own.
      if (upstream.status >= 500) {
        message = `OpenAI returned ${upstream.status} (${message.trim().slice(0, 120)}) for ${model}. ` +
          'This usually means the request took too long upstream — a shorter recording, or ' +
          'gpt-4o-transcribe instead of the diarizing model, should go through.';
      }
      return json(res, upstream.status, { error: message });
    }
  }

  let data;
  try { data = JSON.parse(raw); } catch { data = { text: raw }; }
  return json(res, 200, {
    text: data.text || '',
    logprobs: Array.isArray(data.logprobs)
      ? data.logprobs.map((t) => ({ token: t.token, p: Math.exp(t.logprob) }))
      : [],
    duration: data.duration ?? null,
    language: data.language ?? null,
    segments: Array.isArray(data.segments)
      ? data.segments
          .filter((s) => typeof s.text === 'string' && s.text.trim())
          .map((s) => ({
            start: s.start,
            end: s.end,
            text: s.text.trim(),
            speaker: s.speaker || null,
          }))
      : [],
  });
}

async function handleStructure(req, res) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return json(res, 500, { error: 'OPENAI_API_KEY is not set.' });

  let payload;
  try {
    payload = JSON.parse((await readBody(req, MAX_JSON)).toString('utf8'));
  } catch (err) {
    return json(res, err.statusCode === 413 ? 413 : 400, {
      error: err.statusCode === 413 ? 'Transcript too large to structure.' : 'Invalid JSON body.',
    });
  }
  if (!Array.isArray(payload.units) || !payload.units.length) {
    return json(res, 400, { error: 'No transcript units to structure.' });
  }

  try {
    const result = await structureTranscript(payload.units, { apiKey, model: TEXT_MODEL });
    return json(res, 200, result);
  } catch (err) {
    return json(res, err.statusCode || 502, { error: err.message });
  }
}

function serveStatic(req, res) {
  const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const rel = urlPath === '/' ? 'index.html' : urlPath.replace(/^\/+/, '');
  const file = path.join(PUBLIC_DIR, rel);
  if (!file.startsWith(PUBLIC_DIR)) { res.writeHead(403); return res.end('Forbidden'); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const { pathname } = new URL(req.url, 'http://localhost');

  // The health check stays open so a platform probe is not answered with a 401.
  if (pathname !== '/api/health' && !authorized(req)) {
    res.writeHead(401, {
      'WWW-Authenticate': 'Basic realm="Audio to Transcript", charset="UTF-8"',
      'Content-Type': 'text/plain; charset=utf-8',
    });
    return res.end('Sign in to use this transcriber.');
  }
  if (pathname === '/api/health') {
    return json(res, 200, {
      hasKey: Boolean(process.env.OPENAI_API_KEY),
      maxUpload: MAX_UPLOAD,
      textModel: TEXT_MODEL,
      chunkSeconds: CHUNK_SECONDS,
      diarizeChunkSeconds: DIARIZE_CHUNK_SECONDS,
    });
  }
  if (pathname === '/api/transcribe') {
    if (req.method !== 'POST') return json(res, 405, { error: 'Use POST.' });
    return handleTranscribe(req, res).catch((err) => json(res, 500, { error: err.message }));
  }
  if (pathname === '/api/structure') {
    if (req.method !== 'POST') return json(res, 405, { error: 'Use POST.' });
    return handleStructure(req, res).catch((err) => json(res, 500, { error: err.message }));
  }
  if (req.method !== 'GET') return json(res, 405, { error: 'Use GET.' });
  return serveStatic(req, res);
});

// Fail closed: a public instance without a password would let anyone spend the API key.
if (HOST !== '127.0.0.1' && HOST !== 'localhost' && !APP_PASSWORD) {
  console.error('\n  Refusing to start: HOST is public but APP_PASSWORD is not set.');
  console.error('  Set APP_PASSWORD (any shared secret) and restart.\n');
  process.exit(1);
}

server.listen(PORT, HOST, () => {
  console.log(`\n  Audio → Transcript running at http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}`);
  if (HOST !== '127.0.0.1') console.log('  ✓ Password protection is on (APP_PASSWORD)');
  if (!process.env.OPENAI_API_KEY) {
    console.log('  ⚠  OPENAI_API_KEY is not set — add it to .env and restart.\n');
  } else {
    console.log('  ✓ OpenAI API key loaded\n');
  }
});
