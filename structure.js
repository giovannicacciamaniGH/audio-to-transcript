'use strict';
/**
 * Turns a flat transcript into paragraphs.
 *
 * A model labels the language of each unit (and any topic/turn change); the paragraph
 * cuts themselves are then made deterministically here, so a change of language ALWAYS
 * starts a new paragraph — which is what bilingual and interpreted recordings need.
 */

const CHAT_URL = 'https://api.openai.com/v1/chat/completions';
const LABEL_BATCH = 120;      // units per label call
const MAX_UNITS_PER_PARAGRAPH = 12;
const MAX_CHARS_PER_PARAGRAPH = 900;
const CONCURRENCY = 4;

const LABEL_SCHEMA = {
  name: 'labels',
  strict: true,
  schema: {
    type: 'object', additionalProperties: false, required: ['units'],
    properties: {
      units: {
        type: 'array',
        items: {
          type: 'object', additionalProperties: false,
          required: ['i', 'lang', 'new_paragraph', 'mixed'],
          properties: {
            i: { type: 'integer', description: 'the index given for this unit' },
            lang: { type: 'string', description: 'ISO 639-1 code of the language spoken in this unit' },
            new_paragraph: { type: 'boolean', description: 'true if a new paragraph should start here because the topic or the speaker turn changes' },
            mixed: { type: 'boolean', description: 'true if this one unit contains speech in more than one language' },
          },
        },
      },
    },
  },
};

const LABEL_SYS =
  'You annotate numbered units of a speech transcript.\n' +
  'For EVERY unit you are given, and in the same order, return: its index i, the ISO 639-1 code of the ' +
  'language actually spoken in it, and new_paragraph.\n' +
  'Judge the language of each unit on its own words only. Transcripts of bilingual meetings switch ' +
  'language often, sometimes on every sentence — never assume neighbouring units share a language.\n' +
  'Set new_paragraph=true when this unit starts a new topic or a new speaker turn. Do not set it merely ' +
  'because the language changes; that is handled separately.\n' +
  'Set mixed=true when a single unit itself contains speech in more than one language (for example an ' +
  'English sentence immediately followed by its Italian version inside the same unit). In that case put ' +
  'the language it starts in as lang. Never omit or duplicate an index.';

const SPLIT_SCHEMA = {
  name: 'splits',
  strict: true,
  schema: {
    type: 'object', additionalProperties: false, required: ['splits'],
    properties: {
      splits: {
        type: 'array',
        items: {
          type: 'object', additionalProperties: false,
          required: ['i', 'parts'],
          properties: {
            i: { type: 'integer', description: 'the index of the unit being split' },
            parts: {
              type: 'array',
              items: {
                type: 'object', additionalProperties: false,
                required: ['text', 'lang'],
                properties: {
                  text: { type: 'string', description: 'this part of the unit, copied verbatim' },
                  lang: { type: 'string', description: 'ISO 639-1 code of the language of this part' },
                },
              },
            },
          },
        },
      },
    },
  },
};

const SPLIT_SYS =
  'Each unit you are given contains speech in more than one language.\n' +
  'Split every unit into consecutive parts, so that each part is entirely in ONE language.\n' +
  'Copy the words exactly as they are given, in the same order. Never translate, never correct, never ' +
  'add or drop a single word or character: concatenating your parts back together must reproduce the ' +
  'original unit exactly. Split only where the language actually changes.';

async function chat({ apiKey, model, system, user, schema }) {
  const payload = {
    model,
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    response_format: { type: 'json_schema', json_schema: schema },
  };
  if (/^(gpt-5|o[34])/.test(model)) payload.reasoning_effort = 'low';

  const res = await fetch(CHAT_URL, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const raw = await res.text();
  if (!res.ok) {
    let message = raw;
    try { message = JSON.parse(raw).error?.message || raw; } catch { /* keep raw */ }
    throw Object.assign(new Error(message), { statusCode: res.status });
  }
  const body = JSON.parse(raw);
  return {
    data: JSON.parse(body.choices[0].message.content),
    tokens: body.usage?.total_tokens || 0,
  };
}

/** Run tasks with a small concurrency cap, preserving result order. */
async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  });
  await Promise.all(workers);
  return out;
}

async function labelUnits(units, ctx) {
  const batches = [];
  for (let i = 0; i < units.length; i += LABEL_BATCH) batches.push(units.slice(i, i + LABEL_BATCH));

  const results = await mapLimit(batches, CONCURRENCY, (batch) =>
    chat({
      ...ctx,
      system: LABEL_SYS,
      user: batch.map((u) => `[${u.i}] ${u.text}`).join('\n'),
      schema: LABEL_SCHEMA,
    }));

  const labels = new Map();
  let tokens = 0;
  for (const r of results) {
    tokens += r.tokens;
    for (const u of r.data.units || []) {
      if (Number.isInteger(u.i)) {
        labels.set(u.i, {
          lang: String(u.lang || 'und').slice(0, 8).toLowerCase(),
          brk: Boolean(u.new_paragraph),
          mixed: Boolean(u.mixed),
        });
      }
    }
  }
  return { labels, tokens };
}

/** Compare two strings ignoring case, accents and punctuation. */
const normalise = (text) => text
  .toLowerCase()
  .normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '')
  .replace(/[^a-z0-9]/g, '');

/**
 * A paragraph break can only happen between units, so a unit that itself switches
 * language would hide the switch. Those units are split into language-pure pieces —
 * and the split is only accepted if it reproduces the original text word for word.
 */
async function splitMixedUnits(units, labels, ctx) {
  const mixed = units.filter((u) => labels.get(u.i)?.mixed);
  if (!mixed.length) return { splits: new Map(), tokens: 0 };

  const batches = [];
  for (let i = 0; i < mixed.length; i += 20) batches.push(mixed.slice(i, i + 20));

  const results = await mapLimit(batches, CONCURRENCY, (batch) =>
    chat({
      ...ctx,
      system: SPLIT_SYS,
      user: batch.map((u) => `[${u.i}] ${u.text}`).join('\n'),
      schema: SPLIT_SCHEMA,
    }));

  const byIndex = new Map(units.map((u) => [u.i, u.text]));
  const splits = new Map();
  let tokens = 0;
  for (const r of results) {
    tokens += r.tokens;
    for (const split of r.data.splits || []) {
      const original = byIndex.get(split.i);
      const parts = (split.parts || []).filter((p) => p.text && p.text.trim());
      if (!original || parts.length < 2) continue;
      // Reject anything that paraphrased, translated or dropped text.
      if (normalise(parts.map((p) => p.text).join('')) !== normalise(original)) continue;
      splits.set(split.i, parts.map((p) => ({
        text: p.text.trim(),
        lang: String(p.lang || 'und').slice(0, 8).toLowerCase(),
      })));
    }
  }
  return { splits, tokens };
}

/** The unit list the paragraphs are built from, after mixed units have been split. */
function expandUnits(units, labels, splits) {
  const out = [];
  for (const unit of units) {
    const label = labels.get(unit.i);
    const pieces = splits.get(unit.i);
    if (pieces) {
      pieces.forEach((piece, n) => out.push({
        i: out.length,
        source: unit.i,
        text: piece.text,
        lang: piece.lang,
        brk: n === 0 ? Boolean(label?.brk) : false,
      }));
    } else {
      out.push({
        i: out.length,
        source: unit.i,
        text: unit.text,
        lang: label?.lang || 'und',
        brk: Boolean(label?.brk),
      });
    }
  }
  return out;
}

function buildParagraphs(units) {
  const paragraphs = [];
  for (const unit of units) {
    const current = paragraphs[paragraphs.length - 1];
    const tooLong = current &&
      (current.units.length >= MAX_UNITS_PER_PARAGRAPH || current.chars >= MAX_CHARS_PER_PARAGRAPH);
    // A change of language always starts a new paragraph.
    if (!current || current.lang !== unit.lang || unit.brk || tooLong) {
      paragraphs.push({ lang: unit.lang, units: [unit.i], chars: unit.text.length });
    } else {
      current.units.push(unit.i);
      current.chars += unit.text.length + 1;
    }
  }
  return paragraphs;
}

/**
 * units: [{ i, text }] in spoken order.
 * Returns { units, paragraphs, tokens, model } — `units` is the list the paragraphs index
 * into, which may be longer than the input when mixed-language units had to be split.
 */
async function structureTranscript(units, { apiKey, model }) {
  const clean = units
    .filter((u) => u && typeof u.text === 'string' && u.text.trim())
    .map((u, n) => ({ i: Number.isInteger(u.i) ? u.i : n, text: u.text.trim() }));
  if (!clean.length) return { units: [], paragraphs: [], tokens: 0, model };

  const ctx = { apiKey, model };
  const { labels, tokens: labelTokens } = await labelUnits(clean, ctx);
  const { splits, tokens: splitTokens } = await splitMixedUnits(clean, labels, ctx);
  const expanded = expandUnits(clean, labels, splits);
  const paragraphs = buildParagraphs(expanded);

  return {
    model,
    tokens: labelTokens + splitTokens,
    units: expanded.map((u) => ({ i: u.i, source: u.source, text: u.text, lang: u.lang })),
    paragraphs: paragraphs.map((p, index) => ({ index, lang: p.lang, units: p.units })),
  };
}

module.exports = { structureTranscript };
