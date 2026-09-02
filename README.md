# Audio → Transcript

Upload an audio (or video) file in the browser and get a full transcript back from the
OpenAI speech-to-text API, **split into paragraphs at every change of language** and
checked back against the audio so you can see nothing was dropped.

No dependencies to install — just Node 20+.

## Setup

1. Put your OpenAI API key in a `.env` file next to `server.js`:

   ```
   OPENAI_API_KEY=sk-...
   ```

2. `npm start`, then open http://localhost:5178

## Deploying on Render

The repo has a `render.yaml` blueprint: push it to GitHub, then in Render choose
**New → Blueprint** and pick the repo. Set two secrets in the dashboard (never in git):

| variable | value |
|---|---|
| `OPENAI_API_KEY` | your key — **rotate the one used in development first** |
| `APP_PASSWORD` | any shared secret; coworkers enter it as the browser password (username is ignored) |

Things that specifically matter on Render:

- **It refuses to start on a public address without `APP_PASSWORD`.** The URL is public and
  every transcription spends your key, so an unprotected instance fails closed rather than
  running. `/api/health` stays open so Render's health check isn't answered with a 401.
- **Render cuts a request at roughly 100 seconds.** The diarizing model needs about half the
  audio's duration, so the blueprint sets `DIARIZE_CHUNK_SECONDS=120` and
  `CHUNK_SECONDS=420` — the browser reads these from the server, so you can tune them in the
  dashboard without a redeploy if you see timeouts.
- **Use the Starter plan, not Free.** Free instances sleep after 15 minutes and take about a
  minute to wake, which looks like a hang on the first upload of the day.
- **Set a spending limit** on the OpenAI account. One key means one bill with no per-user
  attribution; a 20-minute recording costs roughly 10–15 cents.
- **Nothing is stored server-side.** Audio is held in memory for the request only, and
  transcripts live in the browser tab until downloaded. For recordings of patients, note
  that audio still leaves your network to reach OpenAI — a BAA with OpenAI and institutional
  sign-off are the prerequisites for using real consultations, not a code change.

## What it does

1. **Transcribes** the file. It is sent whole whenever it fits inside the API's 25 MB
   request limit — these models handle language switching far better with the entire
   recording in front of them. Oversized files are decoded in the browser, cut **on
   silence** (never mid-word), transcribed three parts at a time, and stitched back together.

   Part length depends on the model, because `gpt-4o-transcribe-diarize` needs about half
   the audio's own duration to process it: a 2-minute chunk took 69 s, and a 10-minute
   chunk exceeded OpenAI's gateway timeout and came back as a bare `500 Internal Server
   Error`. So diarized runs are cut into **3-minute** parts and everything else into
   10-minute parts. A 21-minute call now transcribes in about 4 minutes.

   **Parts are cut into equal lengths, never a target size plus a remainder.** Short
   requests lose content badly — measured on 152 seconds of unbroken speech carrying 30
   utterances: a 19-second trailing part returned 10 words where 60 were spoken (27/30
   overall). Cutting the same audio into three equal ~50-second parts returned **30/30**.
   The search for a silent cut point is also kept proportional to the part length, because
   a wide search drifts and recreates the short part it was meant to avoid.
2. **Paragraphs it by language.** Every unit of the transcript is language-labelled by
   `gpt-5.4-mini`, and a paragraph break is then made **deterministically** wherever the
   language changes — so a passage repeated or interpreted in another language simply
   becomes its own paragraph, tagged with that language. Topic and speaker turns break
   paragraphs too.

   A break can only fall *between* units, so a unit that itself switches language mid-way
   ("…welcome to the clinic buongiorno a tutti…") would hide the switch. Those units are
   detected and split at the language boundary, and the split is accepted only if the
   pieces reproduce the original text word for word — anything paraphrased or translated
   is rejected and the unit is left intact.

   **Speakers:** tick *Identify speakers*. The transcript still comes from
   `gpt-4o-transcribe`; `gpt-4o-transcribe-diarize` runs alongside it purely to report who
   is speaking when, and its own text is discarded.

   This split is not stylistic. On a 9-minute bilingual recording carrying 60 marked
   utterances, `gpt-4o-transcribe` returned **60/60** while the diarizing model returned
   **36/60** — losing 40%, consistently, in all three parts tested. Speaker turns are
   therefore mapped onto the complete transcript by position in the *speech* (silence
   excluded, using the loudness envelope), which also gives the transcript approximate
   timestamps. Measured on that recording: 60/60 utterances kept, speaker attribution 98%
   and 100% for the two voices. It doubles the cost and roughly doubles the time.

   **Across parts:** the API assigns speaker letters *per request*, so "Speaker A" in
   part 1 and part 5 need not be the same person — measured on a 4-part recording, the
   labels were permuted in 3 of the 4 parts. `public/voiceprint.js` therefore re-identifies
   voices from the audio itself: it takes a mel-band spectral fingerprint of every
   (part, label) pair and clusters them, with the constraint that two labels inside one
   request are different people by definition. That constraint also **calibrates the
   threshold**: the most similar pair *within* a request shows what "two different voices
   on this microphone" looks like, and anything above it is treated as one voice. Verified
   against ground truth on an 8-minute, 3-voice recording: 100% / 98% / 95% label
   consistency per voice.

   Speakers are identified **by voice, not by language** — the two are independent. Tested
   with each of three voices speaking both English and Italian: every voice kept the same
   label across both languages (Samantha → A in en *and* it; Alice → B in it *and* en;
   Fred → C in both). So one person switching language stays one speaker, and two people
   sharing a language stay two speakers.
3. **Checks itself against the audio.** The page computes a loudness envelope of the
   recording locally, so it knows how much of it actually carries sound — independently
   of anything the model claims. It reports words per minute of sound, and (when the
   model returns timestamps) flags any sounding stretch that produced no text.
4. **Flags the words it was unsure of.** For the `gpt-4o-*` models the API returns a
   probability per token (`include[]=logprobs`), so any paragraph containing a word below
   50% confidence gets a `⚠ check` chip listing those words, and the CSV/JSON carry a
   per-paragraph `confidence`. Measured: a clean sentence scores 0.999; the same sentence
   under heavy noise, where the model dropped a word, scores **0.148 at exactly the word it
   got wrong**.
5. **Cross-checks on demand.** One button re-transcribes with a second model and lists
   any passage that model heard which is missing from your transcript. This is the only
   real check on what a speech model silently left out.

### How much can be guaranteed?

Nothing here can prove a transcript is correct — no speech system can. What it can do is
make the doubtful parts visible, and each check has a different reach:

| check | catches | misses |
|---|---|---|
| loudness vs word rate | gross loss (a whole part failing) | a dropped sentence or two |
| token confidence | words the model itself doubted | confident mistakes |
| cross-check | passages one model heard and the other didn't | errors both models share |
| the pinned player | anything, if you listen | what you don't check |

The loudness check needed one fix of its own: it estimated the noise floor from the 20th
percentile, so on **continuous** speech — where there is no silence to measure — it called
most of the recording silent (46 s of 152 s). The threshold is now capped at a fraction of
the loud level, giving 152/152 s on continuous speech and exactly 60 detected regions for
60 spoken utterances on audio with gaps.

A measured example of the limits: on a 10-second clip where one of three sentences was
dropped, the word-rate check passed at 123 words per minute — well inside normal speech —
while the cross-check named the missing sentence exactly, with its timestamp. That is why
the rate check now reports "no large omission detected" rather than "nothing missing", and
points at the cross-check.

## Model choice

These numbers are from a 57-second English/Italian recording with a known ground truth of
**121 words**, three runs per configuration:

| Model | Words returned | Notes |
|---|---|---|
| `gpt-4o-transcribe` **+ verbatim prompt** | **121, 121, 121** | default — complete and stable |
| `gpt-4o-transcribe` (no prompt) | 121, 121, 92 | drops a language on some runs |
| `gpt-4o-transcribe-diarize` | 119, 111, 126 | adds timestamps and speaker labels |
| `whisper-1` | 100, 100, 100 | consistently drops one language's sentences |
| `whisper-1` + English prompt | 75, 75, 75 | prompt language biases it — all Italian lost |

So the server attaches a **verbatim instruction** ("this recording may switch between
languages… never translate, never skip a sentence") to the `gpt-4o-*` models, never to
`whisper-1` — where an English instruction makes it transcribe only English — and never to
the diarizing model, which rejects `prompt` outright.

Splitting bilingual audio into short pieces makes things **worse**, not better (8–12 second
blocks scored 79 with whisper-1 and 54 with gpt-4o-transcribe): a model given a short clip
commits to one language for it. Hence: send as much context per request as the API allows.

The optional **Hint** field is appended to that instruction — use it for names, jargon or
spellings ("radical prostatectomy, Cacciamani, USC Urology").

## Reading the transcript

Each paragraph carries two chips in the left column, coloured and shaped so the two
dimensions never read as one:

- **Language** — a solid tinted chip
- **Speaker** — an outlined chip

Colours are assigned in order of first appearance, from hue sets offset from each other so
a language and a speaker are never the same colour. A legend above the transcript keys
them, and the speaker chips appear in the Timestamps view too. Both palettes have light
and dark variants.

## Correcting a transcript

Nothing on the page is read-only:

- **Click a chip to change it.** A speaker chip offers the voices already found plus one
  spare label, so a misattributed passage can be moved; a language chip offers the
  languages in the transcript plus the full list from the form. Reassigning a speaker also
  updates the Timestamps view and every export.
- **Add a row where one is missing.** Hovering between paragraphs reveals a
  *+ add a paragraph here* strip; clicking it opens an empty row at that point, taking its
  language and speaker from the paragraph above so it usually needs no adjusting — and both
  chips remain clickable if it does. A `+ added` marker and a `×` to remove it sit in the
  left column. Useful when the model dropped a passage you can hear in the audio.
- **Tick *interpreter* on any passage** that is the interpreter speaking rather than the
  original speaker. The checkbox sits in the left column of every paragraph, and each voice
  in the legend has one too — since an interpreter usually interprets throughout, ticking it
  there marks all of that speaker's paragraphs at once (it shows a mixed state if you then
  untick individual ones). Marked passages get a rule down their left edge, and an
  `interpreter` tag in the Timestamps view.
- **Click the text to correct it.** Each paragraph is editable in place. A dashed
  `✎ edited` chip appears next to any paragraph changed by hand, and disappears again if
  you restore the original wording. The summary line counts how many paragraphs were edited.

Edits survive into the exports, and stay distinguishable from what the model produced:
`.txt` tags a paragraph `interpreter`, `edited` or `added`, `.csv` has dedicated
`interpreter` and `edited` columns, and `.json` has `interpreter` / `edited` / `added`
flags plus the model's original wording under `transcribedText`. An added row that is left
empty is skipped by the text exports.

**Speaker letters follow the order of speaking** — whoever talks first is Speaker A, the
next new voice is B, and so on. The API's own letters are per request and arbitrary, so
they are re-lettered by first appearance after the cross-part voice matching.

## Exports

The **Download** button next to the transcript offers:

- `.csv` — one row per paragraph: `paragraph, start, end, start_seconds, language, speaker, text`
  (UTF-8 with a BOM, so Excel opens accented text correctly)
- `.txt` — the paragraphs, each tagged with its time and language
- `.json` — paragraphs, segments, speakers and the self-check result
- `.srt` / `.vtt` — subtitles, listed only when the model returned timestamps

**Copy** puts the same text as the `.txt` export on the clipboard.

## If it looks stuck

Long recordings take real time, so the progress card shows a **running clock** and a
**Cancel** button — cancelling stops local work immediately, not just requests in flight.

Two things used to freeze the browser tab, and are now handled:

- **Decoding.** An hour of 48 kHz stereo decodes to well over a gigabyte. Files under
  ~15 minutes are decoded at their own sample rate (fast: 186 ms for a 10-minute file);
  longer ones are decoded through a 16 kHz mono context, which the browser resamples to
  as it decodes — a third of the memory (a 50-minute file: 191 MB, 1.4 s).
- **The loudness scan.** It now samples with a stride instead of reading every sample,
  and yields to the browser as it goes, so the page stays responsive.

Past two hours the local self-check is skipped altogether — use the cross-check button
instead, which does its comparison server-side.

## Notes

- Transient upstream failures (429, 500, 502, 503, 504) are retried twice with backoff
  before the error reaches you, and a 5xx is reported with what it usually means rather
  than passed through as a bare "Internal Server Error". Upstream failures are logged to
  the server console with the model, file and status.
- The API key stays on the server; the browser never sees it.
- Audio is held in memory only for the request — nothing is written to disk.
- `OPENAI_TEXT_MODEL` overrides the model used for language labelling (default `gpt-5.4-mini`).
- Video files are accepted; OpenAI extracts the audio track.
