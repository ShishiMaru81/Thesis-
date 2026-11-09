# Stage 9 plan amendment 2

- Date written: 2026-10-07T09:44+06:00
- Amends: plan.json sha256 5d987f5c27c72778a9347c34f96f83dc4a4ec15ea9a5ccad8842f8d4052d611a
  and plan_amendment_1.md sha256(LF) c4987b213b2f0ad32ebea7f136fcf3145e8729728bb2af85b4f528c2de2e21b2
  (both unchanged and still in force)

## Reason

The Gemini free tier allowed too few requests per day to finish the 576
requests, and GitHub Models has been retired. The API model is changed to
gpt-oss-120b served by Groq.

## Change: model and provider

gemini-3.8-flash is replaced by:

- provider: groq, through its OpenAI-compatible endpoint
  `https://api.groq.com/openai/v1/chat/completions`
- model: `openai/gpt-oss-120b`
- API key: read only from the environment variable `GROQ_API_KEY`; never
  printed, logged or saved

The openai and anthropic slots stay unfilled, as before.

## Disclosure: Gemini responses already collected

3 gemini-3.8-flash dry-run responses exist, all in the ZS arm:

- 08854472 (clarify, missing_area, en)
- a10ca4f0 (clarify, unknown_local_unit, en)
- 2c216188 (numeric_runtime, bn)

They stay in `results/stage9_frontier/raw/gemini-3.8-flash/ZS.jsonl`
unedited. They are excluded from every score, table, statistic and macro.
REPORT.md mentions only that they exist and why the model was changed. No
gpt-oss-120b response had been seen when this amendment was written.

## Decoding for gpt-oss-120b on Groq

- reasoning_effort: `low`. This is the lowest level Groq supports for
  gpt-oss-120b (low / medium / high; the default is medium; `none` and
  `default` are accepted only by Qwen models).
- include_reasoning: true. The reasoning text comes back in a separate
  `message.reasoning` field and is stored per row for the record. It is not
  scored; only `message.content` is scored, as with every other arm.
- temperature: 0, as registered in plan.json `decoding.requested_generic`.
  Groq's general reasoning guide recommends 0.5–0.7 to avoid repetition; the
  registered value is kept, and this is noted as a deviation from that advice.
  Any repetition loop would show up as a length-limit stop and count as
  truncated under the registered rule.
- seed: 42 (Groq: best effort, determinism not guaranteed). The
  `system_fingerprint` is stored per row.
- max_completion_tokens: 4096. On Groq this limit counts reasoning plus
  visible tokens, so a 512 cap on the visible answer alone cannot be set. 4096
  keeps reasoning from starving the answer, the same choice as for Gemini.
- logprobs: not requested. Groq's API reference says logprobs are not yet
  supported by any of its models. Stored as null with that reason.
- Recorded per row: reasoning_effort, temperature and seed sent,
  max_completion_tokens, finish_reason, prompt_tokens, completion_tokens,
  reasoning_tokens (`usage.completion_tokens_details.reasoning_tokens`, as
  returned, null if absent), model and system_fingerprint as returned, the
  time the response was received, latency and timestamp.

## Truncation and the 512 rule (amendment 1 applied to gpt-oss-120b)

- Primary measure, as registered: a response is truncated only if
  finish_reason is `length` (Groq's length-limit value). No re-tokenisation.
- Amendment 1's secondary 512 measure uses visible output tokens, defined
  for gpt-oss-120b as completion_tokens − reasoning_tokens, both as reported
  by the API. If reasoning_tokens is null, the visible count is null and the
  row is handled as amendment 1 says for a null count (not counted as over
  512; the number of such rows is reported).
- Reasoning token counts are reported descriptively per arm.

## Throttling (operational, not a design change)

Groq's free limits for gpt-oss-120b are 30 requests/min, 1,000 requests/day,
8,000 tokens/min and 200,000 tokens/day. The runner throttles by tokens per
minute and requests per minute. It stops cleanly before the daily token or
request limit is reached, prints the rows remaining, and resumes from the
cache on the next run.

## Unchanged

The rows (288), the arms (ZS, ZS+P), prompt P and the few-shot examples,
request construction, the imported scorer, every metric, tests F1–F6, Holm
families, the reference arms and everything in amendment 1 stay exactly as
registered.
