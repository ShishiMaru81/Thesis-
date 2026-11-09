#!/usr/bin/env python3
"""Stage 9 runner: query frontier API models on the locked rows, cache every response.

  python run_frontier.py --dry-run 5          # 5 rows per arm, first active model, then exit
  python run_frontier.py --confirm            # full run (resumes from cache)
  python run_frontier.py --confirm --batch-size 40   # at most 40 new API requests this run

Keys are read from the environment only and never printed or stored.
Failed rows go to failures.jsonl, are never cached, and are retried on the next run.
Active model: groq openai/gpt-oss-120b (plan_amendment_2.md). The google path is kept
only so the 3 excluded gemini-3.8-flash dry-run rows stay traceable to their code.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from common import (ARMS, ARM_SLUG, HERE, OUT, RAW, active_models, build_messages, canonical, load_config,
                    load_prompt_assets, load_test, run_rows, select_rows, sha256_bytes)

PLAN = HERE / "plan.json"
FROZEN = HERE / "FROZEN.txt"
FAILURES = OUT / "failures.jsonl"
KEY_ENV = {"groq": "GROQ_API_KEY", "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
MAX_ATTEMPTS = 10
BACKOFF_BASE_S = 4.0
BACKOFF_MAX_S = 120.0
DAY_S = 86400.0


class DailyLimit(Exception):
    pass


def verify_plan() -> dict:
    if not PLAN.exists():
        raise SystemExit("STOP: plan.json missing. Run make_plan.py first (Step 0).")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    want = plan["sha256"]
    plan_c = dict(plan, sha256="")
    got = sha256_bytes((json.dumps(plan_c, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if got != want:
        raise SystemExit("STOP: plan.json does not match its frozen sha256")
    frozen = FROZEN.read_text(encoding="utf-8")
    for name, sha in re.findall(r"^(plan_amendment_\d+\.md)\s+sha256\(LF\)=([0-9a-f]{64})", frozen, re.M):
        p = HERE / name
        if not p.exists() or sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n")) != sha:
            raise SystemExit(f"STOP: {name} missing or does not match its sha256 in FROZEN.txt")
    return plan


def model_dir(model_id: str) -> Path:
    return RAW / re.sub(r"[^A-Za-z0-9._-]", "_", model_id)


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def log_failure(rec: dict) -> None:
    FAILURES.parent.mkdir(parents=True, exist_ok=True)
    with FAILURES.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def load_cache(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["cache_key"]] = r
    return out


def append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# groq (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------
RATE_HEADERS = ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
                "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens", "retry-after")


def groq_payload(model_id: str, cfg: dict, msgs: dict) -> dict:
    g = cfg["groq"]
    messages = [{"role": "system", "content": msgs["system"]}] if msgs["system"] is not None else []
    messages += [{"role": t["role"], "content": t["text"]} for t in msgs["turns"]]
    body = {"model": model_id, "messages": messages, "max_completion_tokens": g["api_max_completion_tokens"],
            "reasoning_effort": g["reasoning_effort"], "include_reasoning": g["include_reasoning"], "stream": False}
    if g["send_temperature"]:
        body["temperature"] = cfg["temperature"]
    if g.get("seed") is not None:
        body["seed"] = g["seed"]
    return {"provider": "groq", "body": body}


def parse_duration_s(s: str | None) -> float | None:
    """Groq reset headers look like '7.66s', '2m59.56s', '1h2m3s' or '120ms'."""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    total, ok = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", s):
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
        ok = True
    return total if ok else None


class GroqError(Exception):
    def __init__(self, code: int | None, message: str, headers: dict, body):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.headers, self.body = code, message, headers, body
        self.status = None


class GroqCaller:
    def __init__(self, api_key: str, cfg: dict):
        import httpx
        self.httpx = httpx
        g = cfg["groq"]
        self.http = httpx.Client(base_url=g["base_url"], timeout=g["timeout_s"],
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        self.rate: dict = {}

    def call(self, payload: dict) -> dict:
        t0 = time.perf_counter()
        try:
            r = self.http.post("/chat/completions", json=payload["body"])
        except self.httpx.HTTPError as e:
            raise GroqError(None, f"{type(e).__name__}: {e}", {}, None) from None
        latency = time.perf_counter() - t0
        received_at = now_iso()
        self.rate = {k: r.headers.get(k) for k in RATE_HEADERS if r.headers.get(k) is not None}
        if r.status_code != 200:
            try:
                body = r.json()
                msg = (body.get("error") or {}).get("message") or json.dumps(body)[:1000]
            except ValueError:
                body, msg = r.text[:2000], r.text[:1000]
            raise GroqError(r.status_code, msg, dict(self.rate), body)
        d = r.json()
        ch = d["choices"][0] if d.get("choices") else {}
        m = ch.get("message") or {}
        u = d.get("usage") or {}
        reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        comp = u.get("completion_tokens")
        return {
            "response_text": m.get("content") or "",
            "reasoning_text": m.get("reasoning"),
            "finish_reason": ch.get("finish_reason"),
            "finish_reason_raw": ch.get("finish_reason"),
            "model_version": d.get("model"),
            "model_version_observed_at": received_at,
            "system_fingerprint": d.get("system_fingerprint"),
            "response_id": d.get("id"),
            "groq_request_id": (d.get("x_groq") or {}).get("id"),
            "input_tokens": u.get("prompt_tokens"),
            "output_tokens": comp - reasoning if comp is not None and reasoning is not None else None,
            "output_tokens_note": "visible = completion_tokens - reasoning_tokens, both API-reported",
            "completion_tokens": comp,
            "thinking_tokens": reasoning,
            "thinking_tokens_note": "usage.completion_tokens_details.reasoning_tokens as returned (null = absent)",
            "cached_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "total_tokens": u.get("total_tokens"),
            "latency_s": round(latency, 3),
            "queue_time_s": u.get("queue_time"),
            "logprobs": None,
            "logprobs_note": "not requested: Groq API reference says logprobs are not supported by any model",
            "rate_limit_headers": dict(self.rate),
        }

    def classify_error(self, e: Exception) -> tuple[str, float | None]:
        if not isinstance(e, GroqError):
            return "client", None
        if e.code is None:
            return "network", None
        if e.code == 429:
            blob = f"{e.message} {json.dumps(e.body, default=str)}"
            if re.search(r"per day|\(TPD\)|\(RPD\)", blob, re.I):
                return "daily_limit", None
            return "rate_limit", parse_duration_s(e.headers.get("retry-after"))
        if e.code >= 500:
            return "server", None
        return "client", None

    def describe_limit(self, e: Exception) -> str:
        return str(getattr(e, "message", e))[:1000]

    def tpm_hint(self) -> tuple[int | None, float | None]:
        rem = self.rate.get("x-ratelimit-remaining-tokens")
        return (int(float(rem)) if rem is not None else None), parse_duration_s(self.rate.get("x-ratelimit-reset-tokens"))


# ---------------------------------------------------------------------------
# google (excluded from results; kept for traceability)
# ---------------------------------------------------------------------------
def google_payload(model_id: str, cfg: dict, msgs: dict) -> dict:
    g = cfg["google"]
    gen = {"max_output_tokens": g["api_max_output_tokens"], "thinking_level": g["thinking_level"]}
    if g["send_temperature"]:
        gen["temperature"] = cfg["temperature"]
    if g["request_logprobs"]:
        gen["response_logprobs"] = True
    return {"provider": "google", "model": model_id, "system_instruction": msgs["system"],
            "contents": [{"role": "model" if t["role"] == "assistant" else "user", "text": t["text"]} for t in msgs["turns"]],
            "generation_config": gen}


class GoogleCaller:
    def __init__(self, api_key: str, cfg: dict):
        from google import genai
        from google.genai import errors, types
        self.client = genai.Client(api_key=api_key)
        self.types, self.errors = types, errors

    def call(self, payload: dict) -> dict:
        t = self.types
        gc = payload["generation_config"]
        config = t.GenerateContentConfig(
            system_instruction=payload["system_instruction"],
            max_output_tokens=gc["max_output_tokens"],
            thinking_config=t.ThinkingConfig(thinking_level=gc["thinking_level"]),
            **({"temperature": gc["temperature"]} if "temperature" in gc else {}),
            **({"response_logprobs": True} if gc.get("response_logprobs") else {}),
        )
        contents = [t.Content(role=c["role"], parts=[t.Part(text=c["text"])]) for c in payload["contents"]]
        t0 = time.perf_counter()
        resp = self.client.models.generate_content(model=payload["model"], contents=contents, config=config)
        latency = time.perf_counter() - t0
        received_at = now_iso()

        cand = resp.candidates[0] if resp.candidates else None
        text_parts = []
        if cand is not None and cand.content is not None and cand.content.parts:
            text_parts = [p.text for p in cand.content.parts if p.text and not getattr(p, "thought", False)]
        fr = getattr(cand, "finish_reason", None) if cand is not None else None
        um = resp.usage_metadata
        pf = resp.prompt_feedback
        lp = getattr(cand, "logprobs_result", None) if cand is not None else None
        return {
            "response_text": "".join(text_parts),
            "finish_reason": getattr(fr, "name", None) if fr is not None else None,
            "finish_reason_raw": str(fr) if fr is not None else None,
            "prompt_block_reason": str(pf.block_reason) if pf is not None and pf.block_reason else None,
            "model_version": resp.model_version,
            "model_version_observed_at": received_at,
            "response_id": resp.response_id,
            "input_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
            "thinking_tokens": getattr(um, "thoughts_token_count", None),
            "thinking_tokens_note": "thoughts_token_count as returned by the API (null = field empty)",
            "total_tokens": getattr(um, "total_token_count", None),
            "latency_s": round(latency, 3),
            "logprobs": lp.model_dump(mode="json") if lp is not None else None,
            "logprobs_note": None if lp is not None else "not requested: gemini-3+ rejects response_logprobs",
        }

    def quota_info(self, e: Exception) -> list[dict]:
        found = []

        def walk(o):
            if isinstance(o, dict):
                if "quotaId" in o:
                    found.append({k: o.get(k) for k in ("quotaMetric", "quotaId", "quotaValue")})
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(getattr(e, "details", None))
        return found

    def describe_limit(self, e: Exception) -> str:
        q = self.quota_info(e)
        return ("; ".join(f"{x['quotaId']} (limit {x['quotaValue'] or '?'})" for x in q)
                or str(getattr(e, "message", e))[:1000])

    def classify_error(self, e: Exception) -> tuple[str, float | None]:
        """-> (kind, retry_after_s). kind in daily_limit / rate_limit / server / client / network."""
        if isinstance(e, self.errors.APIError):
            code = getattr(e, "code", None)
            blob = json.dumps(getattr(e, "details", None) or {}, default=str) + " " + str(getattr(e, "message", ""))
            if code == 429:
                if re.search(r"PerDay|per day|RequestsPerDay|daily", blob, re.I):
                    return "daily_limit", None
                m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', blob)
                return "rate_limit", float(m.group(1)) if m else None
            if code is not None and code >= 500:
                return "server", None
            return "client", None
        return "network", None

    def tpm_hint(self) -> tuple[int | None, float | None]:
        return None, None


PROVIDERS = {"groq": (groq_payload, GroqCaller), "google": (google_payload, GoogleCaller)}


def payload_settings(payload: dict) -> dict:
    if payload["provider"] == "groq":
        b = payload["body"]
        return {"reasoning_effort": b["reasoning_effort"], "include_reasoning": b["include_reasoning"],
                "api_max_completion_tokens": b["max_completion_tokens"],
                "temperature_sent": b.get("temperature"), "seed_sent": b.get("seed")}
    gc = payload["generation_config"]
    return {"thinking_level": gc["thinking_level"], "api_max_output_tokens": gc["max_output_tokens"],
            "temperature_sent": gc.get("temperature")}


# ---------------------------------------------------------------------------
# throttle: requests/min, tokens/min, rolling 24 h requests and tokens
# ---------------------------------------------------------------------------
def iso_epoch(s: str) -> float:
    return datetime.datetime.fromisoformat(s).timestamp()


class Throttle:
    def __init__(self, thr: dict, history: list[tuple[float, int, int]]):
        self.rpm = thr.get("requests_per_minute")
        self.rpd = thr.get("requests_per_day")
        self.tpm = thr.get("tokens_per_minute")
        self.tpd = thr.get("tokens_per_day")
        self.max_daily_wait = float(thr.get("max_wait_for_daily_budget_s") or 0)
        self.events = sorted(history)          # (epoch_s, requests, tokens)
        self.min_gap = 60.0 / self.rpm if self.rpm else 0.0
        self.last = 0.0

    def _window(self, seconds: float) -> list[tuple[float, int, int]]:
        cut = time.time() - seconds
        return [e for e in self.events if e[0] >= cut]

    def _free_at(self, est: int) -> float | None:
        """Epoch time at which the rolling-24 h token budget has room for est more tokens."""
        day = self._window(DAY_S)
        used = sum(e[2] for e in day)
        for ts, _, tok in day:
            used -= tok
            if used + est <= self.tpd:
                return ts + DAY_S
        return None

    def wait(self, est: int, hint: tuple[int | None, float | None] = (None, None)) -> None:
        while True:
            day = self._window(DAY_S)
            r24, t24 = sum(e[1] for e in day), sum(e[2] for e in day)
            if self.rpd is not None and r24 + 1 > self.rpd:
                raise DailyLimit(f"local guard: requests_per_day={self.rpd}, {r24} requests in the last 24 h")
            if self.tpd is None or t24 + est <= self.tpd:
                break
            free_at = self._free_at(est)
            when = datetime.datetime.fromtimestamp(free_at).astimezone().isoformat(timespec="minutes") if free_at else "unknown"
            if free_at is not None and free_at - time.time() <= self.max_daily_wait:
                print(f"    daily token budget full ({t24} in last 24 h); waiting until {when}", flush=True)
                time.sleep(max(1.0, free_at - time.time() + 1.0))
                continue
            raise DailyLimit(f"local guard: tokens_per_day={self.tpd}, {t24} tokens in the last 24 h, next request ~{est}; "
                             f"enough budget frees up after {when}")
        gap = time.monotonic() - self.last
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        if self.tpm:
            while True:
                minute = self._window(60.0)
                t60 = sum(e[2] for e in minute)
                if t60 == 0 or t60 + est <= self.tpm:
                    break
                time.sleep(max(0.5, minute[0][0] + 60.0 - time.time() + 0.5))
            remaining, reset = hint
            if remaining is not None and reset is not None and remaining < est:
                time.sleep(min(reset + 0.5, 60.0))
        self.last = time.monotonic()

    def record(self, tokens: int) -> None:
        self.events.append((time.time(), 1, tokens))


def history_for(caches: list[dict[str, dict]], model_id: str) -> list[tuple[float, int, int]]:
    hist = [(iso_epoch(r["timestamp"]), r.get("attempts") or 1, r.get("total_tokens") or 0)
            for c in caches for r in c.values()]
    if FAILURES.exists():
        for line in FAILURES.read_text(encoding="utf-8").splitlines():
            f = json.loads(line) if line.strip() else {}
            if f.get("model_id") == model_id and f.get("kind") not in ("skipped_provider", "daily_limit"):
                hist.append((iso_epoch(f["timestamp"]), f.get("attempts") or 1, 0))
    return hist


def backoff_wait(attempt: int, retry_after: float | None) -> float:
    """Exponential backoff with jitter: uniform(0.5, 1.0) x min(max, base * 2^(attempt-1)), never above BACKOFF_MAX_S."""
    ceiling = min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
    wait = ceiling * random.uniform(0.5, 1.0)
    return min(BACKOFF_MAX_S, max(wait, retry_after or 0.0))


def call_with_retry(caller, payload: dict, throttle: Throttle, est: int, max_attempts: int = MAX_ATTEMPTS) -> dict:
    for attempt in range(1, max_attempts + 1):
        throttle.wait(est, caller.tpm_hint())
        try:
            out = caller.call(payload)
        except Exception as e:  # noqa: BLE001
            throttle.record(0)
            kind, retry_after = caller.classify_error(e)
            if kind == "daily_limit":
                raise DailyLimit(caller.describe_limit(e)) from None
            if kind == "client" or attempt == max_attempts:
                e.attempts = attempt
                raise
            wait = backoff_wait(attempt, retry_after)
            msg = str(getattr(e, "message", e))[:160]
            print(f"    {kind} error {getattr(e, 'code', '')} {getattr(e, 'status', '') or ''} (attempt {attempt}/{max_attempts}): "
                  f"{msg}; retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        throttle.record(out.get("total_tokens") or 0)
        out["attempts"] = attempt
        return out
    raise RuntimeError("unreachable")


def dry_rows(sub: dict, n: int) -> list[dict]:
    first = lambda rows, pred: next(r for r in rows if pred(r))  # noqa: E731
    picks = [
        first(sub["clarify_68"], lambda r: r["subtype"] == "missing_area" and r["lang"] == "en"),
        first(sub["clarify_68"], lambda r: r["subtype"] == "unknown_local_unit"),
        sub["numeric_single_116"][0],
        sub["abstain_seed_34"][0],
        sub["ood_hard_40"][0],
    ]
    pool = [r for r in run_rows(sub) if r not in picks]
    return (picks + pool)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", type=int, metavar="N")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--model", help="model_id (default: first active in config)")
    ap.add_argument("--batch-size", type=int, metavar="N",
                    help="stop after N new (uncached) API requests per model; re-run to continue")
    args = ap.parse_args()
    if not args.dry_run and not args.confirm:
        raise SystemExit("Full run requires --confirm (or use --dry-run N).")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    run_started_at = now_iso()

    plan = verify_plan()
    cfg = load_config()
    rows = load_test()
    sub = select_rows(rows)
    p_text, examples = load_prompt_assets()
    active, skipped = active_models(cfg)
    for s in skipped:
        print(f"SKIP {s['provider']}: {s['reason']}")

    models = [m for m in active if not args.model or m["model_id"] == args.model]
    if args.dry_run:
        models = models[:1]
    todo_rows = dry_rows(sub, args.dry_run) if args.dry_run else run_rows(sub)
    plan_ids = set(plan["rows"]["clarify_68"] + plan["rows"]["numeric_146"] + plan["rows"]["abstain_seed_34"] + plan["rows"]["ood_hard_40"])
    assert all(r["row_id"] in plan_ids for r in todo_rows), "row not in frozen plan"

    summary = []
    for m in models:
        prov, mid = m["provider"], m["model_id"]
        if prov not in PROVIDERS:
            print(f"SKIP {prov}/{mid}: no caller for this provider")
            continue
        key = os.environ.get(KEY_ENV[prov])
        if not key:
            print(f"SKIP {prov}/{mid}: {KEY_ENV[prov]} not set")
            log_failure({"timestamp": now_iso(), "model_id": mid, "kind": "skipped_provider", "reason": f"{KEY_ENV[prov]} not set"})
            continue
        build, caller_cls = PROVIDERS[prov]
        caller = caller_cls(key, cfg)
        del key

        paths = {arm: model_dir(mid) / f"{ARM_SLUG[arm]}.jsonl" for arm in ARMS}
        caches = {arm: load_cache(paths[arm]) for arm in ARMS}
        thr_cfg = cfg["throttle"][prov]
        throttle = Throttle(thr_cfg, history_for(list(caches.values()), mid))
        default_est = int(thr_cfg.get("default_request_tokens") or 0)

        def estimate(arm: str) -> int:
            seen = [r.get("total_tokens") for r in caches[arm].values() if r.get("total_tokens")]
            return round(sum(seen) / len(seen)) if seen else default_est

        def key_of(arm: str, r: dict) -> tuple[dict, str, str]:
            payload = build(mid, cfg, build_messages(arm, r["question"], p_text, examples))
            req_sha = sha256_bytes(canonical(payload).encode("utf-8"))
            return payload, req_sha, f"{r['row_id']}|{req_sha}"

        stopped = None
        batch_full = False
        new_requests = 0
        n_failed = 0
        for arm in ARMS:
            done_rows = []
            for i, r in enumerate(todo_rows, 1):
                payload, req_sha, ck = key_of(arm, r)
                if ck in caches[arm]:
                    done_rows.append(caches[arm][ck])
                    continue
                if args.batch_size is not None and new_requests >= args.batch_size:
                    batch_full = True
                    break
                new_requests += 1
                try:
                    out = call_with_retry(caller, payload, throttle, estimate(arm))
                except DailyLimit as e:
                    stopped = str(e)
                    log_failure({"timestamp": now_iso(), "run_started_at": run_started_at, "model_id": mid, "arm": arm,
                                 "row_id": r["row_id"], "request_sha256": req_sha, "kind": "daily_limit",
                                 "message": stopped[:1000], "cached": False})
                    break
                except Exception as e:  # noqa: BLE001
                    kind, _ = caller.classify_error(e)
                    n_failed += 1
                    log_failure({"timestamp": now_iso(), "run_started_at": run_started_at, "model_id": mid, "arm": arm,
                                 "row_id": r["row_id"], "request_sha256": req_sha, "kind": kind,
                                 "attempts": getattr(e, "attempts", None), "error_type": type(e).__name__,
                                 "code": getattr(e, "code", None), "message": str(getattr(e, "message", e))[:500],
                                 "cached": False})
                    print(f"  FAIL {arm} {r['row_id'][:8]} {kind}: {type(e).__name__} {getattr(e, 'code', '')} "
                          f"{str(getattr(e, 'message', e))[:200]} (not cached; retried next run)", flush=True)
                    continue
                rec = {"cache_key": ck, "row_id": r["row_id"], "model_id": mid, "provider": prov, "arm": arm,
                       "request_sha256": req_sha, "run_mode": "dry_run" if args.dry_run else "full",
                       "run_started_at": run_started_at, **payload_settings(payload),
                       "plan_sha256": plan["sha256"], "timestamp": now_iso(), **out}
                append_cache(paths[arm], rec)
                caches[arm][ck] = rec
                done_rows.append(rec)
                if not args.dry_run and i % 10 == 0:
                    print(f"  {mid} {arm}: {i}/{len(todo_rows)}", flush=True)
            summary.append((mid, arm, done_rows))
            if stopped or batch_full:
                break

        remaining = sum(1 for arm in ARMS for r in todo_rows if key_of(arm, r)[2] not in caches[arm])
        print(f"\n{mid}: new requests this run: {new_requests}, failed (logged, not cached): {n_failed}")
        if stopped:
            print(f"DAILY LIMIT reached for {mid}: {stopped}\nRows remaining: {remaining}. Re-run later; the cache resumes.")
        elif batch_full:
            print(f"Batch size {args.batch_size} reached. Rows remaining: {remaining}. Re-run to continue.")
        else:
            print(f"Rows remaining after this run: {remaining}")

        if args.dry_run:
            report_dry(summary, todo_rows, cfg, len(run_rows(sub)))
            r0 = todo_rows[0]
            print("\n" + "=" * 100 + f"\nZS+P REQUEST PAYLOAD (row {r0['row_id'][:8]}; sent as JSON body, key only in the "
                  f"Authorization header and not shown)\n" + "=" * 100)
            print(json.dumps(key_of("ZS+P", r0)[0], ensure_ascii=False, indent=2))
    return 0


def report_dry(summary, todo_rows, cfg, n_full) -> None:
    qs = {r["row_id"]: r for r in todo_rows}
    import s6_eval
    tot = {"in": 0.0, "out": 0.0, "think": 0.0, "n": 0}
    per_arm = {}
    for mid, arm, recs in summary:
        print("\n" + "=" * 100 + f"\nDRY RUN — {mid} — arm {arm}\n" + "=" * 100)
        a = per_arm.setdefault((mid, arm), {"in": 0, "out": 0, "think": 0, "n": 0})
        for rec in recs:
            r = qs[rec["row_id"]]
            setting = (f"reasoning_effort={rec['reasoning_effort']}" if "reasoning_effort" in rec
                       else f"thinking_level={rec.get('thinking_level')}")
            print(f"\n[{r['qtype']}/{r['category']}/{r.get('subtype')}/{r['lang']}] row {rec['row_id'][:8]}")
            print(f"Q: {r['question']}")
            print(f"A: {rec['response_text']}")
            if rec.get("reasoning_text"):
                rt = rec["reasoning_text"]
                print(f"[reasoning, not scored, {len(rt)} chars]: {rt[:600]}{' …' if len(rt) > 600 else ''}")
            print(f"-> class={s6_eval.classify_response(rec['response_text'])}  finish_reason={rec['finish_reason']}  "
                  f"model={rec['model_version']} fp={rec.get('system_fingerprint')} @ {rec.get('model_version_observed_at', rec['timestamp'])}  "
                  f"{setting}  tokens in/visible/reasoning={rec['input_tokens']}/{rec['output_tokens']}/{rec['thinking_tokens']}  "
                  f"latency={rec['latency_s']}s  attempts={rec.get('attempts')}")
            for k, v in (("in", rec["input_tokens"]), ("out", rec["output_tokens"]), ("think", rec["thinking_tokens"])):
                a[k] += v or 0
            a["n"] += 1
    print("\n" + "=" * 100 + "\nFULL-RUN ESTIMATE (from dry-run token means; per model)\n" + "=" * 100)
    mid0 = None
    for (mid, arm), a in per_arm.items():
        if not a["n"]:
            continue
        mid0 = mid
        price = cfg["pricing_usd_per_mtok"].get(mid)
        ei, eo, et = (a[k] / a["n"] * n_full for k in ("in", "out", "think"))
        tot["in"] += ei; tot["out"] += eo; tot["think"] += et; tot["n"] += n_full
        cost = (ei * price["input"] + (eo + et) * price["output_incl_thinking"]) / 1e6 if price else float("nan")
        print(f"{mid} {arm}: {n_full} rows  input≈{ei:,.0f}  visible≈{eo:,.0f}  reasoning≈{et:,.0f}  paid-tier cost≈${cost:.3f}")
    if mid0 is None:
        return
    price = cfg["pricing_usd_per_mtok"][mid0]
    cost = (tot["in"] * price["input"] + (tot["out"] + tot["think"]) * price["output_incl_thinking"]) / 1e6
    all_tok = tot["in"] + tot["out"] + tot["think"]
    print(f"TOTAL: {tot['n']} requests  input≈{tot['in']:,.0f}  visible≈{tot['out']:,.0f}  reasoning≈{tot['think']:,.0f}  "
          f"all≈{all_tok:,.0f}  paid-tier≈${cost:.3f}  (free tier: $0)")
    tpd = (cfg["throttle"].get(next((m["provider"] for m in cfg["models"] if m["model_id"] == mid0), ""), {}) or {}).get("tokens_per_day")
    if tpd:
        print(f"Free-tier days needed at tokens_per_day={tpd:,}: ≈{all_tok / tpd:.1f}")


if __name__ == "__main__":
    sys.exit(main())
