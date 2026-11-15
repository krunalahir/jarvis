# JARVIS architecture

Current as of November 16, 2026. Reflects the code on `main`.

## Shape

Single-user Python backend (FastAPI), local-first:

```
user / chat client
      |
      v
FastAPI HTTP API (api.py) --------------- X-API-Key required on every route
      except /health when JARVIS_API_KEY is set; the app refuses to start
      with actions enabled and no key (unless JARVIS_ALLOW_INSECURE=true)
      |
      +--> Agent loop (agent/graph.py, agent/tools.py)
      |      LangChain create_agent over LangGraph. The model plans,
      |      calls one tool, reads the result, decides the next step.
      |      Bounded by JARVIS_AGENT_MAX_STEPS; hitting the bound
      |      returns stopped_step_limit instead of running away.
      |      Read tools run directly (web search, research, Gmail read,
      |      calendar list, connector status). Write tools NEVER execute:
      |      they queue a PendingAction through the approval gate.
      |
      +--> Research pipeline (research/pipeline.py, providers.py)
      |      Question -> Exa web search -> local model synthesizes an
      |      answer whose claims cite those sources. Unsupported claims
      |      are surfaced, not dropped. Answers persist in the store.
      |      Long questions run as background jobs (research/jobs.py);
      |      interrupted jobs requeue on startup (no external effects).
      |
      +--> Content + scheduling (content.py, scheduler.py)
      |      Topic -> post draft via the model (never publishes).
      |      Scheduling a draft is the approval for that post; due drafts
      |      are claimed atomically, then published through the connector.
      |
      +--> Approval gate (actions.py)
      |      Every send/post/call/play/publish is a PendingAction that
      |      executes only after POST /actions/{id}/approve. The approve
      |      call claims the action atomically, so concurrent approvals
      |      cannot execute twice.
      |
      +--> Connectors (connectors/)
      |      Gmail, Google Calendar, YouTube (Google OAuth token),
      |      Twilio calling, Instagram (Meta), X, Spotify. Each is inert
      |      until its credentials exist AND JARVIS_ACTIONS_ENABLED=true.
      |      Snapchat is a scaffold that fails loudly (partner API only).
      |      YouTube draft publishing downloads the draft's media URL;
      |      that fetch is hardened: http(s) only, host must resolve to a
      |      public IP (SSRF guard), redirects re-validated, 200 MB cap, optional host allowlist (JARVIS_MEDIA_URL_ALLOWLIST) for fail-closed deployments.
      |
      +--> Store (research/store.py)
             SQLite: answers, drafts, jobs, actions. One store-wide lock,
             WAL mode, busy timeout. Atomic transitions for draft claims
             and action approvals.
      |
      +--> Crash recovery (recovery.py)
             On startup, anything stuck in 'executing' or 'publishing' is
             marked failed-unknown. Never retried: the external effect may
             already have happened; a human checks the platform first.
```

## The model

Default: self-hosted Qwen3 8B over an OpenAI-compatible endpoint (LM
Studio, Ollama, mlx-lm or vLLM), configured with three env vars:
JARVIS_LOCAL_LLM_BASE_URL, JARVIS_LOCAL_LLM_MODEL,
JARVIS_LOCAL_LLM_API_KEY. Any OpenAI-compatible host works with the same
settings (llm.py builds a plain ChatOpenAI client). Hosted OpenAI is a
separate explicit opt-in (JARVIS_LLM_PROVIDER=openai + OPENAI_API_KEY).
There is no automatic paid fallback.

## Safety model (three gates, in order)

1. JARVIS_ACTIONS_ENABLED - capability switch, off by default. Not consent.
2. Per-platform credentials - an unconfigured connector refuses to act.
3. Per-action approval - every real-world effect is queued and approved
   individually. Approvals and draft claims are atomic; crash recovery
   flags unknown outcomes instead of retrying.

The app refuses to start with actions enabled and no API key unless
JARVIS_ALLOW_INSECURE=true (local demos only).

## Testing

113 offline tests, all passing (docs/testing.md). Everything external is
mocked; they verify code behavior, not live inference or live platform
calls.

## Known gaps (honest list)

- Real Qwen3 inference not yet exercised: tests use a scripted fake
  model. First live run needs the local model server.
- Live connector calls untested end to end: need real credentials and a
  controlled live pass, one platform at a time.
- Exa web search needs EXA_API_KEY.
- YouTube upload uses the single-request multipart protocol; large files
  should move to resumable upload.
- Voice: Twilio calls speak a fixed text and hang up. A real voice-agent
  bridge (STT -> agent -> TTS webhook) is a later milestone.
- Snapchat posting is partner-gated; the connector is a scaffold.
- Single-user, single-store: no multi-tenant isolation.
- Spotify: new apps are limited to 5 allowlisted users until extended
  quota is granted.
