# JARVIS

A personal AI assistant: research-first, then connected actions.

JARVIS answers questions with source-backed research, then acts through
connected apps: Google mail and calendar, phone calls, social content
creation with scheduled publishing, and Spotify playback.

The brain is a self-hosted [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B)
running on your own machine (free, Apache 2.0; no per-message cost), wired
through LangChain and a LangGraph agent loop: the model plans, calls a
tool, reads the result and decides the next step. Web search stays a
separate tool so fresh facts come from real sources, not model memory.

This repository is the base version. The research/Q&A core and the agent
loop run and are tested offline. App connectors are real API client code,
but each one stays inert until you supply that platform's credentials and
explicitly enable actions. Nothing here posts, sends, calls or plays
without that.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env   # fill in what you have; everything is optional to start
pytest                 # run the test suite (no credentials needed)
uvicorn jarvis.api:app --host 127.0.0.1 --reload
```

To make the model live, start a local Qwen3 8B server (LM Studio, Ollama,
mlx-lm or vLLM) and point `.env` at it - see
[docs/local-model.md](docs/local-model.md). Without it the API still runs;
model-backed features report clearly that the server is unreachable.

API docs: http://localhost:8000/docs

## The agent loop

`POST /agent/chat` runs the LangGraph agent (LangChain 1.0 `create_agent`):

1. **Plan** - the model reads your request and decides the next step.
2. **Act** - it calls one tool: web search, research, a Google sign-in
   link, a Gmail/Calendar read, a post draft, or an action request.
3. **Observe** - the tool result goes back into the conversation.
4. **Decide** - call another tool or answer.

The loop is bounded (`JARVIS_AGENT_MAX_STEPS`, default 8) and reports
`stopped_step_limit` instead of ever running away. Writes are never
executed by the agent: its `request_*` tools create the same pending
actions as the HTTP endpoints, and each one still needs your explicit
approval via `POST /actions/{id}/approve`.

## Connecting Google

Gmail, Calendar and YouTube share one Google sign-in, and it starts in
chat. There is no access token to paste.

1. Put `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` (from your own Google
   Cloud OAuth client) in `.env`, and register `GOOGLE_REDIRECT_URI` on
   that client.
2. Ask the agent to connect Google (`POST /agent/chat`). The reply's
   `google_sign_in_url` field holds the link to open.
3. Open it, choose your account and press Allow. Google sends your
   browser back to `/auth/google/callback`, which saves the sign-in in
   `GOOGLE_TOKEN_FILE`. The access token renews itself from then on.

The agent can hand you the link but cannot sign in for you. Keep the
token file private; it is listed in `.gitignore` for that reason.

## Feature map

| Feature | Status | What works today |
|---|---|---|
| Agent loop | Working, tested | LangGraph plan/tool/observe loop over the features below; bounded; writes only queue approvals. Needs the local model server running. |
| Research & Q&A | Working, tested | Local Qwen3 writes the answer, Exa fetches fresh web sources; claims are linked to sources, unsupported ones are surfaced. Without EXA_API_KEY it reports unconfigured rather than guessing. Hosted providers (OpenAI/Perplexity) remain explicit opt-ins. |
| Content creation | Working, tested | Topic -> post draft via the local model, per platform. Never publishes. |
| Gmail / Google Calendar | Coded, needs Google sign-in | Sign in from chat. Read mail and list events (read-only), send mail, create events. Writes go through per-action approval. |
| Phone calls | Coded, needs Twilio | Start an outbound call via Twilio Voice. Same gating. |
| Instagram | Coded, needs Meta app | Photo/Reel publishing containers for professional accounts. |
| X (Twitter) | Coded, needs paid X API | Post creation. X has no permanent free tier (pay-per-use). |
| YouTube | Coded, needs Google sign-in | Upload with native scheduling via `publishAt`. |
| Snapchat | Scaffold only | Posting is limited to Snap's partner Public Profile API. See docs. |
| Spotify | Coded, needs Spotify app | Search and start playback on a user's device. New apps are limited to 5 allowlisted users until Spotify grants extended quota. |

Per-platform credentials, scopes and limits: [docs/integrations.md](docs/integrations.md).

## Safety model

Three layers gate every real-world action:

1. `JARVIS_ACTIONS_ENABLED=true` - capability switch, off by default. This
   is NOT consent.
2. Per-platform credentials - a connector with no credentials reports
   `not_configured` and refuses to act.
3. Per-action approval - every send, post, call, play or publish becomes a
   pending action and executes only after you approve that specific action
   (`POST /actions/{id}/approve`). Scheduling a post is itself the approval
   for that post: drafts publish at their scheduled time and cannot be
   posted twice (drafts are claimed atomically before publishing).

Read operations (search mail, list events, status) only need the platform
credential. Tests never touch the network.

## Running the API

Bind locally unless you add real authentication:

```bash
uvicorn jarvis.api:app --host 127.0.0.1 --reload
```

Set `JARVIS_API_KEY` to require an `X-API-Key` header on every request.
The app refuses to start with actions enabled and no API key unless
`JARVIS_ALLOW_INSECURE=true` (local demos only). Single-user, single-store:
there is no multi-tenant isolation.

## HTTP surface

```
POST /agent/chat            one bounded agent turn: plan, use tools, answer
POST /research/ask            question -> answer (synchronous)
POST /research/jobs           question -> background job (durable, requeued after restart)
GET  /research/jobs/{id}      poll job status -> answer_id
GET  /research/{id}           fetch a saved answer
GET  /research                recent answers
GET  /connectors/status       per-platform readiness
POST /content/generate        topic -> post draft via LLM (never publishes)
POST /content/drafts          create a social post draft (never publishes)
POST /content/drafts/{id}/schedule
POST /content/publish-due     publish everything due (run on a cron)
POST /gmail/send              request an email send -> pending action
POST /calendar/events         request an event -> pending action
POST /calls                   request a call -> pending action
POST /spotify/play            request playback -> pending action
POST /content/drafts/{id}/publish-now -> pending action
GET  /actions                 list pending/decided actions
POST /actions/{id}/approve    execute one specific action
POST /actions/{id}/reject     block it
GET  /auth/google/callback    where Google sends your browser after sign-in (never called by hand)
```

## Layout

```
src/jarvis/
  config.py             settings from environment, credential gating
  llm.py                LangChain chat model: local Qwen3 default, OpenAI opt-in
  google_oauth.py       Google sign-in: link, callback, token renewal
  agent/                LangGraph agent loop + tool wiring
  schemas.py            shared pydantic models
  research/             question -> research job -> sourced answer -> store
  connectors/           one module per platform family, all gated
  actions.py            per-action approval gate
  content.py            LLM-backed post draft generation
  scheduler.py          scheduled publishing queue (SQLite, atomic claims)
  api.py                FastAPI surface
tests/                  offline test suite
docs/integrations.md    per-platform setup, scopes and limits
```

## Notes on platform names

The original feature list mentioned "snaptrack" and "putube". These are
treated as Snapchat and YouTube throughout, but that reading was never
confirmed - rename or drop the Snapchat scaffold if wrong.
