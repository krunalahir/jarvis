# JARVIS

A personal AI assistant: research-first, then connected actions.

JARVIS answers questions with source-backed research, then acts through
connected apps: Google mail and calendar, phone calls, social content
creation with scheduled publishing, and Spotify playback.

This repository is the base version. The research/Q&A core runs and is
tested offline. App connectors are real API client code, but each one
stays inert until you supply that platform's credentials and explicitly
enable actions. Nothing here posts, sends, calls or plays without that.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env   # fill in what you have; everything is optional to start
pytest                 # run the test suite (no credentials needed)
uvicorn jarvis.api:app --host 127.0.0.1 --reload
```

API docs: http://localhost:8000/docs

## Feature map

| Feature | Status | What works today |
|---|---|---|
| Research & Q&A | Working, tested | Ask a question (sync or as a durable background job), get a source-backed answer, saved results. Needs one search provider key (OpenAI, Perplexity or Exa). |
| Content creation | Working, tested | Topic -> post draft via OpenAI, per platform. Never publishes. |
| Gmail / Google Calendar | Coded, needs Google OAuth | Read mail and list events (read-only), send mail, create events. Writes go through per-action approval. |
| Phone calls | Coded, needs Twilio | Start an outbound call via Twilio Voice. Same gating. |
| Instagram | Coded, needs Meta app | Photo/Reel publishing containers for professional accounts. |
| X (Twitter) | Coded, needs paid X API | Post creation. X has no permanent free tier (pay-per-use). |
| YouTube | Coded, needs Google OAuth | Upload with native scheduling via `publishAt`. |
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
```

## Layout

```
src/jarvis/
  config.py             settings from environment, credential gating
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
