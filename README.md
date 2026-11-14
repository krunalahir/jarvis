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
uvicorn jarvis.api:app --reload
```

API docs: http://localhost:8000/docs

## Feature map

| Feature | Status | What works today |
|---|---|---|
| Research & Q&A | Working, tested | Ask a question (sync or as a background job), get a source-backed answer, saved results. Needs one search provider key (OpenAI, Perplexity or Exa). |
| Gmail / Google Calendar | Coded, needs Google OAuth | Read mail and list events (read-only), send mail, create events. Writes stay disabled until a Google access token is configured and actions are enabled. |
| Phone calls | Coded, needs Twilio | Start an outbound call via Twilio Voice. Same gating. |
| Instagram | Coded, needs Meta app | Photo/Reel publishing containers for professional accounts. |
| X (Twitter) | Coded, needs paid X API | Post creation. X has no permanent free tier (pay-per-use). |
| YouTube | Coded, needs Google OAuth | Upload with native scheduling via `publishAt`. |
| Snapchat | Scaffold only | Posting is limited to Snap's partner Public Profile API. See docs. |
| Spotify | Coded, needs Spotify app | Search and start playback on a user's device. New apps are limited to 5 allowlisted users until Spotify grants extended quota. |

Per-platform credentials, scopes and limits: [docs/integrations.md](docs/integrations.md).

## Safety model

Two switches gate every real-world action:

1. `JARVIS_ACTIONS_ENABLED=true` - master switch, off by default.
2. Per-platform credentials - a connector with no credentials reports
   `not_configured` and refuses to act.

Read operations (like checking status) only need the platform credential.
Write operations (send, post, call, play, publish) need both switches.
Tests never touch the network.

## HTTP surface

```
POST /research/ask            question -> answer (synchronous)
POST /research/jobs           question -> background job (deep research can take minutes)
GET  /research/jobs/{id}      poll job status -> answer_id
GET  /research/{id}           fetch a saved answer
GET  /research                recent answers
GET  /connectors/status       per-platform readiness
POST /content/drafts          create a social post draft (never publishes)
POST /content/drafts/{id}/schedule
POST /content/publish-due     publish everything due (run on a cron)
POST /spotify/play            start playback
```

## Layout

```
src/jarvis/
  config.py             settings from environment, credential gating
  schemas.py            shared pydantic models
  research/             question -> research job -> sourced answer -> store
  connectors/           one module per platform family, all gated
  scheduler.py          scheduled publishing queue (SQLite)
  api.py                FastAPI surface
tests/                  offline test suite
docs/integrations.md    per-platform setup, scopes and limits
```

## Notes on platform names

The original feature list mentioned "snaptrack" and "putube". These are
treated as Snapchat and YouTube throughout, but that reading was never
confirmed - rename or drop the Snapchat scaffold if wrong.
