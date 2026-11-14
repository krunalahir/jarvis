# Platform integrations

Every integration is real API client code gated behind credentials plus
the `JARVIS_ACTIONS_ENABLED` switch. Facts below were checked against
official platform documentation in November 2026; re-verify before
launch, since platform terms change.

## Research providers (pick one)

| Provider | Env var | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | Responses API with the web_search tool; inline URL citations. |
| Perplexity | `PERPLEXITY_API_KEY` | `sonar` model; citations list maps to inline [n] markers. |
| Exa | `EXA_API_KEY` | Search-only: returns sources, no synthesized summary. |

## Google (Gmail, Calendar, YouTube)

One OAuth token, env var `GOOGLE_ACCESS_TOKEN`. Scopes:
`gmail.send`, `calendar.events`, `youtube.upload`.

You must create your own Google Cloud project, configure the OAuth
consent screen and run your own OAuth flow. Apps using sensitive scopes
need Google app verification before broad rollout. JARVIS does not and
cannot reuse credentials from any other assistant product.

## Calling (Twilio)

Google has no consumer calling API. JARVIS calls through Twilio Voice:
`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`.
`start_call` plays text-to-speech via TwiML. A real voice-agent bridge
(STT -> agent -> TTS over Twilio Media Streams) is a later milestone.

## Instagram

Meta Graph API Content Publishing. Requires a professional (business or
creator) account and a Meta app. `META_ACCESS_TOKEN`,
`INSTAGRAM_BUSINESS_ACCOUNT_ID`. Two-step publish (container, then
publish), media must sit at a public URL, 100 API posts per 24h, and the
API publishes immediately - scheduling is JARVIS's own scheduler.

## X (Twitter)

`X_BEARER_TOKEN`. Posting works but X API access is pay-per-use with no
permanent free tier; every publish call costs the developer account.

## YouTube

Uses the Google token (`youtube.upload` scope). `videos.insert` supports
native scheduling: upload private with `status.publishAt` and YouTube
releases at that time. Google's documented upload quota is 100 calls per
day in the Video Uploads bucket.

## Snapchat (unconfirmed name)

The original request said "snaptrack" - read as Snapchat, never
confirmed. There is no general public posting API. Snap's Public Profile
API (partner-gated) posts Stories/Spotlights for Public Profiles;
Creative Kit covers only user-initiated sharing. The connector is a
scaffold that fails loudly until partner access exists.

## Spotify

`SPOTIFY_ACCESS_TOKEN` with `user-read-playback-state` / `user-modify-playback-state`
scopes. New apps are in development mode: 5 allowlisted users max and the
owner needs Premium for playback control. Apply for extended quota mode
before launch.
