# Testing

Last full run: **105 passed** (5 consecutive full-suite runs, no failures),
November 16, 2026. Run it yourself with `pytest` - no credentials, no
network, no model server needed.

| File | Tests | Covers |
|---|---|---|
| tests/test_actions.py | 8 | Approval gate cycle, executor failure visibility, durable jobs, citation validation, duplicate-post prevention |
| tests/test_agent.py | 5 | Agent loop iteration, write tools only queue approvals, step-limit bound, stored cited answers |
| tests/test_api.py | 10 | Health, research endpoints, draft/schedule flow, API key enforcement, pending-until-approved, refusal to start insecure |
| tests/test_connectors.py | 10 | Connector gating (configured/enabled), Gmail send shape, Instagram two-step, X post, Spotify search/play, Twilio gate, Snapchat scaffold |
| tests/test_content.py | 4 | Content generation via model, unavailable states |
| tests/test_features.py | 32 | Scheduler semantics, approval over HTTP, recovery edge cases, store internals, Gmail/Calendar/Instagram/X/YouTube/Spotify/Twilio request shapes, agent write tools, API security surface |
| tests/test_llm.py | 4 | Local default, OpenAI opt-in gating, unreachable-server errors |
| tests/test_local_research.py | 5 | Local provider: search-key requirement, synthesis only from sources |
| tests/test_providers.py | 6 | Provider mapping, HTTP failure handling, scheduler success path |
| tests/test_regressions.py | 15 | The Nov 2026 review findings (below) |
| tests/test_research.py | 6 | Research pipeline, unsupported claims surfaced |

## What the November 2026 review found, and the fix + test for each

1. **Approval race (double execution)** - `ActionGate.approve` read the
   status, then executed, then saved. Two overlapping approvals could both
   pass the check and both execute. Fixed: approvals claim the action
   atomically (`UPDATE ... WHERE status='pending'`), only the winner runs
   the executor. Tests: `test_concurrent_approvals_execute_once`,
   `test_concurrent_reject_and_approve_only_one_wins`. The old code was
   verified to double-execute under the same test before the fix.
2. **Gmail search returned only message IDs** - the agent received id
   stubs, never content. Fixed: `GmailConnector.read_messages` fetches
   each message (sender, subject, date, plain-text body) and the
   `read_emails` agent tool formats those. Tests:
   `test_gmail_read_messages_fetches_bodies`,
   `test_gmail_html_only_message_falls_back_to_snippet`,
   `test_gmail_read_messages_empty_search`.
3. **YouTube publishing could never work end to end** - the connector
   required a local file path that drafts and the scheduler never carried.
   Fixed: the connector downloads the draft's first media URL when no
   path is given; local paths still work. Tests:
   `test_youtube_publish_downloads_video_from_draft_media_url`,
   `test_youtube_draft_publishes_end_to_end_through_scheduler`,
   `test_youtube_upload_from_local_file_path`,
   `test_youtube_scheduled_draft_sets_native_publish_at`,
   `test_youtube_publish_without_video_source_fails_loudly`.
4. **SQLite connection shared across threads** - one connection, no lock,
   no busy timeout, shared by the API, job pool and scheduler. Fixed: a
   store-wide re-entrant lock around every method, WAL journal mode, 5s
   busy timeout. Tests: `test_store_survives_concurrent_writers`,
   `test_file_store_sets_wal_and_busy_timeout`.
5. **Draft stuck in 'publishing' after a crash** - startup recovery only
   requeued research jobs. Fixed: `recovery.py` marks interrupted actions
   and drafts failed-unknown on startup. It never retries: the external
   effect may already have happened, and a blind retry could double-send.
   Tests: `test_interrupted_action_marked_unknown_and_not_retried`,
   `test_interrupted_draft_marked_unknown_and_not_republished`,
   `test_create_app_recovers_interrupted_state_on_startup`,
   `test_recovery_leaves_pending_actions_alone`,
   `test_recovery_is_noop_on_clean_store`.
6. **Unescaped TwiML** - spoken text with `<`, `>` or `&` broke the call
   markup. Fixed: XML-escaped. Test: `test_twilio_twiml_escapes_spoken_text`.
7. **Unvalidated agent history** - arbitrary roles and unbounded size
   could break the run or overflow the model context. Fixed: request
   validation (roles, content type, per-message and total caps) plus a
   recent-turns cap in the loop. Tests:
   `test_agent_chat_rejects_bad_history_role`,
   `test_agent_chat_rejects_oversized_history_content`,
   `test_agent_chat_rejects_overlong_history`,
   `test_run_agent_truncates_history_to_recent_turns`.

## What these tests prove, and what they cannot

These are offline tests: every external call is mocked, and the model is
a scripted fake. They prove the code paths, gating, atomicity and failure
handling behave as designed. They do NOT prove:

- real inference quality of the local Qwen3 8B (needs the model server
  running - see docs/local-model.md),
- live Gmail, Calendar, Twilio, Instagram, X, YouTube or Spotify calls
  (need real credentials and controlled live testing),
- Exa web search (needs EXA_API_KEY).

Before enabling real actions: run the suite, start the local model, do a
dummy-data agent chat, then enable one connector at a time with
`JARVIS_ACTIONS_ENABLED=true` and an API key set.
