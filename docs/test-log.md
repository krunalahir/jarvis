# Test log

All times IST. Every entry is a real `pytest` run against the code named
in the entry. This log started after the review-fix work landed; the
earlier chronology is recorded once here, not backfilled as fake
per-test commits.

## 2026-11-16

- 00:38-00:46 - Local development runs while fixing the 7 review
  findings: full suite reached 105 passed (58 pre-existing + 47 new in
  tests/test_regressions.py and tests/test_features.py); 5 consecutive
  full-suite runs, no failures. The approval race was reproduced on the
  pre-fix code (executor ran twice) before the fix, proving the test.
- 00:47-00:55 - Fix commits landed on main through the GitHub web UI
  (9 commits), then a repair series replaced the accidentally prefixed
  upload files (1-actions.py et al.) with the correctly named files.
- 00:56 - Fresh checkout of main, clean environment: 105 passed in 2.24s.
- 00:57 - Per-area runs on the fresh checkout, review-fix checks:
  `tests/test_regressions.py` 15 passed in 1.41s; `tests/test_features.py`
  32 passed in 0.99s.
- 00:57 - Core safety areas: `tests/test_actions.py` 8 passed in 0.53s
  (approval gate, jobs, citations); `tests/test_agent.py` 5 passed in
  0.72s (agent loop bounds, write tools only queue);
  `tests/test_api.py` 10 passed in 0.96s (HTTP surface, API key).
- 00:57 - Connector and provider areas: `tests/test_connectors.py` 10
  passed in 0.07s; `tests/test_content.py` 4 passed in 0.44s;
  `tests/test_llm.py` 4 passed in 0.43s; `tests/test_local_research.py`
  5 passed in 0.53s; `tests/test_providers.py` 6 passed in 0.51s;
  `tests/test_research.py` 6 passed in 0.52s.
- 00:58 - Final full-suite confirmation on the fresh checkout after every
  repair commit: 105 passed in 2.13s. Tree verified clean: no prefixed
  duplicate files, no missing files, remote content byte-identical to the
  tested working copy (trailing newline aside).

## Individual per-test runs

Each line below is one distinct test executed on its own
(`pytest <test>`), with this file committed to GitHub right after that
execution and before the next one, in the order shown. All times IST,
November 16, 2026. Run against the working copy that matches `main`
(trailing newline aside), after the media-download hardening commits.
- 01:08 IST - PASS (1.54s) - `tests/test_regressions.py::test_concurrent_approvals_execute_once`
- 01:08 IST - PASS (1.49s) - `tests/test_regressions.py::test_concurrent_reject_and_approve_only_one_wins`
- 01:09 IST - PASS (1.22s) - `tests/test_regressions.py::test_interrupted_action_marked_unknown_and_not_retried`
- 01:09 IST - PASS (1.40s) - `tests/test_regressions.py::test_interrupted_draft_marked_unknown_and_not_republished`
- 01:09 IST - PASS (1.27s) - `tests/test_regressions.py::test_create_app_recovers_interrupted_state_on_startup`
- 01:09 IST - PASS (1.26s) - `tests/test_regressions.py::test_store_survives_concurrent_writers`
- 01:10 IST - PASS (1.31s) - `tests/test_regressions.py::test_gmail_read_messages_fetches_bodies`
- 01:10 IST - PASS (1.42s) - `tests/test_regressions.py::test_youtube_publish_downloads_video_from_draft_media_url`
- 01:10 IST - PASS (1.19s) - `tests/test_regressions.py::test_youtube_publish_without_video_source_fails_loudly`
- 01:11 IST - PASS (1.19s) - `tests/test_regressions.py::test_youtube_scheduled_draft_sets_native_publish_at`
- 01:11 IST - PASS (1.35s) - `tests/test_regressions.py::test_media_url_rejects_non_http_schemes`
- 01:11 IST - PASS (1.25s) - `tests/test_regressions.py::test_media_url_rejects_private_and_metadata_ips`
- 01:11 IST - PASS (1.28s) - `tests/test_regressions.py::test_media_url_rejects_unresolvable_host`
- 01:12 IST - PASS (1.27s) - `tests/test_regressions.py::test_youtube_download_rejects_redirect_to_metadata_endpoint`
- 01:12 IST - PASS (1.21s) - `tests/test_regressions.py::test_youtube_download_enforces_size_cap_and_cleans_up`
- 01:12 IST - PASS (1.30s) - `tests/test_regressions.py::test_youtube_download_follows_valid_redirect`
- 01:12 IST - PASS (1.24s) - `tests/test_regressions.py::test_twilio_twiml_escapes_spoken_text`
- 01:13 IST - PASS (1.22s) - `tests/test_regressions.py::test_agent_chat_rejects_bad_history_role`
- 01:13 IST - PASS (1.41s) - `tests/test_regressions.py::test_agent_chat_rejects_oversized_history_content`
- 01:13 IST - PASS (1.20s) - `tests/test_regressions.py::test_agent_chat_rejects_overlong_history`
- 01:14 IST - PASS (1.30s) - `tests/test_regressions.py::test_run_agent_truncates_history_to_recent_turns`
- 01:14 IST - PASS (1.32s) - `tests/test_features.py::test_schedule_naive_datetime_is_treated_as_utc`
- 01:14 IST - PASS (1.24s) - `tests/test_features.py::test_schedule_unknown_draft_raises`
- 01:15 IST - PASS (1.24s) - `tests/test_features.py::test_failed_draft_is_not_retried_by_publish_due`
- 01:15 IST - PASS (1.39s) - `tests/test_features.py::test_draft_with_platform_post_id_is_never_reposted`
- 01:15 IST - PASS (1.21s) - `tests/test_features.py::test_youtube_draft_publishes_end_to_end_through_scheduler`
- 01:15 IST - PASS (1.24s) - `tests/test_features.py::test_approve_unknown_action_returns_404`
- 01:16 IST - PASS (1.41s) - `tests/test_features.py::test_publish_now_queues_approval_and_never_publishes_directly`
- 01:16 IST - PASS (1.23s) - `tests/test_features.py::test_publish_now_unknown_draft_is_404`
- 01:16 IST - PASS (1.36s) - `tests/test_features.py::test_approved_publish_executes_through_connector`
- 01:16 IST - PASS (1.22s) - `tests/test_features.py::test_recovery_is_noop_on_clean_store`
- 01:17 IST - PASS (1.25s) - `tests/test_features.py::test_recovery_leaves_pending_actions_alone`
- 01:17 IST - PASS (1.34s) - `tests/test_features.py::test_recovery_unknown_outcome_message_mentions_no_retry`
- 01:17 IST - PASS (1.24s) - `tests/test_features.py::test_claim_action_returns_none_for_decided_action`
- 01:18 IST - PASS (1.26s) - `tests/test_features.py::test_file_store_sets_wal_and_busy_timeout`
- 01:18 IST - PASS (1.53s) - `tests/test_features.py::test_gmail_html_only_message_falls_back_to_snippet`
- 01:18 IST - PASS (1.24s) - `tests/test_features.py::test_gmail_read_messages_empty_search`
- 01:18 IST - PASS (1.30s) - `tests/test_features.py::test_calendar_create_event_builds_payload`
- 01:19 IST - PASS (1.28s) - `tests/test_features.py::test_calendar_create_event_refused_when_actions_disabled`
