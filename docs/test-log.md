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
