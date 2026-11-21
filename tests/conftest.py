"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _run_in_an_empty_folder(tmp_path, monkeypatch):
    """Run every test from an empty temporary folder.

    Settings() uses relative default paths (.google_tokens.json,
    jarvis_store.sqlite3). Without this, once you have really signed in to
    Google, tests run from the project folder would find your token file
    and behave as if Google were connected.
    """
    monkeypatch.chdir(tmp_path)
