"""End-to-end smoke tests for the full pipeline using real OpenAI calls.

These tests are skipped automatically when OPENAI_API_KEY is not set,
so CI stays green without credentials. Run them locally with the key in .env.

Each test runs the full 5-stage pipeline:
  EmailMeta -> load_raw_email -> resolve_sender -> LLM extract -> match -> decide

We assert on the final Decision, not on intermediate stages — these are
regression tests for the LLM-driven layer, not unit tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from openai import OpenAI

from verifier.extractor import OpenAIExtractor
from verifier.pipeline import Pipeline
from verifier.repositories import DataContext, load_inbox
from verifier.schemas import Severity

# Load .env before checking the key
load_dotenv()

_NEEDS_KEY = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-LLM tests",
)

# Paths are resolved relative to this file so tests work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "candidate-package" / "data"
_INBOX_ROOT = _DATA_DIR / "inbox"


@pytest.fixture(scope="module")
def pipeline():
    """Build the full pipeline once per module — data loading is the slow part."""
    data = DataContext.from_data_dir(_DATA_DIR)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    extractor = OpenAIExtractor(client=client, model="gpt-4o-mini-2024-07-18")
    return Pipeline(data=data, extractor=extractor, inbox_root=_INBOX_ROOT)


@pytest.fixture(scope="module")
def inbox():
    """All EmailMeta objects from the real inbox index, keyed by msg_id."""
    emails = load_inbox(_INBOX_ROOT / "index.csv")
    return {m.msg_id: m for m in emails}


# ---------------------------------------------------------------------------
# Smoke test 1 — clean invoice APPROVE (happy path keystone)
# ---------------------------------------------------------------------------


@_NEEDS_KEY
def test_pipeline_approves_clean_invoice_M0001(pipeline, inbox):
    """M-0001: Mälardal Komponenter, SEK 36975, PO-2026-0001. Should be APPROVE/CLEAN_MATCH."""
    meta = inbox["M-0001"]
    decision = pipeline.process(meta)

    assert decision.decision == Severity.APPROVE, (
        f"Expected APPROVE, got {decision.decision.value}. "
        f"Reasons: {decision.reason_codes}"
    )
    assert "CLEAN_MATCH" in decision.reason_codes, (
        f"Expected CLEAN_MATCH in reasons, got: {decision.reason_codes}"
    )
    # Verify the LLM extracted the right total and currency
    assert decision.extracted is not None
    assert str(decision.extracted.claimed_total) == "36975"
    assert decision.extracted.currency == "SEK"


# ---------------------------------------------------------------------------
# Smoke test 2 — typo-squat sender REJECT (fraud detection keystone)
# ---------------------------------------------------------------------------


@_NEEDS_KEY
def test_pipeline_rejects_unknown_sender_M0058(pipeline, inbox):
    """M-0058: finance@lndustri-verktyg.se (L not I) — should be REJECT/UNKNOWN_SENDER."""
    meta = inbox["M-0058"]
    decision = pipeline.process(meta)

    assert decision.decision == Severity.REJECT, (
        f"Expected REJECT, got {decision.decision.value}. "
        f"Reasons: {decision.reason_codes}"
    )
    assert "UNKNOWN_SENDER" in decision.reason_codes, (
        f"Expected UNKNOWN_SENDER in reasons, got: {decision.reason_codes}"
    )
    # Vendor must be None — the sender is unknown
    assert decision.vendor is None, (
        f"Expected vendor=None for unknown sender, got: {decision.vendor}"
    )
