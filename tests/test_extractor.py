"""Stage [2] extractor wiring tests.

All tests are mock-based — no real OpenAI calls, runs in milliseconds.

We test three things:
1. The happy path: extractor returns what the LLM gives it.
2. The retry path: first call returns None; second call succeeds.
3. The permanent-failure path: all calls return None; extractor returns an
   empty ExtractedEmail so the policy engine can HOLD via FAILED_EXTRACTION.
4. load_raw_email: correctly reads body + attachments from disk.
5. build_user_message: assembled prompt contains required fields.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from verifier.extractor import OpenAIExtractor, build_user_message, load_raw_email
from verifier.schemas import EmailMeta, ExtractedEmail, RawEmail


# ---------------------------------------------------------------------------
# Helpers — build minimal EmailMeta without touching the filesystem
# ---------------------------------------------------------------------------


def _make_meta(**overrides) -> EmailMeta:
    base = dict(
        msg_id="M-TEST",
        thread_id="T-TEST",
        in_reply_to=None,
        sent_at="2026-01-01T00:00:00+00:00",
        sender_name="Test Sender",
        sender_email="billing@test-vendor.se",
        subject="Test invoice",
        language="en",
        message_kind="invoice",
        body_path="bodies/test.md",
        attachment_paths=[],
    )
    base.update(overrides)
    return EmailMeta(**base)


def _make_raw(body: str = "Hello", attachments: dict | None = None) -> RawEmail:
    return RawEmail(
        meta=_make_meta(),
        body=body,
        attachments=attachments or {},
    )


def _make_parsed_response(extracted: ExtractedEmail) -> MagicMock:
    """Build a mock that looks like the object OpenAI returns from .parse()."""
    parsed_response = MagicMock()
    parsed_response.choices[0].message.parsed = extracted
    return parsed_response


# ---------------------------------------------------------------------------
# Test 1 — happy path: extractor returns what the LLM parsed
# ---------------------------------------------------------------------------


def test_extractor_returns_parsed_object_on_success():
    expected = ExtractedEmail(
        invoice_number="INV-001",
        po_numbers=["PO-2026-0001"],
        claimed_total="36975.00",
        currency="SEK",
        iban_in_message=None,
        intent="invoice",
        reasoning="The email explicitly states invoice INV-001.",
    )

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = _make_parsed_response(expected)

    extractor = OpenAIExtractor(client=mock_client)
    result = extractor.extract(_make_raw())

    assert result == expected
    assert mock_client.beta.chat.completions.parse.call_count == 1


# ---------------------------------------------------------------------------
# Test 2 — retry path: first call returns None parsed; second call succeeds
# ---------------------------------------------------------------------------


def test_extractor_retries_on_empty_response():
    success_extract = ExtractedEmail(
        invoice_number="INV-002",
        po_numbers=["PO-2026-0002"],
        claimed_total="1000.00",
        currency="SEK",
        intent="invoice",
        reasoning="Second attempt succeeded.",
    )

    # First call: parsed is None (refusal or schema mismatch)
    first_response = MagicMock()
    first_response.choices[0].message.parsed = None

    second_response = _make_parsed_response(success_extract)

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = [first_response, second_response]

    extractor = OpenAIExtractor(client=mock_client, max_retries=1)
    result = extractor.extract(_make_raw())

    assert result == success_extract
    # Must have tried twice — once for the None, once for the success
    assert mock_client.beta.chat.completions.parse.call_count == 2


# ---------------------------------------------------------------------------
# Test 3 — permanent failure: all calls return None; extractor returns empty
# ---------------------------------------------------------------------------


def test_extractor_returns_empty_object_after_all_retries_fail():
    failing_response = MagicMock()
    failing_response.choices[0].message.parsed = None

    mock_client = MagicMock()
    # Both attempts fail
    mock_client.beta.chat.completions.parse.side_effect = [
        failing_response,
        failing_response,
    ]

    extractor = OpenAIExtractor(client=mock_client, max_retries=1)
    result = extractor.extract(_make_raw())

    # Result must be a default ExtractedEmail — all fields None / empty
    assert isinstance(result, ExtractedEmail)
    assert result.invoice_number is None
    assert result.po_numbers == []
    assert result.claimed_total is None
    assert result.currency is None
    assert result.iban_in_message is None
    assert result.intent == "unknown"
    # Critically: the policy engine's FAILED_EXTRACTION rule must fire for this
    # We verify it by checking the exact fields the rule tests (see policy.py)
    assert not result.po_numbers
    assert not result.claimed_total
    assert not result.invoice_number


# ---------------------------------------------------------------------------
# Test 4 — load_raw_email: reads body and attachments from disk
# ---------------------------------------------------------------------------


def test_load_raw_email_concatenates_body_and_attachments():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create directory structure mirroring the real inbox layout
        (root / "bodies").mkdir()
        (root / "attachments").mkdir()

        body_content = "Invoice total: SEK 1,234.00"
        attachment_content = "IBAN: SE12 3456 7890 1234 5678 9012"

        (root / "bodies" / "test.md").write_text(body_content, encoding="utf-8")
        (root / "attachments" / "inv_test.md").write_text(attachment_content, encoding="utf-8")

        meta = _make_meta(
            body_path="bodies/test.md",
            attachment_paths=["attachments/inv_test.md"],
        )

        raw = load_raw_email(meta, root)

        assert raw.meta == meta
        assert raw.body == body_content
        assert "attachments/inv_test.md" in raw.attachments
        assert raw.attachments["attachments/inv_test.md"] == attachment_content


# ---------------------------------------------------------------------------
# Test 5 — build_user_message: prompt contains subject, sender, body, attachments
# ---------------------------------------------------------------------------


def test_build_user_message_includes_metadata_and_attachments():
    meta = _make_meta(
        subject="Invoice 42",
        sender_name="Lars Forsberg",
        sender_email="billing@nordic-maskin.se",
        sent_at="2026-04-13T19:21:31+00:00",
        language="en",
    )
    body_text = "Please find invoice 42 attached."
    attachment_text = "Total: SEK 999.00"

    raw = RawEmail(
        meta=meta,
        body=body_text,
        attachments={"attachments/inv_042.md": attachment_text},
    )

    message = build_user_message(raw)

    # Subject and sender must appear
    assert "Invoice 42" in message
    assert "Lars Forsberg" in message
    assert "billing@nordic-maskin.se" in message
    # Sent timestamp must appear
    assert "2026-04-13T19:21:31+00:00" in message
    # Body content must appear
    assert body_text in message
    # Attachment path and content must appear
    assert "attachments/inv_042.md" in message
    assert attachment_text in message
