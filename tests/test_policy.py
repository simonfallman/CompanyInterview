"""Stage [4] tests — policy engine.

Each rule has at least one test where it fires and one where it does not.
Plus engine-level property tests for severity hierarchy and reason collection.
Test names read as business rules.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from verifier.policy import decide
from verifier.schemas import (
    ExtractedEmail,
    LineDiff,
    MatchResult,
    POMatch,
    ResolvedVendor,
    Severity,
)


# ---------------------------------------------------------------------------
# defaults — small helpers for clean inputs that pass every rule
# ---------------------------------------------------------------------------


def _vendor(**overrides) -> ResolvedVendor:
    base = dict(
        vendor_id="V-007",
        legal_name="Nordic Bearings AB",
        iban="SE45 5000 0000 0583 9825 7466",
        default_currency="SEK",
        last_verified_at=date(2026, 2, 14),
        match_basis="exact_email",
    )
    base.update(overrides)
    return ResolvedVendor(**base)


def _extracted(**overrides) -> ExtractedEmail:
    base = dict(
        invoice_number="333954",
        po_numbers=["PO-2026-0001"],
        claimed_total=Decimal("1000.00"),
        currency="SEK",
        iban_in_message=None,
        intent="invoice",
        reasoning="ok",
    )
    base.update(overrides)
    return ExtractedEmail(**base)


def _po_match(**overrides) -> POMatch:
    base = dict(
        po_number="PO-2026-0001",
        po_found=True,
        po_vendor_id="V-007",
        po_currency="SEK",
        authorized_total=Decimal("1000.00"),
        received_total=Decimal("1000.00"),
        line_diffs=[
            LineDiff(
                po_number="PO-2026-0001",
                po_line_no=1,
                sku="BRG-001",
                quantity_ordered=10,
                quantity_received=10,
                condition="OK",
                unit_price=Decimal("100.00"),
                line_total=Decimal("1000.00"),
            )
        ],
    )
    base.update(overrides)
    return POMatch(**base)


def _match(**overrides) -> MatchResult:
    po = overrides.pop("po_match", _po_match())
    base = dict(
        matched_pos=[po],
        aggregate_authorized_total=po.authorized_total,
        aggregate_received_total=po.received_total,
        currencies_seen={po.po_currency} if po.po_currency else set(),
    )
    base.update(overrides)
    return MatchResult(**base)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_clean_inputs_approve():
    d = decide("M-0001", _vendor(), _extracted(), _match())
    assert d.decision == Severity.APPROVE
    assert d.reason_codes == ["CLEAN_MATCH"]


# ---------------------------------------------------------------------------
# REJECT-severity rules
# ---------------------------------------------------------------------------


def test_unknown_sender_rejects():
    d = decide("M-0002", None, _extracted(), _match())
    assert d.decision == Severity.REJECT
    assert "UNKNOWN_SENDER" in d.reason_codes


def test_iban_mismatch_rejects():
    extracted = _extracted(iban_in_message="DE89999999999999999999")
    d = decide("M-0003", _vendor(), extracted, _match())
    assert d.decision == Severity.REJECT
    assert "IBAN_MISMATCH" in d.reason_codes


def test_iban_normalization_handles_spaces():
    """`SE45 5000` and `SE455000` must compare equal."""
    extracted = _extracted(iban_in_message="SE4550000000058398257466")
    d = decide("M-0004", _vendor(), extracted, _match())
    assert "IBAN_MISMATCH" not in d.reason_codes


def test_iban_absent_does_not_fire():
    d = decide("M-0005", _vendor(), _extracted(iban_in_message=None), _match())
    assert "IBAN_MISMATCH" not in d.reason_codes


def test_vendor_po_mismatch_rejects():
    """PO is registered to V-099 but email resolved to V-007."""
    pom = _po_match(po_vendor_id="V-099")
    d = decide("M-0006", _vendor(), _extracted(), _match(po_match=pom))
    assert d.decision == Severity.REJECT
    assert "VENDOR_PO_MISMATCH" in d.reason_codes


# ---------------------------------------------------------------------------
# HOLD-severity rules
# ---------------------------------------------------------------------------


def test_bank_change_holds():
    d = decide("M-0007", _vendor(), _extracted(intent="bank_change"), _match())
    assert d.decision == Severity.HOLD
    assert "BANK_CHANGE_REQUIRES_VERIFICATION" in d.reason_codes


def test_mixed_message_with_iban_holds_as_bank_change():
    """A `mixed` message that contains an IBAN should also trigger bank-change."""
    extracted = _extracted(intent="mixed", iban_in_message=_vendor().iban)
    d = decide("M-0008", _vendor(), extracted, _match())
    assert "BANK_CHANGE_REQUIRES_VERIFICATION" in d.reason_codes


def test_unverified_alias_holds():
    """Sender domain matched but email alias is not in the authorized list."""
    d = decide(
        "M-0009",
        _vendor(match_basis="domain"),
        _extracted(),
        _match(),
    )
    assert d.decision == Severity.HOLD
    assert "UNVERIFIED_ALIAS" in d.reason_codes


def test_missing_po_reference_holds():
    extracted = _extracted(po_numbers=[])
    # No POs referenced -> empty match
    d = decide("M-0010", _vendor(), extracted, MatchResult())
    assert d.decision == Severity.HOLD
    assert "MISSING_PO_REFERENCE" in d.reason_codes


def test_followup_without_po_does_not_fire_missing_po():
    """A follow-up email is not an invoice; missing PO is OK on follow-ups."""
    extracted = _extracted(intent="followup", po_numbers=[])
    d = decide("M-0011", _vendor(), extracted, MatchResult())
    assert "MISSING_PO_REFERENCE" not in d.reason_codes


def test_unknown_po_holds():
    """Email references a PO not in the ERP."""
    pom = _po_match(po_found=False, po_vendor_id=None, po_currency=None,
                    authorized_total=Decimal("0"), received_total=Decimal("0"),
                    line_diffs=[])
    d = decide("M-0012", _vendor(), _extracted(), _match(po_match=pom))
    assert d.decision == Severity.HOLD
    assert "UNKNOWN_PO" in d.reason_codes


def test_currency_mismatch_holds():
    extracted = _extracted(currency="EUR")
    d = decide("M-0013", _vendor(), extracted, _match())
    assert d.decision == Severity.HOLD
    assert "CURRENCY_MISMATCH" in d.reason_codes


def test_overbilling_holds():
    extracted = _extracted(claimed_total=Decimal("1500.00"))  # authorized is 1000
    d = decide("M-0014", _vendor(), extracted, _match())
    assert "OVERBILLING" in d.reason_codes


def test_overbilling_within_tolerance_does_not_fire():
    extracted = _extracted(claimed_total=Decimal("1000.005"))
    d = decide("M-0015", _vendor(), extracted, _match())
    assert "OVERBILLING" not in d.reason_codes


def test_underdelivered_holds_when_invoice_exceeds_received():
    """Invoice asks for 1000 but only 500 worth of goods has arrived."""
    pom = _po_match(received_total=Decimal("500.00"))
    d = decide("M-0016", _vendor(), _extracted(), _match(po_match=pom))
    assert "UNDERDELIVERED" in d.reason_codes


def test_delivery_exception_holds_on_damaged():
    """Any line marked DAMAGED triggers DELIVERY_EXCEPTION."""
    diff = LineDiff(
        po_number="PO-2026-0001", po_line_no=1, sku="BRG-001",
        quantity_ordered=10, quantity_received=10, condition="DAMAGED",
        unit_price=Decimal("100"), line_total=Decimal("1000"),
    )
    pom = _po_match(line_diffs=[diff])
    d = decide("M-0017", _vendor(), _extracted(), _match(po_match=pom))
    assert "DELIVERY_EXCEPTION" in d.reason_codes


def test_delivery_exception_holds_on_short():
    diff = LineDiff(
        po_number="PO-2026-0001", po_line_no=1, sku="BRG-001",
        quantity_ordered=10, quantity_received=8, condition="SHORT",
        unit_price=Decimal("100"), line_total=Decimal("1000"),
    )
    pom = _po_match(line_diffs=[diff], received_total=Decimal("800"))
    d = decide("M-0018", _vendor(), _extracted(), _match(po_match=pom))
    assert "DELIVERY_EXCEPTION" in d.reason_codes


def test_failed_extraction_holds():
    """Empty extraction (LLM failure) holds rather than approves by default."""
    extracted = ExtractedEmail()  # all fields empty / default
    d = decide("M-0019", _vendor(), extracted, MatchResult())
    assert d.decision == Severity.HOLD
    assert "FAILED_EXTRACTION" in d.reason_codes


# ---------------------------------------------------------------------------
# engine-level property tests
# ---------------------------------------------------------------------------


def test_severity_hierarchy_reject_trumps_hold():
    """Both an IBAN_MISMATCH (REJECT) and CURRENCY_MISMATCH (HOLD) fire.
    Final decision must be REJECT, but BOTH reason codes appear in output.
    """
    extracted = _extracted(
        iban_in_message="DE89999999999999999999",  # REJECT
        currency="EUR",  # HOLD
    )
    d = decide("M-0020", _vendor(), extracted, _match())
    assert d.decision == Severity.REJECT
    assert "IBAN_MISMATCH" in d.reason_codes
    assert "CURRENCY_MISMATCH" in d.reason_codes


def test_all_holds_collected_not_just_first():
    """Two HOLD-severity rules fire — both reasons must appear."""
    pom = _po_match(received_total=Decimal("500.00"))
    extracted = _extracted(currency="EUR")  # CURRENCY_MISMATCH
    # received < claimed: UNDERDELIVERED also fires
    d = decide("M-0021", _vendor(), extracted, _match(po_match=pom))
    assert d.decision == Severity.HOLD
    assert "CURRENCY_MISMATCH" in d.reason_codes
    assert "UNDERDELIVERED" in d.reason_codes


def test_clean_match_reason_only_appears_when_decision_is_approve():
    """CLEAN_MATCH must not appear alongside any failure reason."""
    extracted = _extracted(currency="EUR")  # forces a HOLD
    d = decide("M-0022", _vendor(), extracted, _match())
    assert "CLEAN_MATCH" not in d.reason_codes


def test_decision_object_carries_full_evidence():
    """The Decision must carry vendor, extracted, and match for the audit trail."""
    d = decide("M-0023", _vendor(), _extracted(), _match())
    assert d.vendor is not None
    assert d.extracted is not None
    assert d.match is not None
    assert d.audit_summary != ""
