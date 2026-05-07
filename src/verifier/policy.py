"""Stage [4] — Policy engine.

Pure deterministic rule table. Takes the resolved vendor, extracted email,
and match result; emits a `Decision` with stacked reason codes.

Two design properties that matter, in this order:

1. **All rules fire.** Every rule is evaluated; every reason that fires is
   reported. AP wants the full picture, not just the first issue we noticed.

2. **Severity hierarchy decides the action.** REJECT > HOLD > APPROVE.
   A single REJECT-severity reason wins, even if HOLDs also fired.

The rule table is policy-as-code: it doubles as company AP policy
documentation, in a form that compiles and is unit-tested.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Callable, Optional

from verifier.schemas import (
    Decision,
    ExtractedEmail,
    MatchResult,
    Reason,
    ResolvedVendor,
    Severity,
)


# Tolerance for money comparisons (handle rounding cleanly without false positives)
TOLERANCE = Decimal("0.01")


# ---------------------------------------------------------------------------
# Individual rules — each returns a Reason if it fires, else None.
# Each is independent and idempotent.
# ---------------------------------------------------------------------------

RuleFn = Callable[
    [Optional[ResolvedVendor], ExtractedEmail, MatchResult], Optional[Reason]
]


def rule_unknown_sender(v, e, m) -> Optional[Reason]:
    """REJECT when the sender email does not resolve to any vendor in the master file."""
    if v is None:
        return Reason(
            code="UNKNOWN_SENDER",
            severity=Severity.REJECT,
            message="Sender does not match any vendor in the master file.",
        )
    return None


def rule_iban_mismatch(v, e, m) -> Optional[Reason]:
    """REJECT when the IBAN in the email differs from the vendor master's authorized IBAN."""
    if v is None or not e.iban_in_message:
        return None
    if _norm_iban(e.iban_in_message) != _norm_iban(v.iban):
        return Reason(
            code="IBAN_MISMATCH",
            severity=Severity.REJECT,
            message=(
                f"IBAN in message does not match vendor master IBAN. "
                f"Message IBAN: {e.iban_in_message!r}; vendor IBAN on file: {v.iban!r}."
            ),
        )
    return None


def rule_vendor_po_mismatch(v, e, m) -> Optional[Reason]:
    """REJECT when a referenced PO belongs to a different vendor than the email sender.

    Guards against supplier impersonation: a legitimate vendor sending an invoice
    against another vendor's PO.
    """
    if v is None:
        return None
    for po in m.matched_pos:
        if po.po_found and po.po_vendor_id and po.po_vendor_id != v.vendor_id:
            return Reason(
                code="VENDOR_PO_MISMATCH",
                severity=Severity.REJECT,
                message=(
                    f"PO {po.po_number} is registered to vendor {po.po_vendor_id}, "
                    f"but the email resolved to vendor {v.vendor_id}."
                ),
            )
    return None


def rule_bank_change(v, e, m) -> Optional[Reason]:
    if e.intent in ("bank_change", "mixed"):
        # Mixed messages can include bank_change intent buried in them — also HOLD.
        if e.intent == "bank_change" or e.iban_in_message:
            return Reason(
                code="BANK_CHANGE_REQUIRES_VERIFICATION",
                severity=Severity.HOLD,
                message=(
                    "Email requests a change to bank/IBAN details. "
                    "Never auto-action; AP must verify out-of-band before updating."
                ),
            )
    return None


def rule_unverified_alias(v, e, m) -> Optional[Reason]:
    if v is not None and v.match_basis == "domain":
        return Reason(
            code="UNVERIFIED_ALIAS",
            severity=Severity.HOLD,
            message=(
                f"Sender domain matches vendor {v.vendor_id} but the specific email "
                f"alias is not in the authorized list. Soft trust signal."
            ),
        )
    return None


def rule_missing_po_reference(v, e, m) -> Optional[Reason]:
    if e.intent == "invoice" and not e.po_numbers:
        return Reason(
            code="MISSING_PO_REFERENCE",
            severity=Severity.HOLD,
            message="Invoice does not reference a PO. AP must clarify with vendor.",
        )
    return None


def rule_unknown_po(v, e, m) -> Optional[Reason]:
    for po in m.matched_pos:
        if not po.po_found:
            return Reason(
                code="UNKNOWN_PO",
                severity=Severity.HOLD,
                message=f"PO {po.po_number} referenced in email is not in the ERP.",
            )
    return None


def rule_currency_mismatch(v, e, m) -> Optional[Reason]:
    if not e.currency:
        return None
    for po in m.matched_pos:
        if po.po_found and po.po_currency and po.po_currency != e.currency:
            return Reason(
                code="CURRENCY_MISMATCH",
                severity=Severity.HOLD,
                message=(
                    f"Invoice currency {e.currency} does not match "
                    f"PO {po.po_number} currency {po.po_currency}."
                ),
            )
    return None


def rule_overbilling(v, e, m) -> Optional[Reason]:
    if e.claimed_total is None:
        return None
    if m.aggregate_authorized_total <= 0:
        return None
    if e.claimed_total - m.aggregate_authorized_total > TOLERANCE:
        return Reason(
            code="OVERBILLING",
            severity=Severity.HOLD,
            message=(
                f"Invoice claims {e.claimed_total} but only "
                f"{m.aggregate_authorized_total} is authorized across PO(s)."
            ),
        )
    return None


def rule_underdelivered(v, e, m) -> Optional[Reason]:
    """Invoice asks for payment on goods not yet (fully) received."""
    if e.claimed_total is None:
        return None
    if m.aggregate_authorized_total <= 0:
        return None
    if e.claimed_total - m.aggregate_received_total > TOLERANCE:
        return Reason(
            code="UNDERDELIVERED",
            severity=Severity.HOLD,
            message=(
                f"Invoice claims {e.claimed_total} but only "
                f"{m.aggregate_received_total} worth of goods has been received."
            ),
        )
    return None


def rule_delivery_exception(v, e, m) -> Optional[Reason]:
    for po in m.matched_pos:
        for diff in po.line_diffs:
            if diff.condition in ("DAMAGED", "SHORT"):
                return Reason(
                    code="DELIVERY_EXCEPTION",
                    severity=Severity.HOLD,
                    message=(
                        f"PO {po.po_number} line {diff.po_line_no} ({diff.sku}) "
                        f"received in condition: {diff.condition}."
                    ),
                )
    return None


def rule_failed_extraction(v, e, m) -> Optional[Reason]:
    """Marker rule for when extraction itself failed.

    The extractor sets `intent="unknown"` AND no fields populated when it fails.
    We hold rather than approve under that condition.
    """
    if (
        e.intent == "unknown"
        and not e.po_numbers
        and not e.claimed_total
        and not e.invoice_number
    ):
        return Reason(
            code="FAILED_EXTRACTION",
            severity=Severity.HOLD,
            message="LLM could not extract structured fields from this email.",
        )
    return None


# ---------------------------------------------------------------------------
# The rule table — order matters only for the audit narrative; severity
# hierarchy decides the final action.
# ---------------------------------------------------------------------------

RULE_TABLE: list[RuleFn] = [
    rule_unknown_sender,
    rule_iban_mismatch,
    rule_vendor_po_mismatch,
    rule_bank_change,
    rule_unverified_alias,
    rule_missing_po_reference,
    rule_unknown_po,
    rule_currency_mismatch,
    rule_overbilling,
    rule_underdelivered,
    rule_delivery_exception,
    rule_failed_extraction,
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def decide(
    msg_id: str,
    vendor: Optional[ResolvedVendor],
    extracted: ExtractedEmail,
    match: MatchResult,
) -> Decision:
    """Run every rule, collect every reason, pick the worst severity."""
    fired: list[Reason] = []
    for rule in RULE_TABLE:
        reason = rule(vendor, extracted, match)
        if reason is not None:
            fired.append(reason)

    decision = _pick_decision(fired)

    # APPROVE clean — emit a CLEAN_MATCH reason for symmetry in the audit trail.
    if decision == Severity.APPROVE:
        fired.append(
            Reason(
                code="CLEAN_MATCH",
                severity=Severity.APPROVE,
                message="All checks passed within tolerance.",
            )
        )

    return Decision(
        msg_id=msg_id,
        decision=decision,
        reason_codes=[r.code for r in fired],
        reasons=fired,
        vendor=vendor,
        extracted=extracted,
        match=match,
        audit_summary=_render_summary(decision, fired),
    )


def _pick_decision(fired: list[Reason]) -> Severity:
    """Severity hierarchy: one REJECT wins; else one HOLD wins; else APPROVE."""
    if any(r.severity == Severity.REJECT for r in fired):
        return Severity.REJECT
    if any(r.severity == Severity.HOLD for r in fired):
        return Severity.HOLD
    return Severity.APPROVE


def _render_summary(decision: Severity, fired: list[Reason]) -> str:
    """One-line human summary for the audit report header."""
    if decision == Severity.APPROVE:
        return "All checks passed; safe to authorize payment."
    codes = ", ".join(r.code for r in fired) or "(no reasons)"
    return f"{decision.value} — {codes}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _norm_iban(iban: str) -> str:
    """Normalize IBAN: remove spaces, uppercase. Different sources format differently."""
    return iban.replace(" ", "").upper()
