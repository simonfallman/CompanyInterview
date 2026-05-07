"""Stage [3] — Three-way matcher.

Given the PO numbers extracted from an email, look up:
  • what was authorized (PO lines)
  • what actually arrived (receipt lines, joined back via po_line_no)

Compute per-line diffs and per-PO totals. Pure deterministic Python; no LLM.

The output is a `MatchResult` containing one `POMatch` per referenced PO. The
policy engine reads this plus the `ExtractedEmail` and emits the decision.
"""

from __future__ import annotations

from decimal import Decimal

from verifier.repositories import PORepository, ReceiptRepository
from verifier.schemas import (
    ExtractedEmail,
    LineDiff,
    MatchResult,
    POMatch,
    Receipt,
    ReceiptCondition,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def match_email(
    extracted: ExtractedEmail,
    pos: PORepository,
    receipts: ReceiptRepository,
) -> MatchResult:
    """Run the three-way match for all POs referenced in the extracted email.

    For each PO number in `extracted.po_numbers`, look up authorized PO lines and
    goods-receipt records, compute per-line diffs and totals, and aggregate across
    all POs. Returns a `MatchResult` whose `matched_pos` list preserves the order
    of `extracted.po_numbers`; a PO absent from the ERP still gets a `POMatch`
    entry with `po_found=False` so downstream policy rules can fire on it.
    """
    result = MatchResult()
    aggregate_authorized = Decimal("0")
    aggregate_received = Decimal("0")

    for po_number in extracted.po_numbers:
        match = _match_one_po(po_number, pos, receipts)
        result.matched_pos.append(match)
        aggregate_authorized += match.authorized_total
        aggregate_received += match.received_total
        if match.po_currency:
            result.currencies_seen.add(match.po_currency)

    result.aggregate_authorized_total = aggregate_authorized
    result.aggregate_received_total = aggregate_received
    return result


# ---------------------------------------------------------------------------
# Single-PO matching
# ---------------------------------------------------------------------------


def _match_one_po(
    po_number: str, pos: PORepository, receipts: ReceiptRepository
) -> POMatch:
    """Produce a `POMatch` for a single PO number.

    Looks up the PO's authorized lines and all goods-receipt records, joins them
    on `po_line_no`, and builds one `LineDiff` per PO line. `authorized_total` is
    the sum of `line_total` values from the PO; `received_total` is computed as
    qty_received × unit_price per line. Returns a stub `POMatch(po_found=False)`
    when the PO does not exist in the ERP.
    """
    lines = pos.get(po_number)

    if not lines:
        # PO referenced in the email does not exist in the ERP.
        return POMatch(po_number=po_number, po_found=False)

    # Group this PO's receipts by po_line_no so we can sum partial deliveries.
    po_receipts = receipts.get(po_number)
    receipts_by_line: dict[int, list[Receipt]] = {}
    for r in po_receipts:
        receipts_by_line.setdefault(r.po_line_no, []).append(r)

    line_diffs: list[LineDiff] = []
    received_total = Decimal("0")

    for line in lines:
        line_receipts = receipts_by_line.get(line.line_no, [])
        qty_received = sum(r.quantity_received for r in line_receipts)
        condition = _worst_condition(line_receipts)

        line_diffs.append(
            LineDiff(
                po_number=po_number,
                po_line_no=line.line_no,
                sku=line.sku,
                quantity_ordered=line.quantity_ordered,
                quantity_received=qty_received,
                condition=condition,
                unit_price=line.unit_price,
                line_total=line.line_total,
            )
        )
        # received_total = qty_received × unit_price for each line, summed
        received_total += Decimal(qty_received) * line.unit_price

    authorized_total = sum((line.line_total for line in lines), Decimal("0"))

    return POMatch(
        po_number=po_number,
        po_found=True,
        po_vendor_id=lines[0].vendor_id,
        po_currency=lines[0].currency,
        authorized_total=authorized_total,
        received_total=received_total,
        line_diffs=line_diffs,
    )


def _worst_condition(line_receipts: list[Receipt]) -> ReceiptCondition:
    """Severity hierarchy for receipt conditions: DAMAGED > SHORT > OK.

    No receipts yet returns OK (missing qty caught elsewhere).
    """
    if not line_receipts:
        return "OK"
    if any(r.condition == "DAMAGED" for r in line_receipts):
        return "DAMAGED"
    if any(r.condition == "SHORT" for r in line_receipts):
        return "SHORT"
    return "OK"
