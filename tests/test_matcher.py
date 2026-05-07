"""Stage [3] tests — three-way match arithmetic.

The matcher is the financial core; coverage is heavy here on purpose.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from verifier.matcher import match_email
from verifier.repositories import PORepository, ReceiptRepository
from verifier.schemas import ExtractedEmail, POLine, Receipt


# ---------------------------------------------------------------------------
# fixtures — small synthetic POs and receipts
# ---------------------------------------------------------------------------


def _po_line(line_no: int, sku: str, qty: int, unit_price: str) -> POLine:
    line_total = Decimal(unit_price) * qty
    return POLine(
        po_number="PO-2026-0001",
        vendor_id="V-007",
        issued_at=date(2026, 3, 1),
        expected_delivery_at=date(2026, 3, 14),
        currency="SEK",
        po_total_amount=Decimal("0"),  # repeated header — not used by matcher
        status="closed",
        line_no=line_no,
        sku=sku,
        description=f"item {sku}",
        quantity_ordered=qty,
        unit_price=Decimal(unit_price),
        line_total=line_total,
    )


def _receipt(line_no: int, sku: str, qty: int, condition: str = "OK") -> Receipt:
    return Receipt(
        receipt_id=f"GR-2026-0001",
        po_number="PO-2026-0001",
        received_at=date(2026, 3, 14),
        received_by="Mikael Berg",
        receipt_line_no=line_no,
        po_line_no=line_no,
        sku=sku,
        quantity_received=qty,
        condition=condition,
    )


@pytest.fixture
def pos_and_receipts():
    """Build a tiny repo pair with a single 4-line PO, fully received OK."""
    po_lines = [
        _po_line(1, "BRG-001", 24, "165.00"),  # 3960
        _po_line(2, "BRG-002", 10, "200.00"),  # 2000
        _po_line(3, "BRG-003", 5, "500.00"),   # 2500
        _po_line(4, "BRG-004", 2, "1000.00"),  # 2000
    ]
    pos = PORepository()
    for line in po_lines:
        pos.by_number.setdefault(line.po_number, []).append(line)

    receipts = ReceiptRepository()
    for r in [
        _receipt(1, "BRG-001", 24),
        _receipt(2, "BRG-002", 10),
        _receipt(3, "BRG-003", 5),
        _receipt(4, "BRG-004", 2),
    ]:
        receipts.by_po.setdefault(r.po_number, []).append(r)

    return pos, receipts


def _extract(po_numbers: list[str]) -> ExtractedEmail:
    return ExtractedEmail(po_numbers=po_numbers, intent="invoice")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_clean_match_single_po(pos_and_receipts):
    pos, receipts = pos_and_receipts
    result = match_email(_extract(["PO-2026-0001"]), pos, receipts)

    assert len(result.matched_pos) == 1
    match = result.matched_pos[0]
    assert match.po_found is True
    assert match.authorized_total == Decimal("10460.00")
    assert match.received_total == Decimal("10460.00")
    assert match.po_currency == "SEK"
    assert match.po_vendor_id == "V-007"
    # All lines fully received OK
    assert all(d.quantity_ordered == d.quantity_received for d in match.line_diffs)
    assert all(d.condition == "OK" for d in match.line_diffs)


def test_partial_receipt_flagged_per_line(pos_and_receipts):
    """One line received short of ordered quantity -> qty_received < qty_ordered."""
    pos, receipts = pos_and_receipts
    # Mutate: replace line 3's full receipt with a partial one
    receipts.by_po["PO-2026-0001"] = [
        _receipt(1, "BRG-001", 24),
        _receipt(2, "BRG-002", 10),
        _receipt(3, "BRG-003", 3),  # only 3 of 5
        _receipt(4, "BRG-004", 2),
    ]

    result = match_email(_extract(["PO-2026-0001"]), pos, receipts)
    match = result.matched_pos[0]
    line3 = next(d for d in match.line_diffs if d.po_line_no == 3)

    assert line3.quantity_ordered == 5
    assert line3.quantity_received == 3
    assert match.received_total == Decimal("9460.00")  # 10460 - 2*500


def test_damaged_condition_propagates(pos_and_receipts):
    pos, receipts = pos_and_receipts
    receipts.by_po["PO-2026-0001"][0] = _receipt(1, "BRG-001", 24, condition="DAMAGED")

    result = match_email(_extract(["PO-2026-0001"]), pos, receipts)
    line1 = result.matched_pos[0].line_diffs[0]

    assert line1.condition == "DAMAGED"


def test_short_condition_propagates(pos_and_receipts):
    pos, receipts = pos_and_receipts
    receipts.by_po["PO-2026-0001"][1] = _receipt(2, "BRG-002", 8, condition="SHORT")

    result = match_email(_extract(["PO-2026-0001"]), pos, receipts)
    line2 = next(d for d in result.matched_pos[0].line_diffs if d.po_line_no == 2)

    assert line2.condition == "SHORT"
    assert line2.quantity_received == 8


def test_multiple_receipts_for_one_line_are_summed(pos_and_receipts):
    """A PO line received in two batches: total qty is the sum, condition is the worst."""
    pos, receipts = pos_and_receipts
    receipts.by_po["PO-2026-0001"] = [
        _receipt(1, "BRG-001", 10),  # batch one
        _receipt(1, "BRG-001", 14, condition="DAMAGED"),  # batch two
        _receipt(2, "BRG-002", 10),
        _receipt(3, "BRG-003", 5),
        _receipt(4, "BRG-004", 2),
    ]

    result = match_email(_extract(["PO-2026-0001"]), pos, receipts)
    line1 = result.matched_pos[0].line_diffs[0]

    assert line1.quantity_received == 24  # 10 + 14
    assert line1.condition == "DAMAGED"  # worst-of


def test_unknown_po_returned_with_po_found_false(pos_and_receipts):
    pos, receipts = pos_and_receipts
    result = match_email(_extract(["PO-DOES-NOT-EXIST"]), pos, receipts)

    assert len(result.matched_pos) == 1
    assert result.matched_pos[0].po_found is False
    assert result.aggregate_authorized_total == Decimal("0")


def test_empty_po_numbers_returns_empty_result(pos_and_receipts):
    """Email with no PO referenced -> empty match result, no error."""
    pos, receipts = pos_and_receipts
    result = match_email(_extract([]), pos, receipts)

    assert result.matched_pos == []
    assert result.aggregate_authorized_total == Decimal("0")
    assert result.currencies_seen == set()


def test_received_total_excludes_unreceived_lines(pos_and_receipts):
    """A line with no receipts contributes 0 to received_total but still appears in diffs."""
    pos, receipts = pos_and_receipts
    receipts.by_po["PO-2026-0001"] = [
        _receipt(1, "BRG-001", 24),
        _receipt(2, "BRG-002", 10),
        # lines 3 and 4 not yet received
    ]

    result = match_email(_extract(["PO-2026-0001"]), pos, receipts)
    match = result.matched_pos[0]

    assert match.authorized_total == Decimal("10460.00")
    assert match.received_total == Decimal("5960.00")  # 24*165 + 10*200
    assert len(match.line_diffs) == 4  # all four PO lines still represented
    assert match.line_diffs[2].quantity_received == 0
    assert match.line_diffs[3].quantity_received == 0


def test_aggregate_totals_sum_across_multiple_pos():
    """Multi-PO email: aggregate totals sum the per-PO totals."""
    pos = PORepository()
    receipts = ReceiptRepository()

    # PO-A: one line, 10 units @ 100 = 1000
    pos.by_number["PO-A"] = [
        POLine(
            po_number="PO-A", vendor_id="V-007", issued_at=date(2026, 3, 1),
            expected_delivery_at=date(2026, 3, 14), currency="SEK",
            po_total_amount=Decimal("1000"), status="closed", line_no=1,
            sku="X", description="x", quantity_ordered=10,
            unit_price=Decimal("100"), line_total=Decimal("1000"),
        )
    ]
    receipts.by_po["PO-A"] = [
        Receipt(receipt_id="GR-A", po_number="PO-A", received_at=date(2026, 3, 14),
                received_by="m", receipt_line_no=1, po_line_no=1, sku="X",
                quantity_received=10, condition="OK")
    ]
    # PO-B: one line, 5 units @ 200 = 1000
    pos.by_number["PO-B"] = [
        POLine(
            po_number="PO-B", vendor_id="V-007", issued_at=date(2026, 3, 1),
            expected_delivery_at=date(2026, 3, 14), currency="SEK",
            po_total_amount=Decimal("1000"), status="closed", line_no=1,
            sku="Y", description="y", quantity_ordered=5,
            unit_price=Decimal("200"), line_total=Decimal("1000"),
        )
    ]
    receipts.by_po["PO-B"] = [
        Receipt(receipt_id="GR-B", po_number="PO-B", received_at=date(2026, 3, 14),
                received_by="m", receipt_line_no=1, po_line_no=1, sku="Y",
                quantity_received=5, condition="OK")
    ]

    result = match_email(_extract(["PO-A", "PO-B"]), pos, receipts)

    assert len(result.matched_pos) == 2
    assert result.aggregate_authorized_total == Decimal("2000")
    assert result.aggregate_received_total == Decimal("2000")


def test_currencies_seen_collects_distinct_currencies():
    """Multi-PO email where two POs are denominated in different currencies."""
    pos = PORepository()
    pos.by_number["PO-SEK"] = [
        POLine(
            po_number="PO-SEK", vendor_id="V-007", issued_at=date(2026, 3, 1),
            expected_delivery_at=date(2026, 3, 14), currency="SEK",
            po_total_amount=Decimal("100"), status="closed", line_no=1,
            sku="X", description="x", quantity_ordered=1,
            unit_price=Decimal("100"), line_total=Decimal("100"),
        )
    ]
    pos.by_number["PO-EUR"] = [
        POLine(
            po_number="PO-EUR", vendor_id="V-099", issued_at=date(2026, 3, 1),
            expected_delivery_at=date(2026, 3, 14), currency="EUR",
            po_total_amount=Decimal("10"), status="closed", line_no=1,
            sku="Y", description="y", quantity_ordered=1,
            unit_price=Decimal("10"), line_total=Decimal("10"),
        )
    ]

    receipts = ReceiptRepository()  # empty — irrelevant for this test
    result = match_email(_extract(["PO-SEK", "PO-EUR"]), pos, receipts)

    assert result.currencies_seen == {"SEK", "EUR"}
