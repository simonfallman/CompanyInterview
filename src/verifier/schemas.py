"""Typed data models that flow through the pipeline.

The shape of the data is the contract every other module agrees on.
Three groups of models live here:

1. ERP / vendor models   — typed views of the CSV data (Vendor, POLine, Receipt).
2. LLM extraction model  — what the LLM is asked to produce (ExtractedEmail).
3. Decision models       — what the policy engine emits (Decision, Reason, etc.).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# 1. ERP / vendor models — typed views of the CSV rows
# ---------------------------------------------------------------------------

Currency = Literal["SEK", "EUR", "PLN"]
POStatus = Literal["open", "partially_received", "closed"]
ReceiptCondition = Literal["OK", "DAMAGED", "SHORT"]


class Vendor(BaseModel):
    """One vendor record from the master file, representing a trusted supplier.

    The authorization fields control which senders are accepted for this vendor.
    `authorized_sender_emails` and `authorized_sender_domains` are the explicit
    allow-list; `primary_contact_email` is implicitly authorized even when absent
    from the alias list (see `VendorRepository._index`).
    """

    vendor_id: str
    legal_name: str
    country: str
    address: str
    primary_contact_name: str
    primary_contact_email: str
    authorized_sender_domains: list[str]
    authorized_sender_emails: list[str]
    iban: str
    bic: str
    payment_terms: Literal["NET30", "NET45"]
    default_currency: Currency
    tax_id: str
    last_verified_at: date


class POLine(BaseModel):
    """One line item on a purchase order, as recorded in the ERP system.

    Each PO may have multiple lines (different SKUs). During three-way match,
    `quantity_ordered`, `unit_price`, and `line_total` are compared against
    receipt data and the vendor's claimed invoice total.
    """

    po_number: str
    vendor_id: str
    issued_at: date
    expected_delivery_at: date
    currency: Currency
    po_total_amount: Decimal
    status: POStatus
    line_no: int
    sku: str
    description: str
    quantity_ordered: int
    unit_price: Decimal
    line_total: Decimal


class Receipt(BaseModel):
    """One goods-receipt line confirming physical delivery against a PO line.

    `condition` drives hold/reject logic: DAMAGED or SHORT quantities reduce the
    authorized payment amount in the three-way match.
    """

    receipt_id: str
    po_number: str
    received_at: date
    received_by: str
    receipt_line_no: int
    po_line_no: int
    sku: str
    quantity_received: int
    condition: ReceiptCondition


class EmailMeta(BaseModel):
    """Envelope metadata for one inbox message — header fields only, no body.

    The body and attachments are loaded separately into `RawEmail`. `body_path`
    and `attachment_paths` are relative to the inbox directory; the loader
    resolves them to absolute paths before reading.
    """

    msg_id: str
    thread_id: str
    in_reply_to: Optional[str]
    sent_at: str  # ISO-8601 with TZ — keep as str; we don't reason on it numerically
    sender_name: str
    sender_email: str
    subject: str
    language: str
    message_kind: Literal["invoice", "followup", "bank_change", "mixed"]
    body_path: str
    attachment_paths: list[str]


class RawEmail(BaseModel):
    """Fully-loaded email ready for LLM extraction: metadata plus all text content.

    This is the boundary object passed into stage [2]. After this point the LLM
    sees no raw filesystem paths — only the text that was resolved from them.
    """

    meta: EmailMeta
    body: str
    attachments: dict[str, str] = Field(default_factory=dict)  # path -> content


# ---------------------------------------------------------------------------
# 2. LLM extraction model — what the LLM is asked to produce
# ---------------------------------------------------------------------------

EmailIntent = Literal["invoice", "bank_change", "followup", "mixed", "unknown"]


class ExtractedEmail(BaseModel):
    """Structured data extracted from one email by the LLM.

    This is the output of stage [2] — the LLM's only job in the pipeline.
    Once an email has become an ExtractedEmail, the LLM is done.
    Every downstream step is deterministic Python.
    """

    # OpenAI structured outputs is strict about schema features; keep it simple.
    model_config = ConfigDict(extra="forbid")

    invoice_number: Optional[str] = Field(
        default=None,
        description="The invoice number quoted in the email or attachment, if any.",
    )
    po_numbers: list[str] = Field(
        default_factory=list,
        description="All PO numbers referenced in the email or attachments.",
    )
    claimed_total: Optional[Decimal] = Field(
        default=None,
        description="The amount the email is requesting payment for, in the currency below.",
    )
    currency: Optional[Currency] = Field(
        default=None,
        description="The currency of the claimed total. SEK, EUR, or PLN.",
    )
    iban_in_message: Optional[str] = Field(
        default=None,
        description=(
            "Any IBAN explicitly stated in the email or attachment. "
            "Null if no IBAN is mentioned. Do NOT infer from the vendor's known IBAN."
        ),
    )
    intent: EmailIntent = Field(
        default="unknown",
        description=(
            "Classified intent of the email. invoice = requesting payment for goods; "
            "bank_change = requesting an update to bank/IBAN details; "
            "followup = referencing a prior email; mixed = multiple intents in one message; "
            "unknown = could not classify."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "One short paragraph explaining how the fields above were derived from the email text. "
            "Cite the specific phrases that informed each field. This becomes part of the audit trail."
        ),
    )


# ---------------------------------------------------------------------------
# 3. Decision models — what the policy engine emits
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Ordered outcome levels for a payment decision.

    REJECT — hard stop, do not pay. HOLD — flag for human review. APPROVE — pay.
    When multiple reasons fire, the pipeline escalates to the highest severity.
    """

    REJECT = "REJECT"
    HOLD = "HOLD"
    APPROVE = "APPROVE"


class Reason(BaseModel):
    """One fired policy rule: a machine-readable code, its severity, and an optional message.

    Multiple Reasons can appear on a single Decision; the highest `severity` wins.
    `message` is human-readable elaboration for the AP reviewer — not required for
    automated processing.
    """

    code: str
    severity: Severity
    message: str = ""  # human-readable elaboration, optional


class ResolvedVendor(BaseModel):
    """Vendor identity confirmed at stage [1] — who we believe sent the email.

    `match_basis` records how the sender was matched: `exact_email` means the
    sender address appears verbatim in the authorized list; `domain` means only
    the domain matched; `primary_contact` means the sender is the primary contact
    address that was implicitly authorized.
    """

    vendor_id: str
    legal_name: str
    iban: str
    default_currency: Currency
    last_verified_at: date
    match_basis: Literal["exact_email", "domain", "primary_contact"]


class LineDiff(BaseModel):
    """Per-PO-line comparison: ordered vs received vs (implicitly) invoiced."""

    po_number: str
    po_line_no: int
    sku: str
    quantity_ordered: int
    quantity_received: int
    condition: ReceiptCondition
    unit_price: Decimal
    line_total: Decimal


class POMatch(BaseModel):
    """The result of matching one referenced PO to ERP data."""

    po_number: str
    po_found: bool
    po_vendor_id: Optional[str] = None
    po_currency: Optional[Currency] = None
    authorized_total: Decimal = Decimal("0")
    received_total: Decimal = Decimal("0")
    line_diffs: list[LineDiff] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Output of stage [3] — the three-way match across all referenced POs."""

    matched_pos: list[POMatch] = Field(default_factory=list)
    aggregate_authorized_total: Decimal = Decimal("0")
    aggregate_received_total: Decimal = Decimal("0")
    currencies_seen: set[Currency] = Field(default_factory=set)


class Decision(BaseModel):
    """The final output of stage [4] — what AP sees."""

    msg_id: str
    decision: Severity
    reason_codes: list[str]
    reasons: list[Reason]
    vendor: Optional[ResolvedVendor] = None
    extracted: Optional[ExtractedEmail] = None
    match: Optional[MatchResult] = None
    audit_summary: str = ""
