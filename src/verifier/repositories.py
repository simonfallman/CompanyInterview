"""In-memory repositories built once at startup; queried O(1) per email.

The interviewer's pushback: do not "call the database" for every email.
The fix: load the four CSVs once, build typed indexes, expose lookup methods.
The pipeline talks to these repositories — never to the filesystem during processing.

This is the textbook **Repository pattern**: an abstraction over data access
so the rest of the code does not know whether the backing store is a CSV,
a real database, or a Redis snapshot. Today it is CSV; tomorrow we swap the
`from_csv` constructors for `from_postgres` without touching the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from verifier.schemas import EmailMeta, POLine, Receipt, Vendor


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

_SEMI_SPLIT = re.compile(r"\s*;\s*")


def _split_semis(value: str) -> list[str]:
    """`'a@x.se;b@x.se'` -> `['a@x.se', 'b@x.se']`. Empty/NaN -> `[]`."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [piece for piece in _SEMI_SPLIT.split(s) if piece]


def _domain(email: str) -> str:
    """`'billing@nordic-bearings.se'` -> `'nordic-bearings.se'`."""
    return email.split("@", 1)[1].lower() if "@" in email else ""


# ---------------------------------------------------------------------------
# Vendor repository
# ---------------------------------------------------------------------------


@dataclass
class VendorRepository:
    """Lookup vendors by id, by exact authorized email, or by authorized domain."""

    by_id: dict[str, Vendor] = field(default_factory=dict)
    by_authorized_email: dict[str, Vendor] = field(default_factory=dict)
    by_authorized_domain: dict[str, Vendor] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: Path) -> "VendorRepository":
        df = pd.read_csv(path, dtype=str).fillna("")
        repo = cls()
        for row in df.to_dict(orient="records"):
            vendor = Vendor(
                vendor_id=row["vendor_id"],
                legal_name=row["legal_name"],
                country=row["country"],
                address=row["address"],
                primary_contact_name=row["primary_contact_name"],
                primary_contact_email=row["primary_contact_email"],
                authorized_sender_domains=_split_semis(row["authorized_sender_domains"]),
                authorized_sender_emails=_split_semis(row["authorized_sender_emails"]),
                iban=row["iban"],
                bic=row["bic"],
                payment_terms=row["payment_terms"],
                default_currency=row["default_currency"],
                tax_id=row["tax_id"],
                last_verified_at=row["last_verified_at"],
            )
            repo._index(vendor)
        return repo

    def _index(self, vendor: Vendor) -> None:
        self.by_id[vendor.vendor_id] = vendor
        for email in vendor.authorized_sender_emails:
            self.by_authorized_email[email.lower()] = vendor
        # Also accept the primary contact email even if not duplicated in the alias list.
        if vendor.primary_contact_email:
            self.by_authorized_email.setdefault(
                vendor.primary_contact_email.lower(), vendor
            )
        for domain in vendor.authorized_sender_domains:
            self.by_authorized_domain[domain.lower()] = vendor

    # ---- query methods ------------------------------------------------------

    def find_by_sender(
        self, sender_email: str
    ) -> tuple[Optional[Vendor], Optional[str]]:
        """Resolve a sender to a Vendor. Returns (vendor, match_basis) or (None, None).

        Match precedence:
          1. exact authorized email
          2. authorized domain (alias not in known list)
        """
        if not sender_email:
            return None, None
        normalized = sender_email.strip().lower()
        if normalized in self.by_authorized_email:
            return self.by_authorized_email[normalized], "exact_email"
        domain = _domain(normalized)
        if domain and domain in self.by_authorized_domain:
            return self.by_authorized_domain[domain], "domain"
        return None, None

    def get(self, vendor_id: str) -> Optional[Vendor]:
        return self.by_id.get(vendor_id)


# ---------------------------------------------------------------------------
# PO repository
# ---------------------------------------------------------------------------


@dataclass
class PORepository:
    """Lookup PO lines by PO number."""

    by_number: dict[str, list[POLine]] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: Path) -> "PORepository":
        # Read every column as string; let Pydantic coerce. Avoids float
        # round-tripping for money columns — critical for finance correctness.
        df = pd.read_csv(path, dtype=str).fillna("")
        repo = cls()
        for row in df.to_dict(orient="records"):
            line = POLine(
                po_number=row["po_number"],
                vendor_id=row["vendor_id"],
                issued_at=row["issued_at"],
                expected_delivery_at=row["expected_delivery_at"],
                currency=row["currency"],
                po_total_amount=row["po_total_amount"],
                status=row["status"],
                line_no=row["line_no"],
                sku=row["sku"],
                description=row["description"],
                quantity_ordered=row["quantity_ordered"],
                unit_price=row["unit_price"],
                line_total=row["line_total"],
            )
            repo.by_number.setdefault(line.po_number, []).append(line)
        # keep each PO's lines in line_no order so reports read sensibly
        for lines in repo.by_number.values():
            lines.sort(key=lambda l: l.line_no)
        return repo

    def get(self, po_number: str) -> list[POLine]:
        return self.by_number.get(po_number, [])

    def exists(self, po_number: str) -> bool:
        return po_number in self.by_number


# ---------------------------------------------------------------------------
# Receipt repository
# ---------------------------------------------------------------------------


@dataclass
class ReceiptRepository:
    """Lookup receipt lines by PO number (one PO can have many receipts/lines)."""

    by_po: dict[str, list[Receipt]] = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: Path) -> "ReceiptRepository":
        df = pd.read_csv(path, dtype=str).fillna("")
        repo = cls()
        for row in df.to_dict(orient="records"):
            r = Receipt(
                receipt_id=row["receipt_id"],
                po_number=row["po_number"],
                received_at=row["received_at"],
                received_by=row["received_by"],
                receipt_line_no=row["receipt_line_no"],
                po_line_no=row["po_line_no"],
                sku=row["sku"],
                quantity_received=row["quantity_received"],
                condition=row["condition"],
            )
            repo.by_po.setdefault(r.po_number, []).append(r)
        return repo

    def get(self, po_number: str) -> list[Receipt]:
        return self.by_po.get(po_number, [])


# ---------------------------------------------------------------------------
# Inbox loader (not strictly a repository — emails are processed once each)
# ---------------------------------------------------------------------------


def load_inbox(index_csv: Path) -> list[EmailMeta]:
    """Read inbox/index.csv into a list of EmailMeta objects."""
    df = pd.read_csv(index_csv, dtype=str).fillna("")
    emails: list[EmailMeta] = []
    for row in df.to_dict(orient="records"):
        emails.append(
            EmailMeta(
                msg_id=row["msg_id"],
                thread_id=row["thread_id"],
                in_reply_to=row["in_reply_to"] or None,
                sent_at=row["sent_at"],
                sender_name=row["sender_name"],
                sender_email=row["sender_email"],
                subject=row["subject"],
                language=row["language"],
                message_kind=row["message_kind"],
                body_path=row["body_path"],
                attachment_paths=_split_semis(row["attachment_paths"]),
            )
        )
    return emails


# ---------------------------------------------------------------------------
# Convenience: load all four surfaces at once
# ---------------------------------------------------------------------------


@dataclass
class DataContext:
    """Bundle of all the repositories the pipeline needs."""

    vendors: VendorRepository
    pos: PORepository
    receipts: ReceiptRepository

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "DataContext":
        return cls(
            vendors=VendorRepository.from_csv(data_dir / "vendor_master_file.csv"),
            pos=PORepository.from_csv(data_dir / "erp_purchase_orders.csv"),
            receipts=ReceiptRepository.from_csv(data_dir / "erp_receipt_logs.csv"),
        )
