"""Stage [1] — Identity Resolver.

Take an email's sender address, resolve it to a known vendor (or not).
Pure deterministic Python; no LLM. The vendor master file is the gold
source of truth for who we are allowed to pay.

Output is a typed `ResolvedVendor` (or `None` if the sender does not
match any vendor at all). The downstream policy engine treats `None`
as a hard reject — `UNKNOWN_SENDER`.
"""

from __future__ import annotations

from typing import Optional

from verifier.repositories import VendorRepository
from verifier.schemas import ResolvedVendor


def resolve_sender(
    sender_email: str, vendors: VendorRepository
) -> Optional[ResolvedVendor]:
    """Resolve a sender to a vendor, or None if unknown.

    Match precedence (handled in the repository):
      1. exact match against `authorized_sender_emails`  -> match_basis="exact_email"
      2. domain match against `authorized_sender_domains` -> match_basis="domain"
      3. neither                                          -> None (UNKNOWN_SENDER)

    A `domain`-only match is a *softer* trust signal than `exact_email`:
    the sender is plausibly the vendor, but using an alias the AP team has
    not seen before. Downstream rules can flag that for review.
    """
    vendor, basis = vendors.find_by_sender(sender_email)
    if vendor is None or basis is None:
        return None

    return ResolvedVendor(
        vendor_id=vendor.vendor_id,
        legal_name=vendor.legal_name,
        iban=vendor.iban,
        default_currency=vendor.default_currency,
        last_verified_at=vendor.last_verified_at,
        match_basis=basis,  # type: ignore[arg-type]
    )
