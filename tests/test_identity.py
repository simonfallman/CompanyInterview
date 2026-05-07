"""Stage [1] tests — sender → vendor resolution.

Each test name reads as a business rule.
"""

from __future__ import annotations

from verifier.identity import resolve_sender
from verifier.repositories import VendorRepository


def test_resolve_by_exact_authorized_email(vendors: VendorRepository) -> None:
    resolved = resolve_sender("faktura@nordic-bearings.se", vendors)

    assert resolved is not None
    assert resolved.vendor_id == "V-007"
    assert resolved.match_basis == "exact_email"


def test_resolve_by_primary_contact_email(vendors: VendorRepository) -> None:
    """Primary contact email is also accepted, even if not duplicated in the alias list."""
    resolved = resolve_sender("astrid.lindqvist@nordic-bearings.se", vendors)

    assert resolved is not None
    assert resolved.vendor_id == "V-007"
    assert resolved.match_basis == "exact_email"


def test_resolve_by_authorized_domain_when_email_alias_unknown(
    vendors: VendorRepository,
) -> None:
    """Sender uses the right domain but a new alias the AP team has not seen.

    This is a softer trust signal than an exact email match.
    """
    resolved = resolve_sender("new.alias@nordic-bearings.se", vendors)

    assert resolved is not None
    assert resolved.vendor_id == "V-007"
    assert resolved.match_basis == "domain"


def test_resolve_unknown_sender_returns_none(vendors: VendorRepository) -> None:
    """No domain match -> no vendor. Downstream this becomes UNKNOWN_SENDER."""
    resolved = resolve_sender("vendor.finance.team@gmail.com", vendors)

    assert resolved is None


def test_resolve_is_case_insensitive(vendors: VendorRepository) -> None:
    """Real email systems are case-insensitive; we should be too."""
    resolved = resolve_sender("FAKTURA@Nordic-Bearings.SE", vendors)

    assert resolved is not None
    assert resolved.vendor_id == "V-007"


def test_resolve_typo_squat_does_not_match(vendors: VendorRepository) -> None:
    """`lndustri-...` (L instead of I) must not resolve to a similarly-named real vendor.

    This is the classic supplier-impersonation pattern; the resolver has to reject it.
    """
    resolved = resolve_sender("finance@nordic-beerings.se", vendors)

    assert resolved is None


def test_resolve_malformed_email_returns_none(vendors: VendorRepository) -> None:
    """Defensive: garbage input does not crash, just fails to resolve."""
    assert resolve_sender("", vendors) is None
    assert resolve_sender("not-an-email", vendors) is None


def test_different_vendors_are_isolated(
    vendors: VendorRepository,
) -> None:
    """Two vendors in the repo do not bleed into each other's resolution."""
    swedish = resolve_sender("faktura@nordic-bearings.se", vendors)
    german = resolve_sender("billing@berlin-werkzeuge.de", vendors)

    assert swedish is not None and swedish.vendor_id == "V-007"
    assert german is not None and german.vendor_id == "V-099"
