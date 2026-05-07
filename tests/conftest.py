"""Shared pytest fixtures for the verifier test suite.

Tests use these fixtures so they don't have to rebuild repositories
or re-load CSVs in every test. Each fixture is small and obvious;
production data only loads when an explicit fixture asks for it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from verifier.repositories import VendorRepository
from verifier.schemas import Vendor


# ---------------------------------------------------------------------------
# Hand-built tiny vendor repository — fast, deterministic, no I/O.
# Use this for unit tests of the identity resolver and the policy engine.
# ---------------------------------------------------------------------------


@pytest.fixture
def vendor_nordic() -> Vendor:
    return Vendor(
        vendor_id="V-007",
        legal_name="Nordic Bearings AB",
        country="SE",
        address="Industrigatan 14, 411 04 Göteborg, Sweden",
        primary_contact_name="Astrid Lindqvist",
        primary_contact_email="astrid.lindqvist@nordic-bearings.se",
        authorized_sender_domains=["nordic-bearings.se"],
        authorized_sender_emails=[
            "faktura@nordic-bearings.se",
            "ekonomi@nordic-bearings.se",
        ],
        iban="SE45 5000 0000 0583 9825 7466",
        bic="ESSESESS",
        payment_terms="NET30",
        default_currency="SEK",
        tax_id="SE556677889901",
        last_verified_at=date(2026, 2, 14),
    )


@pytest.fixture
def vendor_german() -> Vendor:
    return Vendor(
        vendor_id="V-099",
        legal_name="Berlin Werkzeuge GmbH",
        country="DE",
        address="Hauptstr. 1, 10115 Berlin",
        primary_contact_name="Klaus Schmidt",
        primary_contact_email="k.schmidt@berlin-werkzeuge.de",
        authorized_sender_domains=["berlin-werkzeuge.de"],
        authorized_sender_emails=["billing@berlin-werkzeuge.de"],
        iban="DE89 3704 0044 0532 0130 00",
        bic="COBADEFF",
        payment_terms="NET30",
        default_currency="EUR",
        tax_id="DE123456789",
        last_verified_at=date(2026, 1, 10),
    )


@pytest.fixture
def vendors(vendor_nordic: Vendor, vendor_german: Vendor) -> VendorRepository:
    repo = VendorRepository()
    repo._index(vendor_nordic)
    repo._index(vendor_german)
    return repo


# ---------------------------------------------------------------------------
# Real candidate-package data — opt in by depending on this fixture.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    """Path to the candidate-package data, if present locally."""
    return Path(__file__).resolve().parents[1] / "candidate-package" / "data"
