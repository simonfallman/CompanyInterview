"""Pipeline orchestration — wires the five stages together.

Owns no business logic; just flows data through:

  EmailMeta
    -> load_raw_email      (stage [2] loader)
    -> resolve_sender       (stage [1] identity)
    -> extractor.extract    (stage [2] LLM)
    -> match_email          (stage [3] three-way match)
    -> decide               (stage [4] policy)
    => Decision

The pipeline takes the prebuilt `DataContext` (in-memory repos) so processing
each email is a sequence of pure function calls plus one LLM round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from verifier.extractor import LLMExtractor, load_raw_email
from verifier.identity import resolve_sender
from verifier.matcher import match_email
from verifier.policy import decide
from verifier.repositories import DataContext
from verifier.schemas import Decision, EmailMeta


@dataclass
class Pipeline:
    data: DataContext
    extractor: LLMExtractor
    inbox_root: Path

    def process(self, meta: EmailMeta) -> Decision:
        """Process one email through all five stages. Returns a Decision."""
        raw = load_raw_email(meta, self.inbox_root)
        vendor = resolve_sender(meta.sender_email, self.data.vendors)
        extracted = self.extractor.extract(raw)
        match = match_email(extracted, self.data.pos, self.data.receipts)
        return decide(meta.msg_id, vendor, extracted, match)
