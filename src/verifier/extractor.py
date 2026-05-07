"""Stage [2] — LLM-driven content extraction.

This is the *only* stage where the LLM is on the path. Its job: turn a raw
email (body + attachments) into an `ExtractedEmail` Pydantic object. Once
that conversion is done, the LLM never runs again for this email — every
downstream step is deterministic Python.

We use OpenAI's **structured outputs** feature: we hand the API the Pydantic
schema and the model is *guaranteed* to return JSON conforming to it. No
fragile string parsing, no JSON-cleanup hacks.

If extraction fails (rare with structured outputs, but possible — refusal,
network blip, schema-incompatible content), we retry once. After two
failures, we return an empty `ExtractedEmail` and let the policy engine's
`FAILED_EXTRACTION` rule HOLD it. We never fabricate fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from openai import OpenAI

from verifier.schemas import EmailMeta, ExtractedEmail, RawEmail

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loader — reads body & attachments off disk
# ---------------------------------------------------------------------------


def load_raw_email(meta: EmailMeta, inbox_root: Path) -> RawEmail:
    """Read body + attachments from disk and bundle them into a RawEmail.

    `inbox_root` is the directory containing `index.csv`; body and attachment
    paths in the CSV are relative to it.
    """
    body_text = (inbox_root / meta.body_path).read_text(encoding="utf-8")
    attachments: dict[str, str] = {}
    for path in meta.attachment_paths:
        full = inbox_root / path
        if full.exists():
            attachments[path] = full.read_text(encoding="utf-8")
    return RawEmail(meta=meta, body=body_text, attachments=attachments)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are an extraction engine for an Accounts Payable verification system. \
Your only job is to read one vendor email (body + any attachments) and return the structured \
fields defined by the schema. Follow these rules strictly:

- Return PO numbers exactly as they appear (e.g. "PO-2026-0142"). Include every PO referenced anywhere in the email or attachments.
- For monetary totals, return ONLY the amount the email is requesting payment for (the invoice grand total). Do not invent a total if none is given. Use a decimal number (e.g. 36975.00, not "SEK 36,975").
- Currency must be exactly one of: SEK, EUR, PLN. If the email does not state a currency clearly, return null.
- For `iban_in_message`: return an IBAN ONLY if the email body or an attachment explicitly states one. Do not infer from context. Strip spaces. If no IBAN is mentioned, return null.
- For `intent`: classify what the email is asking for. invoice = requesting payment for goods received; bank_change = requesting an update to bank/IBAN details; followup = referring to a prior email without a new request; mixed = combines multiple intents in one message; unknown = cannot classify.
- For `reasoning`: one short paragraph (2-4 sentences). Cite the specific phrases that informed each non-null field. This is the audit trail; be concrete.

Be conservative. If a field is genuinely ambiguous, return null. Do NOT hallucinate values from prior knowledge of the vendor or the company."""


def build_user_message(raw: RawEmail) -> str:
    """Compose the user message: metadata + body + attachments."""
    parts: list[str] = []
    parts.append(f"Subject: {raw.meta.subject}")
    parts.append(f"From: {raw.meta.sender_name} <{raw.meta.sender_email}>")
    parts.append(f"Sent: {raw.meta.sent_at}")
    parts.append(f"Language: {raw.meta.language}")
    parts.append("")
    parts.append("---- BODY ----")
    parts.append(raw.body.strip())
    if raw.attachments:
        for path, content in raw.attachments.items():
            parts.append("")
            parts.append(f"---- ATTACHMENT: {path} ----")
            parts.append(content.strip())
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM client abstraction (so tests can inject a fake)
# ---------------------------------------------------------------------------


class LLMExtractor(Protocol):
    """Anything that can turn a RawEmail into an ExtractedEmail."""

    def extract(self, raw: RawEmail) -> ExtractedEmail: ...


@dataclass
class OpenAIExtractor:
    """Real OpenAI-backed extractor, using structured outputs."""

    client: OpenAI
    model: str = "gpt-4o-mini-2024-07-18"
    max_retries: int = 1  # i.e. 2 attempts total

    def extract(self, raw: RawEmail) -> ExtractedEmail:
        """Extract structured fields from raw email via OpenAI structured outputs."""
        user_message = build_user_message(raw)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=ExtractedEmail,
                    temperature=0,  # deterministic-ish output for the same input
                )
                parsed = resp.choices[0].message.parsed
                if parsed is not None:
                    return parsed
                # Refusal or empty parse — retry path
                last_error = RuntimeError("structured output returned None")
            except Exception as e:  # network blip, schema validation, refusal
                last_error = e
                log.warning(
                    "extraction attempt %d failed for %s: %s",
                    attempt + 1,
                    raw.meta.msg_id,
                    e,
                )

        # All attempts exhausted — return an empty ExtractedEmail.
        # The policy engine's FAILED_EXTRACTION rule will HOLD this email.
        log.error(
            "extraction permanently failed for %s after %d attempts: %s",
            raw.meta.msg_id,
            self.max_retries + 1,
            last_error,
        )
        return ExtractedEmail()
