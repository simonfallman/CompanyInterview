"""Stage [5] — Audit output.

Two artifacts per email, both derived from the same `Decision` object:

  out/<msg_id>.json   machine-readable; the structured audit record.
  out/<msg_id>.md     human-readable; the AP user's actual artifact.

Plus an aggregate `out/summary.csv` over the whole inbox run so reviewers
can see distributions at a glance.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from verifier.schemas import Decision, Severity


def _emoji(decision: Severity) -> str:
    """Format decision as a readable tag for markdown headers."""
    return {
        Severity.APPROVE: "[APPROVE]",
        Severity.HOLD: "[HOLD]",
        Severity.REJECT: "[REJECT]",
    }[decision]


def render_markdown(d: Decision) -> str:
    """Human-readable audit report for one email."""
    lines: list[str] = []

    lines.append(f"# Audit report — {d.msg_id}")
    lines.append("")
    lines.append(f"**Decision: {_emoji(d.decision)} {d.decision.value}**")
    if d.reason_codes:
        lines.append("")
        lines.append("**Reason codes:** " + ", ".join(f"`{c}`" for c in d.reason_codes))
    lines.append("")
    lines.append(f"_{d.audit_summary}_")
    lines.append("")

    # ----- Vendor -----
    lines.append("## Vendor identity")
    if d.vendor:
        lines.append(f"- **Resolved to:** {d.vendor.legal_name} (`{d.vendor.vendor_id}`)")
        lines.append(f"- **Match basis:** `{d.vendor.match_basis}`")
        lines.append(f"- **IBAN on file:** `{d.vendor.iban}`")
        lines.append(f"- **Last verified:** {d.vendor.last_verified_at}")
    else:
        lines.append("- **Sender did not match any vendor in the master file.**")
    lines.append("")

    # ----- Extracted fields -----
    lines.append("## What we extracted from the email")
    if d.extracted:
        e = d.extracted
        lines.append(f"- **Intent:** `{e.intent}`")
        lines.append(f"- **Invoice number:** {e.invoice_number or '_none_'}")
        lines.append(
            f"- **PO numbers:** {', '.join(f'`{po}`' for po in e.po_numbers) or '_none_'}"
        )
        lines.append(
            f"- **Claimed total:** "
            f"{e.claimed_total} {e.currency or ''}".rstrip()
            if e.claimed_total is not None
            else "- **Claimed total:** _none_"
        )
        lines.append(f"- **IBAN in message:** {e.iban_in_message or '_none_'}")
        if e.reasoning:
            lines.append("")
            lines.append("**LLM extraction reasoning:**")
            lines.append("")
            lines.append("> " + e.reasoning.replace("\n", "\n> "))
    lines.append("")

    # ----- Match -----
    if d.match and d.match.matched_pos:
        lines.append("## Three-way match")
        for po in d.match.matched_pos:
            if not po.po_found:
                lines.append(f"### `{po.po_number}` — **not found in ERP**")
                continue
            lines.append(f"### `{po.po_number}` ({po.po_currency}, vendor `{po.po_vendor_id}`)")
            lines.append(
                f"- **Authorized:** {po.authorized_total} {po.po_currency}"
            )
            lines.append(
                f"- **Received:**   {po.received_total} {po.po_currency}"
            )
            if po.line_diffs:
                lines.append("")
                lines.append("| Line | SKU | Ordered | Received | Condition | Unit price |")
                lines.append("|---|---|---|---|---|---|")
                for diff in po.line_diffs:
                    lines.append(
                        f"| {diff.po_line_no} | `{diff.sku}` | "
                        f"{diff.quantity_ordered} | {diff.quantity_received} | "
                        f"`{diff.condition}` | {diff.unit_price} |"
                    )
            lines.append("")
    elif d.match is not None and not d.match.matched_pos:
        lines.append("## Three-way match")
        lines.append("- _No PO references in this email; nothing to match against._")
        lines.append("")

    # ----- Reasons in detail -----
    if d.reasons:
        lines.append("## Reasons in detail")
        for r in d.reasons:
            lines.append(f"- **`{r.code}`** ({r.severity.value}): {r.message}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_decision(decision: Decision, out_dir: Path) -> None:
    """Write one decision's JSON + markdown into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{decision.msg_id}.json").write_text(
        decision.model_dump_json(indent=2), encoding="utf-8"
    )
    (out_dir / f"{decision.msg_id}.md").write_text(
        render_markdown(decision), encoding="utf-8"
    )


def write_summary_csv(decisions: list[Decision], out_dir: Path) -> None:
    """Aggregate one-row-per-email summary so reviewers can scan the run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "msg_id",
                "decision",
                "reason_codes",
                "vendor_id",
                "match_basis",
                "intent",
                "po_numbers",
                "claimed_total",
                "currency",
            ]
        )
        for d in decisions:
            w.writerow(
                [
                    d.msg_id,
                    d.decision.value,
                    ";".join(d.reason_codes),
                    d.vendor.vendor_id if d.vendor else "",
                    d.vendor.match_basis if d.vendor else "",
                    d.extracted.intent if d.extracted else "",
                    ";".join(d.extracted.po_numbers) if d.extracted else "",
                    str(d.extracted.claimed_total) if d.extracted and d.extracted.claimed_total else "",
                    d.extracted.currency if d.extracted and d.extracted.currency else "",
                ]
            )


def write_aggregate_stats(decisions: list[Decision], out_dir: Path) -> None:
    """Append an aggregate stats markdown file alongside the per-email reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for d in decisions:
        counts[d.decision.value] = counts.get(d.decision.value, 0) + 1
        for code in d.reason_codes:
            reason_counts[code] = reason_counts.get(code, 0) + 1

    lines: list[str] = []
    lines.append("# Run summary")
    lines.append("")
    lines.append(f"**Emails processed:** {len(decisions)}")
    lines.append("")
    lines.append("## Decisions")
    for k in ("APPROVE", "HOLD", "REJECT"):
        lines.append(f"- {k}: {counts.get(k, 0)}")
    lines.append("")
    lines.append("## Reason-code distribution")
    for code in sorted(reason_counts, key=lambda c: -reason_counts[c]):
        lines.append(f"- `{code}`: {reason_counts[code]}")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
