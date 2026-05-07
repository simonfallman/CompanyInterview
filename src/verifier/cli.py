"""Command-line entry point.

Usage:
    verifier --data-dir candidate-package/data --out out/
    verifier --workers 10                 # 10× wall-time speedup, same cost

Loads the four CSVs once into in-memory repositories, then walks the inbox
processing each email. Writes per-email JSON + markdown reports plus an
aggregate summary CSV and markdown.

Concurrency: with --workers > 1, emails are processed in a thread pool.
The workload is embarrassingly parallel — read-only data layer, independent
emails, network-I/O-bound LLM call. No locks needed; each worker writes a
distinct out/<msg_id>.json file. Thread-safety properties relied on:
    * DataContext is built before any worker starts; never mutated.
    * OpenAIExtractor (and the OpenAI sync client) is thread-safe.
    * Pipeline.process is a pure function over read-only inputs.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import OpenAI

from verifier.extractor import OpenAIExtractor
from verifier.pipeline import Pipeline
from verifier.repositories import DataContext, load_inbox
from verifier.reporter import write_aggregate_stats, write_decision, write_summary_csv

log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("candidate-package/data"),
    show_default=True,
    help="Directory containing vendor_master_file.csv, erp_*, and inbox/.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("out"),
    show_default=True,
    help="Where to write per-email reports + summary.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N emails (useful for smoke tests).",
)
@click.option(
    "--model",
    type=str,
    default="gpt-4o-mini-2024-07-18",
    show_default=True,
    help="OpenAI model name for extraction.",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Number of concurrent workers. The workload is embarrassingly parallel "
        "(I/O-bound on the LLM call). Try 5–10 to cut wall time ~10×. Larger "
        "values risk OpenAI rate limiting."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Print per-email progress to stdout.",
)
def main(
    data_dir: Path,
    out_dir: Path,
    limit: int | None,
    model: str,
    workers: int,
    verbose: bool,
) -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise click.UsageError(
            "OPENAI_API_KEY not set. Put it in .env or export it before running."
        )

    # ---- One-time setup at startup --------------------------------------
    click.echo(f"Loading data from {data_dir} ...")
    data = DataContext.from_data_dir(data_dir)
    inbox_path = data_dir / "inbox" / "index.csv"
    inbox = load_inbox(inbox_path)
    click.echo(
        f"  vendors:  {len(data.vendors.by_id)}\n"
        f"  POs:      {len(data.pos.by_number)}\n"
        f"  receipts: {sum(len(v) for v in data.receipts.by_po.values())} lines\n"
        f"  inbox:    {len(inbox)} emails"
    )

    extractor = OpenAIExtractor(client=OpenAI(api_key=api_key), model=model)
    pipeline = Pipeline(data=data, extractor=extractor, inbox_root=data_dir / "inbox")

    # ---- Per-email processing -------------------------------------------
    todo = inbox if limit is None else inbox[:limit]
    decisions: list = []
    click.echo(
        f"\nProcessing {len(todo)} emails "
        f"({'sequential' if workers == 1 else f'{workers} workers'}) ...\n"
    )

    if workers == 1:
        # Sequential path — simpler and easier to debug. Output is in inbox order.
        for i, meta in enumerate(todo, 1):
            decision = pipeline.process(meta)
            write_decision(decision, out_dir)
            decisions.append(decision)
            _log_progress(i, len(todo), meta, decision, verbose)
    else:
        # Concurrent path — emails complete in finish order, not inbox order.
        # Each worker writes a distinct out/<msg_id>.json file (no collision).
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(pipeline.process, meta): meta for meta in todo}
            done = 0
            for future in as_completed(futures):
                meta = futures[future]
                done += 1
                try:
                    decision = future.result()
                except Exception as exc:  # noqa: BLE001 — log and continue
                    log.error("processing %s failed: %s", meta.msg_id, exc)
                    continue
                write_decision(decision, out_dir)
                decisions.append(decision)
                _log_progress(done, len(todo), meta, decision, verbose)

    # ---- Aggregate output ------------------------------------------------
    write_summary_csv(decisions, out_dir)
    write_aggregate_stats(decisions, out_dir)

    counts = {"APPROVE": 0, "HOLD": 0, "REJECT": 0}
    for d in decisions:
        counts[d.decision.value] += 1
    click.echo("\n=== Run summary ===")
    click.echo(f"  APPROVE: {counts['APPROVE']}")
    click.echo(f"  HOLD:    {counts['HOLD']}")
    click.echo(f"  REJECT:  {counts['REJECT']}")
    click.echo(f"\nReports written to {out_dir}/")


def _log_progress(i: int, total: int, meta, decision, verbose: bool) -> None:
    """Print progress: email count, ID, kind, language, decision, reason codes."""
    if not (verbose or i % 10 == 0 or i == total):
        return
    click.echo(
        f"  [{i:>3}/{total}] {meta.msg_id} "
        f"({meta.message_kind:>11}, {meta.language})  "
        f"=> {decision.decision.value:<7}  "
        f"{', '.join(decision.reason_codes)}"
    )


if __name__ == "__main__":
    main()
