from __future__ import annotations

import time
from datetime import datetime, timezone

from botocore.exceptions import ClientError
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .client import handle_client_error
from .cloudwatch import iter_log_events, aggregate, get_time_range, parse_since, normalize_model_id
from .display import build_table, total_cost, period_label, since_label
from .pricing import init_pricing, lookup, save_override, OVERRIDES_PATH, get_model_display_name

console = Console()

_POLL_SECONDS    = 5
_LIVE_OVERLAP_MS = 90_000


def _resolve(period: str, since: str | None, live: bool = False) -> tuple[int, int, str]:
    """Return (start_ms, end_ms, label) for either a period or a --since duration."""
    if since:
        start_ms, end_ms = parse_since(since)
        return start_ms, end_ms, since_label(since)
    start_ms, end_ms = get_time_range(period)
    label = period_label(period)
    if live:
        label = f"Since {label}"
    return start_ms, end_ms, label


def _prompt_for_pricing(model_ids: list[str]) -> None:
    """Interactively prompt the user to enter pricing for models not yet in the
    AWS Price List API.  Prices are saved to the user's overrides config file so
    they don't need to be re-entered on subsequent runs.
    """
    if not model_ids:
        return

    console.print(
        "\n[yellow]The following model(s) are missing pricing data "
        "(not yet in the AWS Price List API):[/yellow]"
    )

    for model_id in model_ids:
        console.print(f"\n  [bold cyan]{model_id}[/bold cyan]")

        suggested = get_model_display_name(model_id)
        if suggested != model_id:
            prompt_name = f"  Display name [{suggested}]: "
        else:
            prompt_name = "  Display name (Enter to use model ID): "

        try:
            raw_name = input(prompt_name).strip()
            display_name = raw_name or suggested

            raw_in = input("  Input price per 1M tokens  (USD): ").strip()
            input_per_1m = float(raw_in)

            raw_out = input("  Output price per 1M tokens (USD): ").strip()
            output_per_1m = float(raw_out)

            # Cache pricing — offer standard-ratio defaults so the user can
            # just press Enter for models that follow Anthropic's conventions
            default_cw = round(input_per_1m * 1.25, 4)
            default_cr = round(input_per_1m * 0.10, 4)

            raw_cw = input(
                f"  Cache write price per 1M tokens [{default_cw} (1.25× input), Enter to accept]: "
            ).strip()
            cache_write_per_1m = float(raw_cw) if raw_cw else default_cw

            raw_cr = input(
                f"  Cache read price per 1M tokens  [{default_cr} (0.10× input), Enter to accept]: "
            ).strip()
            cache_read_per_1m = float(raw_cr) if raw_cr else default_cr

        except (ValueError, EOFError, KeyboardInterrupt):
            console.print(f"  [dim]Skipped — {model_id} will show N/A cost.[/dim]")
            continue

        save_override(
            model_id, input_per_1m, output_per_1m,
            cache_write_per_1m, cache_read_per_1m, display_name,
        )
        console.print(f"  [green]Saved to {OVERRIDES_PATH}[/green]")

    console.print()


def _collect_unknown_models(usage: dict[str, dict]) -> list[str]:
    """Return model IDs in usage that have no pricing data."""
    return [mid for mid in usage if lookup(mid).needs_pricing]


def run_once(client, bedrock_client, period: str, threshold: float | None, since: str | None) -> None:
    start_ms, end_ms, label = _resolve(period, since)

    init_pricing(client.meta.region_name, bedrock_client)
    console.print(f"[dim]Fetching {label} from CloudWatch…[/dim]")

    try:
        records = list(iter_log_events(client, start_ms, end_ms))
    except ClientError as exc:
        handle_client_error(exc)
        return

    usage = aggregate(records)

    if not usage:
        console.print(f"\n[yellow]No Bedrock invocations found for {label}.[/yellow]")
        console.print(
            "[dim]If you haven't enabled logging yet, run "
            "[bold]bedrock-lens --setup[/bold].[/dim]"
        )
        return

    unknown = _collect_unknown_models(usage)
    if unknown:
        _prompt_for_pricing(unknown)

    console.print()
    console.print(build_table(usage, title=f"Bedrock Usage — {label}"))
    console.print()

    if threshold is not None:
        cost = total_cost(usage)
        if cost >= threshold:
            console.print(
                f"[bold red]⚠  THRESHOLD EXCEEDED:[/bold red]  "
                f"${cost:.4f} ≥ ${threshold:.2f}"
            )


def run_live(client, bedrock_client, period: str, threshold: float | None, since: str | None) -> None:
    init_pricing(client.meta.region_name, bedrock_client)
    start_ms, _, label = _resolve(period, since, live=True)

    # --since gives a relative label that goes stale as the tool runs.
    # Show the duration + the pinned anchor time so it's always unambiguous.
    if since:
        start_dt = datetime.fromtimestamp(start_ms / 1000)
        if start_dt.date() < datetime.now().date():
            anchor = start_dt.strftime("%b %d, %H:%M")
        else:
            anchor = start_dt.strftime("%H:%M:%S")
        label = f"Last {since} · from {anchor}"

    usage: dict[str, dict] = {}
    seen_ids: set[str] = set()
    threshold_alerted = False

    def ingest(from_ms: int) -> None:
        to_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        try:
            for r in iter_log_events(client, from_ms, to_ms):
                eid = r.get("_eventId", "")
                if eid and eid in seen_ids:
                    continue
                if eid:
                    seen_ids.add(eid)
                model    = normalize_model_id(r.get("modelId", "unknown"))
                inp_data = r.get("input") or {}
                inp = inp_data.get("inputTokenCount")           or 0
                cw  = inp_data.get("cacheWriteInputTokenCount") or 0
                cr  = inp_data.get("cacheReadInputTokenCount")  or 0
                out = (r.get("output") or {}).get("outputTokenCount") or 0
                if model not in usage:
                    usage[model] = {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cache_write_tokens": 0,
                        "cache_read_tokens": 0,
                    }
                usage[model]["calls"]              += 1
                usage[model]["input_tokens"]       += inp
                usage[model]["output_tokens"]      += out
                usage[model]["cache_write_tokens"] += cw
                usage[model]["cache_read_tokens"]  += cr
        except ClientError as exc:
            handle_client_error(exc)

    def render(last_update: str) -> Panel:
        body = build_table(usage) if usage else Text("Waiting for Bedrock invocations…", style="dim italic")
        return Panel(
            body,
            title=f"[bold]Bedrock Live Monitor[/bold]  [dim]—  {label}[/dim]",
            subtitle=(
                f"[dim]Last updated: {last_update}"
                f"  •  refreshing every {_POLL_SECONDS}s"
                f"  •  Ctrl+C to exit[/dim]"
            ),
            border_style="blue",
        )

    console.print("[dim]Loading initial data…[/dim]")
    ingest(start_ms)

    # Prompt for any unknown models before the live display takes over the terminal
    unknown = _collect_unknown_models(usage)
    if unknown:
        _prompt_for_pricing(unknown)

    poll_from_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - _LIVE_OVERLAP_MS

    with Live(render(datetime.now().strftime("%H:%M:%S")), refresh_per_second=2, console=console) as live:
        try:
            while True:
                time.sleep(_POLL_SECONDS)
                ingest(max(start_ms, poll_from_ms))
                poll_from_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - _LIVE_OVERLAP_MS
                live.update(render(datetime.now().strftime("%H:%M:%S")))

                if threshold is not None and not threshold_alerted:
                    cost = total_cost(usage)
                    if cost >= threshold:
                        threshold_alerted = True
                        console.print(
                            f"\n[bold red]⚠  THRESHOLD EXCEEDED:[/bold red]  "
                            f"${cost:.4f} ≥ ${threshold:.2f}\n"
                        )
        except KeyboardInterrupt:
            pass

    console.print("[dim]Exited live mode.[/dim]")
