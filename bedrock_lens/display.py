from __future__ import annotations

from datetime import datetime, timedelta

from rich import box
from rich.table import Table

from .pricing import lookup, calculate_cost


def period_label(period: str) -> str:
    now = datetime.now()
    if period == "today":
        return f"Today ({now.strftime('%Y-%m-%d')})"
    if period == "yesterday":
        y = now - timedelta(days=1)
        return f"Yesterday ({y.strftime('%Y-%m-%d')})"
    if period == "week":
        return "Past 7 Days"
    return period


def since_label(value: str) -> str:
    unit_names = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    v = value.strip().lower()
    unit = unit_names.get(v[-1], v[-1])
    amount = v[:-1].rstrip()
    amount = amount.rstrip("0").rstrip(".") if "." in amount else amount
    return f"Last {amount} {unit}"


def _fmt_tokens(n: int) -> str:
    """Format a token count.  Use compact K/M suffixes for values ≥ 10,000
    so that large cache token counts don't overflow the table columns.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def _fmt_cost(cost: float, known: bool) -> str:
    if not known:
        return "[dim]N/A[/dim]"
    return f"${cost:.4f}"


def build_table(usage: dict[str, dict], title: str = "") -> Table:
    """Build a Rich Table summarising token usage and estimated cost.

    Cache Write and Cache Read columns are shown only when at least one model
    in the current window has cache activity, keeping the table compact for
    workloads that don't use prompt caching.
    """
    has_cache = any(
        s.get("cache_write_tokens", 0) + s.get("cache_read_tokens", 0) > 0
        for s in usage.values()
    )

    table = Table(
        title=title,
        box=box.ROUNDED,
        show_footer=True,
        expand=False,
        title_style="bold",
    )
    table.add_column("Model",         style="cyan",         no_wrap=True,    footer="TOTAL")
    table.add_column("Calls",         style="bright_white", justify="right", footer_style="bold")
    table.add_column("Input Tokens",  style="green",        justify="right", footer_style="bold green")
    if has_cache:
        table.add_column("Cache Write", style="blue",       justify="right", footer_style="bold blue")
        table.add_column("Cache Read",  style="cyan",       justify="right", footer_style="bold cyan")
    table.add_column("Output Tokens", style="yellow",       justify="right", footer_style="bold yellow")
    table.add_column("Est. Cost",     style="magenta",      justify="right", footer_style="bold magenta")

    total_calls = total_input = total_output = 0
    total_cache_write = total_cache_read = 0
    total_cost_val = 0.0
    all_prices_known = True

    for model_id, stats in sorted(usage.items()):
        inp   = stats["input_tokens"]
        out   = stats["output_tokens"]
        cw    = stats.get("cache_write_tokens", 0)
        cr    = stats.get("cache_read_tokens",  0)
        calls = stats["calls"]

        p     = lookup(model_id)
        known = not p.needs_pricing
        cost  = calculate_cost(model_id, inp, out, cw, cr)

        total_calls       += calls
        total_input       += inp
        total_output      += out
        total_cache_write += cw
        total_cache_read  += cr
        total_cost_val    += cost
        if not known:
            all_prices_known = False

        row = [p.display_name, str(calls), _fmt_tokens(inp)]
        if has_cache:
            row += [_fmt_tokens(cw), _fmt_tokens(cr)]
        row += [_fmt_tokens(out), _fmt_cost(cost, known)]
        table.add_row(*row)

    # Footer column index depends on whether cache columns are present
    col = 1
    table.columns[col].footer = str(total_calls);       col += 1
    table.columns[col].footer = _fmt_tokens(total_input); col += 1
    if has_cache:
        table.columns[col].footer = _fmt_tokens(total_cache_write); col += 1
        table.columns[col].footer = _fmt_tokens(total_cache_read);  col += 1
    table.columns[col].footer = _fmt_tokens(total_output); col += 1
    table.columns[col].footer = _fmt_cost(total_cost_val, all_prices_known)

    return table


def total_cost(usage: dict[str, dict]) -> float:
    return sum(
        calculate_cost(
            mid,
            s["input_tokens"],
            s["output_tokens"],
            s.get("cache_write_tokens", 0),
            s.get("cache_read_tokens",  0),
        )
        for mid, s in usage.items()
    )
