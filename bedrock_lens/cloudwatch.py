from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from typing import Generator

from botocore.exceptions import ClientError

LOG_GROUP = "/aws/bedrock/model-invocations"


def get_time_range(period: str) -> tuple[int, int]:
    """Return (start_ms, end_ms) epoch milliseconds for a named period."""
    now = datetime.now(timezone.utc)
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif period == "yesterday":
        y = now - timedelta(days=1)
        start = y.replace(hour=0, minute=0, second=0, microsecond=0)
        end = y.replace(hour=23, minute=59, second=59, microsecond=999000)
    elif period == "week":
        start = now - timedelta(days=7)
        end = now
    else:
        raise ValueError(f"Unknown period: {period!r}")
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


_SINCE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_SINCE_RE    = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])$")


def parse_since(value: str) -> tuple[int, int]:
    """Parse a duration string (e.g. '30m', '2h', '1d') into (start_ms, end_ms)."""
    m = _SINCE_RE.match(value.strip().lower())
    if not m:
        raise ValueError(
            f"Invalid duration {value!r}. "
            "Use a number followed by s, m, h, or d — e.g. 30m, 2h, 1d."
        )
    seconds = float(m.group(1)) * _SINCE_UNITS[m.group(2)]
    now     = datetime.now(timezone.utc)
    start   = now - timedelta(seconds=seconds)
    return int(start.timestamp() * 1000), int(now.timestamp() * 1000)


def iter_log_events(
    client, start_ms: int, end_ms: int
) -> Generator[dict, None, None]:
    """Yield parsed Bedrock ModelInvocationLog records from CloudWatch.

    Each yielded dict is the raw JSON record with an extra '_eventId' key
    added for deduplication in live mode.
    """
    kwargs: dict = {
        "logGroupName": LOG_GROUP,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 10_000,
    }
    while True:
        try:
            resp = client.filter_log_events(**kwargs)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code == "ResourceNotFoundException":
                return  # log group doesn't exist yet
            raise

        for event in resp.get("events", []):
            try:
                record = json.loads(event["message"])
            except (json.JSONDecodeError, KeyError):
                continue
            if record.get("schemaType") != "ModelInvocationLog":
                continue
            record["_eventId"] = event.get("eventId", "")
            record["_ingestionTime"] = event.get("ingestionTime", 0)
            yield record

        token = resp.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token


def normalize_model_id(model_id: str) -> str:
    """Normalize model ID so all variants of the same model merge into one row.

    Strips full ARNs down to the model/profile ID, then removes any
    cross-region geographic prefix (us., eu., ap., …).  The prefix list is
    fetched dynamically from list_inference_profiles() at startup via
    pricing.get_cross_region_prefixes(), so it stays current without code
    changes when AWS adds new geographic regions.
    """
    from .pricing import get_cross_region_prefixes  # late import avoids circular dep

    if model_id.startswith("arn:"):
        model_id = model_id.split("/")[-1]
    for prefix in get_cross_region_prefixes():
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def aggregate(records) -> dict[str, dict]:
    """Sum token counts and call counts keyed by normalized modelId.

    Input tokens are split into three buckets that map to different billing
    rates: regular input, prompt-cache writes, and prompt-cache reads.
    """
    usage: dict[str, dict] = {}
    for r in records:
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
    return usage
