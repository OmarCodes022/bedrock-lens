"""
Pricing strategy (priority order):

1. AmazonBedrockFoundationModels CSV  — downloaded at startup via
   list_price_lists + get_price_list_file_url.  Covers all Anthropic/Claude
   models including the latest releases (often weeks ahead of the older
   AmazonBedrock service code).  Prices are the standard on-demand (Global)
   rate — not the geo-CRIS regional premium.

2. AmazonBedrock get_products  — paginated fallback covering non-Anthropic
   providers (Meta, Mistral, DeepSeek, Amazon Nova, Google, Nvidia, etc.)
   that are absent from the CSV.  The two sources are merged at startup;
   the CSV wins on any overlap.

3. User config file overrides — ~/.config/bedrock-lens/overrides.json
   Populated interactively when an unknown model is first seen.  Entries are
   automatically removed once the Pricing API starts covering that model.

4. Unknown — lookup() returns needs_pricing=True so runner.py can prompt
   the user.

Supplementary data fetched from the Bedrock control-plane at startup:
  - list_foundation_models()  → {modelId: displayName} for human-readable names
  - list_inference_profiles() → dynamic cross-region prefix set (us., eu., ap., …)
    consumed by cloudwatch.normalize_model_id via get_cross_region_prefixes()

All prices are per 1 million tokens (USD).
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import boto3

# ---------------------------------------------------------------------------
# Public return type for lookup()
# ---------------------------------------------------------------------------

class ModelPricing(NamedTuple):
    """All per-1M-token prices and metadata for a single model."""
    input_per_1m:       float   # standard input tokens
    output_per_1m:      float
    cache_write_per_1m: float   # prompt cache write (standard TTL)
    cache_read_per_1m:  float   # prompt cache read
    display_name:       str
    needs_pricing:      bool    # True → no price found, caller should prompt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERRIDES_PATH = Path.home() / ".config" / "bedrock-lens" / "overrides.json"

# Fallback used if list_inference_profiles() fails
_DEFAULT_CROSS_REGION_PREFIXES: tuple[str, ...] = (
    "us.", "eu.", "ap.", "us-gov.", "global."
)

# ---------------------------------------------------------------------------
# Module-level caches (populated by init_pricing)
# ---------------------------------------------------------------------------

# normalized_name → (input, output, cache_write, cache_read, display_name)
_live_cache: dict[str, tuple[float, float, float, float, str]] = {}

# model_id → (input, output, cache_write, cache_read, display_name)
_overrides: dict[str, tuple[float, float, float, float, str]] = {}

# modelId → human-readable name  (from list_foundation_models)
_model_names: dict[str, str] = {}

# Populated from list_inference_profiles; falls back to _DEFAULT_CROSS_REGION_PREFIXES
_cross_region_prefixes: tuple[str, ...] = _DEFAULT_CROSS_REGION_PREFIXES


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Strip everything except lowercase letters and digits for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', s.lower())


# ---------------------------------------------------------------------------
# Config-file override helpers
# ---------------------------------------------------------------------------

def load_overrides() -> dict[str, tuple[float, float, float, float, str]]:
    """Read ~/.config/bedrock-lens/overrides.json; return empty dict if missing/corrupt."""
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        data = json.loads(OVERRIDES_PATH.read_text())
        return {
            model_id: (
                float(entry["input_per_1m"]),
                float(entry["output_per_1m"]),
                float(entry.get("cache_write_per_1m", 0.0)),
                float(entry.get("cache_read_per_1m",  0.0)),
                str(entry["display_name"]),
            )
            for model_id, entry in data.items()
        }
    except json.JSONDecodeError:
        from rich.console import Console
        Console().print(
            f"[yellow]Warning:[/yellow] {OVERRIDES_PATH} is corrupt (invalid JSON) "
            "— price overrides ignored.  Delete the file to suppress this message."
        )
        return {}
    except (KeyError, TypeError, ValueError) as exc:
        from rich.console import Console
        Console().print(
            f"[yellow]Warning:[/yellow] {OVERRIDES_PATH} has unexpected structure "
            f"({exc}) — price overrides ignored."
        )
        return {}


def save_override(
    model_id: str,
    input_per_1m: float,
    output_per_1m: float,
    cache_write_per_1m: float,
    cache_read_per_1m: float,
    display_name: str,
) -> None:
    """Persist a single override entry, updating the in-memory cache and the file."""
    global _overrides
    _overrides[model_id] = (
        input_per_1m, output_per_1m,
        cache_write_per_1m, cache_read_per_1m,
        display_name,
    )
    _write_overrides()


def _write_overrides() -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        model_id: {
            "input_per_1m":       in_p,
            "output_per_1m":      out_p,
            "cache_write_per_1m": cw_p,
            "cache_read_per_1m":  cr_p,
            "display_name":       name,
        }
        for model_id, (in_p, out_p, cw_p, cr_p, name) in _overrides.items()
    }
    OVERRIDES_PATH.write_text(json.dumps(data, indent=2))


def cleanup_overrides() -> int:
    """Remove override entries now covered by the live Pricing API.

    Compares each override's normalised key against the live cache using the
    same substring logic as lookup(). Returns the number of entries removed.
    """
    global _overrides
    if not _overrides or not _live_cache:
        return 0

    to_remove = [
        model_id
        for model_id in list(_overrides)
        if any(live_key in _normalize(model_id) for live_key in _live_cache)
    ]

    if to_remove:
        for model_id in to_remove:
            del _overrides[model_id]
        _write_overrides()

    return len(to_remove)


# ---------------------------------------------------------------------------
# Live API fetchers
# ---------------------------------------------------------------------------

def _fetch_live_csv(region: str) -> dict[str, tuple[float, float, float, float, str]]:
    """Fetch on-demand Bedrock prices from the AmazonBedrockFoundationModels
    price-list CSV.

    This service code covers all Anthropic/Claude models including the latest
    releases, which are often absent from the older AmazonBedrock get_products
    path.  Discovery approach (via https://github.com/boto/boto3/issues/4531):

      1. list_price_lists with a future EffectiveDate → latest available list
      2. get_price_list_file_url → public CDN URL, no auth required
      3. Download and parse the CSV directly

    For each token type we prefer the Global (on-demand) rate over the
    Regional (geo-CRIS, ~10% premium) rate.  AWS uses two naming conventions:

      CamelCase (Claude ≤4.6)                 snake_case (Claude ≥4.7)
      ───────────────────────────────────────  ──────────────────────────────────────
      InputTokenCount-Units                    input_tokens_standard-Units
      InputTokenCount_Global-Units             input_tokens_global_standard-Units
      OutputTokenCount-Units                   output_tokens_standard-Units
      OutputTokenCount_Global-Units            output_tokens_global_standard-Units
      CacheWriteInputTokenCount-Units          cache_write_tokens_standard-Units
      CacheWriteInputTokenCount_Global-Units   cache_write_tokens_global_standard-Units
      CacheReadInputTokenCount-Units           cache_read_tokens_standard-Units
      CacheReadInputTokenCount_Global-Units    cache_read_tokens_global_standard-Units

    Prices in the CSV are already per 1M tokens — no conversion needed.
    The 1-hour TTL cache-write variants (CacheWrite1hInputTokenCount) are
    intentionally skipped: CloudWatch logs do not record which TTL was used.
    """
    try:
        client = boto3.client("pricing", region_name="us-east-1")

        # A future EffectiveDate causes AWS to return the latest available list
        price_lists = client.list_price_lists(
            ServiceCode="AmazonBedrockFoundationModels",
            EffectiveDate=datetime(2030, 1, 1),
            CurrencyCode="USD",
        )

        target_arn: str | None = None
        for pl in price_lists.get("PriceLists", []):
            if pl.get("RegionCode") == region:
                target_arn = pl["PriceListArn"]
                break

        if not target_arn:
            return {}

        url = client.get_price_list_file_url(
            PriceListArn=target_arn,
            FileFormat="csv",
        )["Url"]

        with urllib.request.urlopen(url, timeout=15) as resp:
            raw_content = resp.read().decode("utf-8")

        # The CSV has 5 metadata rows before the column-header row.
        # Strip them so csv.DictReader sees the real header first.
        lines = raw_content.splitlines(keepends=True)
        data_section = "".join(lines[5:])  # row 6 is header, rows 7+ are data

        # Metric → slot mapping (prefer Global over Regional for each type)
        _INPUT_REGIONAL       = {"InputTokenCount-Units",                   "input_tokens_standard-Units"}
        _INPUT_GLOBAL         = {"InputTokenCount_Global-Units",             "input_tokens_global_standard-Units"}
        _OUTPUT_REGIONAL      = {"OutputTokenCount-Units",                   "output_tokens_standard-Units"}
        _OUTPUT_GLOBAL        = {"OutputTokenCount_Global-Units",            "output_tokens_global_standard-Units"}
        _CACHE_WRITE_REGIONAL = {"CacheWriteInputTokenCount-Units",          "cache_write_tokens_standard-Units"}
        _CACHE_WRITE_GLOBAL   = {"CacheWriteInputTokenCount_Global-Units",   "cache_write_tokens_global_standard-Units"}
        _CACHE_READ_REGIONAL  = {"CacheReadInputTokenCount-Units",           "cache_read_tokens_standard-Units"}
        _CACHE_READ_GLOBAL    = {"CacheReadInputTokenCount_Global-Units",    "cache_read_tokens_global_standard-Units"}

        raw: dict[str, dict] = {}
        for row in csv.DictReader(io.StringIO(data_section)):
            usage_type   = row.get("usageType", "")
            service_name = row.get("serviceName", "").replace(" (Amazon Bedrock Edition)", "").strip()
            price_str    = row.get("PricePerUnit", "")

            if not service_name or not price_str:
                continue

            # Extract the metric segment after the region prefix, e.g.:
            #   "USE1-MP:USE1_InputTokenCount-Units" → "InputTokenCount-Units"
            colon_part = usage_type.split(":")[-1]
            metric     = colon_part.split("_", 1)[-1]

            if metric in _INPUT_REGIONAL:
                slot = "input_regional"
            elif metric in _INPUT_GLOBAL:
                slot = "input_global"
            elif metric in _OUTPUT_REGIONAL:
                slot = "output_regional"
            elif metric in _OUTPUT_GLOBAL:
                slot = "output_global"
            elif metric in _CACHE_WRITE_REGIONAL:
                slot = "cache_write_regional"
            elif metric in _CACHE_WRITE_GLOBAL:
                slot = "cache_write_global"
            elif metric in _CACHE_READ_REGIONAL:
                slot = "cache_read_regional"
            elif metric in _CACHE_READ_GLOBAL:
                slot = "cache_read_global"
            else:
                continue

            try:
                price_f = float(price_str)
            except ValueError:
                continue

            key = _normalize(service_name)
            raw.setdefault(key, {"display": service_name})
            raw[key][slot] = price_f

        result: dict[str, tuple[float, float, float, float, str]] = {}
        for key, data in raw.items():
            inp  = data.get("input_global",       data.get("input_regional"))
            out  = data.get("output_global",      data.get("output_regional"))
            if inp is None or out is None:
                continue
            cw = data.get("cache_write_global",  data.get("cache_write_regional", 0.0))
            cr = data.get("cache_read_global",   data.get("cache_read_regional",  0.0))
            result[key] = (inp, out, cw, cr, data["display"])
        return result
    except Exception:
        return {}


def _fetch_live_products(region: str) -> dict[str, tuple[float, float, float, float, str]]:
    """Fallback: fetch prices via get_products(ServiceCode='AmazonBedrock').

    Less complete than the CSV approach — newer models are often missing —
    but kept as a fallback covering non-Anthropic providers (Meta, Mistral,
    DeepSeek, Amazon Nova, etc.) that are absent from the CSV.
    Prices from the API are per 1K tokens; we multiply by 1000 → per 1M.
    These providers do not support prompt caching on Bedrock, so cache rates
    are set to 0.
    """
    try:
        client = boto3.client("pricing", region_name="us-east-1")
        paginator = client.get_paginator("get_products")

        raw: dict[str, dict] = {}
        for page in paginator.paginate(
            ServiceCode="AmazonBedrock",
            Filters=[{"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}],
        ):
            for p in page["PriceList"]:
                obj  = json.loads(p)
                attr = obj["product"]["attributes"]

                model_name     = attr.get("model", "")
                inference_type = attr.get("inferenceType", "")
                if not model_name or inference_type not in ("Input tokens", "Output tokens"):
                    continue

                try:
                    dim = list(
                        list(obj["terms"]["OnDemand"].values())[0][
                            "priceDimensions"
                        ].values()
                    )[0]
                    unit         = dim.get("unit", "")
                    price_raw    = float(dim["pricePerUnit"]["USD"])
                except (KeyError, IndexError, ValueError):
                    continue

                # The Pricing API returns per-1K-token rates for Bedrock models,
                # but guard against future unit changes to avoid silent 1000x errors.
                if unit in ("1K tokens", "1000 Tokens"):
                    price_per_1m = price_raw * 1000
                elif unit in ("1M tokens", "1000000 Tokens"):
                    price_per_1m = price_raw          # already per 1M
                else:
                    continue  # unknown unit — skip rather than guess

                key = _normalize(model_name)
                raw.setdefault(key, {"display": model_name})
                if "Input" in inference_type:
                    raw[key]["input"] = price_per_1m
                else:
                    raw[key]["output"] = price_per_1m

        return {
            key: (data["input"], data["output"], 0.0, 0.0, data["display"])
            for key, data in raw.items()
            if "input" in data and "output" in data
        }
    except Exception:
        return {}


def _fetch_live(region: str) -> dict[str, tuple[float, float, float, float, str]]:
    """Fetch on-demand Bedrock token prices by merging both pricing sources.

    The two service codes cover complementary model sets:
      AmazonBedrockFoundationModels CSV  →  Anthropic/Claude, Cohere, AI21, legacy
      AmazonBedrock get_products         →  Meta, Mistral, DeepSeek, Google,
                                            Amazon Nova, Nvidia, Qwen, etc.

    Both are fetched and merged; the CSV wins on any overlap (it carries the
    correct on-demand Global rate and cache pricing for models it covers).
    """
    csv_results  = _fetch_live_csv(region)
    prod_results = _fetch_live_products(region)

    # Start with products, then overwrite/extend with CSV (CSV wins on overlap)
    return {**prod_results, **csv_results}


def _fetch_model_names(bedrock_client) -> dict[str, str]:
    """Return {modelId: modelName} for every on-demand foundation model in the region."""
    try:
        resp = bedrock_client.list_foundation_models(byInferenceType="ON_DEMAND")
        return {
            m["modelId"]: m["modelName"]
            for m in resp.get("modelSummaries", [])
        }
    except Exception:
        return {}


def _fetch_cross_region_prefixes(bedrock_client) -> tuple[str, ...]:
    """Derive geographic prefix strings (e.g. 'us.', 'eu.') from live inference profiles.

    Falls back to _DEFAULT_CROSS_REGION_PREFIXES if the API call fails or returns
    no results.
    """
    try:
        prefixes: set[str] = set()
        kwargs: dict = {"typeEquals": "SYSTEM_DEFINED", "maxResults": 1000}
        while True:
            resp = bedrock_client.list_inference_profiles(**kwargs)
            for profile in resp.get("inferenceProfileSummaries", []):
                pid = profile.get("inferenceProfileId", "")
                dot = pid.find(".")
                # Guard: prefix must be 2–10 chars (e.g. "us.", "us-gov.")
                if 1 < dot < 10:
                    prefixes.add(pid[: dot + 1])
            next_token = resp.get("nextToken")
            if not next_token:
                break
            kwargs["nextToken"] = next_token

        return tuple(prefixes) if prefixes else _DEFAULT_CROSS_REGION_PREFIXES
    except Exception:
        return _DEFAULT_CROSS_REGION_PREFIXES


# ---------------------------------------------------------------------------
# Public initialiser
# ---------------------------------------------------------------------------

def init_pricing(region: str | None, bedrock_client=None) -> None:
    """Populate all pricing caches.  Call once at startup before any lookup().

    bedrock_client — a boto3 bedrock client for the user's region.  When
    provided, model display names and cross-region prefixes are refreshed from
    the live API.  If None, the previous values (or module defaults) are kept.
    """
    global _live_cache, _overrides, _model_names, _cross_region_prefixes

    resolved = region or "us-east-1"

    if bedrock_client is not None:
        _model_names           = _fetch_model_names(bedrock_client)
        _cross_region_prefixes = _fetch_cross_region_prefixes(bedrock_client)

    _live_cache = _fetch_live(resolved)
    _overrides  = load_overrides()
    cleanup_overrides()


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_cross_region_prefixes() -> tuple[str, ...]:
    """Return the current cross-region inference prefix tuple."""
    return _cross_region_prefixes


def _derive_display_name(model_id: str) -> str:
    """Derive a human-readable name from a model ID string when no API name exists.

    e.g. "anthropic.claude-sonnet-4-5-20250929-v1:0" → "Claude Sonnet 4.5"
         "anthropic.claude-opus-4-6-v1"              → "Claude Opus 4.6"
         "meta.llama3-2-90b-instruct-v1:0"           → "Llama3 2 90B Instruct"
    """
    # Strip provider prefix (everything up to and including first dot)
    name = re.sub(r'^[^.]+\.', '', model_id)
    # Strip date stamp and everything after (e.g. -20250929-v1:0)
    name = re.sub(r'[-_]\d{6,}.*$', '', name)
    # Strip trailing version suffix if still present (e.g. -v1, -v2:0)
    name = re.sub(r'[-_]v\d+[:\d]*$', '', name)

    # Split on hyphens; merge consecutive digit tokens with a dot (4-5 → 4.5)
    parts = name.split('-')
    merged: list[str] = []
    i = 0
    while i < len(parts):
        if (i + 1 < len(parts)
                and parts[i].isdigit()
                and parts[i + 1].isdigit()):
            merged.append(f"{parts[i]}.{parts[i + 1]}")
            i += 2
        else:
            merged.append(parts[i].capitalize())
            i += 1

    return ' '.join(merged)


def get_model_display_name(model_id: str) -> str:
    """Return a display name for a model: API-provided first, then derived from
    the model ID, so callers always get something readable instead of a raw ID.
    """
    return _model_names.get(model_id) or _derive_display_name(model_id)


def lookup(model_id: str) -> ModelPricing:
    """Return a ModelPricing with per-1M prices and metadata for a model ID.

    needs_pricing=True  → no price found; caller should prompt the user and
                          call save_override() with the result.
    needs_pricing=False → a price was found (Pricing API or user override).
    """
    lower = model_id.lower()
    norm  = _normalize(lower)

    # Tier 1: live AWS Pricing API (longest key first to avoid substring collisions)
    if _live_cache:
        for key, (in_p, out_p, cw_p, cr_p, name) in sorted(
            _live_cache.items(), key=lambda x: -len(x[0])
        ):
            if key in norm:
                return ModelPricing(in_p, out_p, cw_p, cr_p, name, False)

    # Tier 2: user config-file overrides
    if _overrides:
        for mid, (in_p, out_p, cw_p, cr_p, name) in _overrides.items():
            if _normalize(mid) in norm or norm in _normalize(mid):
                return ModelPricing(in_p, out_p, cw_p, cr_p, name, False)

    # Unknown — derive a readable display name (API name → derived → raw ID)
    return ModelPricing(0.0, 0.0, 0.0, 0.0, get_model_display_name(model_id), True)


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    p = lookup(model_id)
    return (
        input_tokens       * p.input_per_1m
        + output_tokens    * p.output_per_1m
        + cache_write_tokens * p.cache_write_per_1m
        + cache_read_tokens  * p.cache_read_per_1m
    ) / 1_000_000
