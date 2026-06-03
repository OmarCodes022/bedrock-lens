"""
Pricing strategy (priority order):

1. AmazonBedrockFoundationModels CSV — downloaded at startup via
   list_price_lists + get_price_list_file_url. Covers all Anthropic/Claude
   models. Both Regional (standard on-demand) and Global (cross-region
   inference profile) rates are stored separately; lookup() picks the right
   one based on the model ID prefix.

2. AmazonBedrock get_products — paginated fallback for non-Anthropic providers
   (Meta, Mistral, DeepSeek, Nova, etc.) absent from the CSV. Regional rates
   only; these providers don't support prompt caching on Bedrock.

3. User overrides — ~/.config/bedrock-lens/overrides.json, populated
   interactively when an unknown model is first seen. Auto-removed once the
   Pricing API starts covering that model.

4. Unknown — lookup() returns needs_pricing=True so the caller can prompt.

Also fetched at startup from the Bedrock control plane:
  list_foundation_models()  → human-readable display names
  list_inference_profiles() → live cross-region prefix set (us., eu., global., …)

All prices are per 1M tokens (USD).
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import boto3
from rich.console import Console

_console = Console()


class ModelPricing(NamedTuple):
    """Per-1M-token prices and metadata for a single model."""
    input_per_1m:       float
    output_per_1m:      float
    cache_write_per_1m: float
    cache_read_per_1m:  float
    display_name:       str
    needs_pricing:      bool


OVERRIDES_PATH      = Path.home() / ".config" / "bedrock-lens" / "overrides.json"
_PRICING_CACHE_PATH = Path.home() / ".config" / "bedrock-lens" / "pricing_cache.json"
_PRICING_CACHE_TTL  = 86_400  # 24 hours — pricing changes rarely

# Fallback only — overwritten at startup by list_inference_profiles() if available.
_DEFAULT_CROSS_REGION_PREFIXES: tuple[str, ...] = (
    "us.", "eu.", "ap.", "us-gov.", "global."
)

# normalized_name → (input, output, cache_write, cache_read, display_name)
_live_cache:   dict[str, tuple[float, float, float, float, str]] = {}  # regional rates
_global_cache: dict[str, tuple[float, float, float, float, str]] = {}  # global-profile rates
_live_sorted:   list = []  # _live_cache sorted by key length desc, built at init
_global_sorted: list = []
_overrides:    dict[str, tuple[float, float, float, float, str]] = {}
_model_names:  dict[str, str] = {}
_cross_region_prefixes: tuple[str, ...] = _DEFAULT_CROSS_REGION_PREFIXES

# CSV metric name → slot key. Both CamelCase (Claude ≤4.6) and snake_case (Claude ≥4.7).
_METRIC_SLOTS: dict[str, str] = {
    "InputTokenCount-Units":                    "input_regional",
    "input_tokens_standard-Units":              "input_regional",
    "InputTokenCount_Global-Units":             "input_global",
    "input_tokens_global_standard-Units":       "input_global",
    "OutputTokenCount-Units":                   "output_regional",
    "output_tokens_standard-Units":             "output_regional",
    "OutputTokenCount_Global-Units":            "output_global",
    "output_tokens_global_standard-Units":      "output_global",
    "CacheWriteInputTokenCount-Units":          "cache_write_regional",
    "cache_write_tokens_standard-Units":        "cache_write_regional",
    "CacheWriteInputTokenCount_Global-Units":   "cache_write_global",
    "cache_write_tokens_global_standard-Units": "cache_write_global",
    "CacheReadInputTokenCount-Units":           "cache_read_regional",
    "cache_read_tokens_standard-Units":         "cache_read_regional",
    "CacheReadInputTokenCount_Global-Units":    "cache_read_global",
    "cache_read_tokens_global_standard-Units":  "cache_read_global",
}


def _normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def load_overrides() -> dict[str, tuple[float, float, float, float, str]]:
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
        _console.print(
            f"[yellow]Warning:[/yellow] {OVERRIDES_PATH} is corrupt (invalid JSON) "
            "— price overrides ignored. Delete the file to suppress this message."
        )
        return {}
    except (KeyError, TypeError, ValueError) as exc:
        _console.print(
            f"[yellow]Warning:[/yellow] {OVERRIDES_PATH} has unexpected structure "
            f"({exc}) — price overrides ignored."
        )
        return {}


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


def save_override(
    model_id: str,
    input_per_1m: float,
    output_per_1m: float,
    cache_write_per_1m: float,
    cache_read_per_1m: float,
    display_name: str,
) -> None:
    global _overrides
    _overrides[model_id] = (
        input_per_1m, output_per_1m,
        cache_write_per_1m, cache_read_per_1m,
        display_name,
    )
    _write_overrides()


def cleanup_overrides() -> int:
    """Remove override entries now covered by the live Pricing API."""
    global _overrides
    if not _overrides or not _live_cache:
        return 0
    to_remove = [
        mid for mid in list(_overrides)
        if any(live_key in _normalize(mid) for live_key in _live_sorted)
    ]
    if to_remove:
        for mid in to_remove:
            del _overrides[mid]
        _write_overrides()
    return len(to_remove)


def _fetch_live_csv(region: str) -> tuple[dict, dict]:
    """Fetch prices from the AmazonBedrockFoundationModels CSV for the given region.

    Returns (regional_cache, global_cache). The CSV is per-1M already.
    The 1h TTL cache-write variant is skipped — CloudWatch doesn't record which TTL was used.
    """
    try:
        client = boto3.client("pricing", region_name="us-east-1")

        price_lists = client.list_price_lists(
            ServiceCode="AmazonBedrockFoundationModels",
            EffectiveDate=datetime(2030, 1, 1),  # future date → latest list
            CurrencyCode="USD",
        )
        target_arn = next(
            (pl["PriceListArn"] for pl in price_lists.get("PriceLists", [])
             if pl.get("RegionCode") == region),
            None,
        )
        if not target_arn:
            return {}, {}

        url = client.get_price_list_file_url(PriceListArn=target_arn, FileFormat="csv")["Url"]
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw_content = resp.read().decode("utf-8")

        # First 5 rows are AWS metadata; row 6 is the CSV header.
        data_section = "".join(raw_content.splitlines(keepends=True)[5:])

        raw: dict[str, dict] = {}
        for row in csv.DictReader(io.StringIO(data_section)):
            service_name = row.get("serviceName", "").replace(" (Amazon Bedrock Edition)", "").strip()
            price_str    = row.get("PricePerUnit", "")
            if not service_name or not price_str:
                continue

            # "USE1-MP:USE1_InputTokenCount-Units" → "InputTokenCount-Units"
            metric = row.get("usageType", "").split(":")[-1].split("_", 1)[-1]
            slot   = _METRIC_SLOTS.get(metric)
            if slot is None:
                continue

            try:
                price_f = float(price_str)
            except ValueError:
                continue

            key = _normalize(service_name)
            raw.setdefault(key, {"display": service_name})
            raw[key][slot] = price_f

        regional: dict[str, tuple[float, float, float, float, str]] = {}
        global_:  dict[str, tuple[float, float, float, float, str]] = {}
        for key, data in raw.items():
            name = data["display"]
            inp_r, out_r = data.get("input_regional"), data.get("output_regional")
            inp_g, out_g = data.get("input_global"),   data.get("output_global")
            if inp_r is not None and out_r is not None:
                regional[key] = (inp_r, out_r,
                                 data.get("cache_write_regional", 0.0),
                                 data.get("cache_read_regional",  0.0), name)
            if inp_g is not None and out_g is not None:
                global_[key]  = (inp_g, out_g,
                                 data.get("cache_write_global", 0.0),
                                 data.get("cache_read_global",  0.0), name)
        return regional, global_
    except Exception:
        return {}, {}


def _fetch_live_products(region: str) -> dict[str, tuple[float, float, float, float, str]]:
    """Fallback via get_products(ServiceCode='AmazonBedrock') for non-Anthropic models.

    Returns regional rates only; cache prices are 0 (these providers don't support caching).
    API returns per-1K rates — multiplied by 1000 to normalise to per-1M.
    """
    try:
        client    = boto3.client("pricing", region_name="us-east-1")
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
                    terms = obj["terms"]["OnDemand"]
                    dims  = next(iter(terms.values()))["priceDimensions"]
                    dim   = next(iter(dims.values()))
                    unit  = dim.get("unit", "")
                    price = float(dim["pricePerUnit"]["USD"])
                except (KeyError, StopIteration, ValueError):
                    continue

                if unit in ("1K tokens", "1000 Tokens"):
                    price_per_1m = price * 1000
                elif unit in ("1M tokens", "1000000 Tokens"):
                    price_per_1m = price
                else:
                    continue  # unknown unit — skip rather than guess

                key = _normalize(model_name)
                raw.setdefault(key, {"display": model_name})
                raw[key]["input" if "Input" in inference_type else "output"] = price_per_1m

        return {
            key: (data["input"], data["output"], 0.0, 0.0, data["display"])
            for key, data in raw.items()
            if "input" in data and "output" in data
        }
    except Exception:
        return {}



def _fetch_model_names(bedrock_client) -> dict[str, str]:
    try:
        resp = bedrock_client.list_foundation_models(byInferenceType="ON_DEMAND")
        return {m["modelId"]: m["modelName"] for m in resp.get("modelSummaries", [])}
    except Exception:
        return {}


def _fetch_cross_region_prefixes(bedrock_client) -> tuple[str, ...]:
    """Build prefix set from live inference profiles; falls back to defaults on error."""
    try:
        prefixes: set[str] = set()
        kwargs: dict = {"typeEquals": "SYSTEM_DEFINED", "maxResults": 1000}
        while True:
            resp = bedrock_client.list_inference_profiles(**kwargs)
            for profile in resp.get("inferenceProfileSummaries", []):
                pid = profile.get("inferenceProfileId", "")
                dot = pid.find(".")
                if 1 < dot < 10:  # guard: prefix must be 2–10 chars
                    prefixes.add(pid[: dot + 1])
            if not (token := resp.get("nextToken")):
                break
            kwargs["nextToken"] = token
        return tuple(prefixes) if prefixes else _DEFAULT_CROSS_REGION_PREFIXES
    except Exception:
        return _DEFAULT_CROSS_REGION_PREFIXES


def _load_pricing_cache(region: str) -> tuple[dict, dict] | None:
    """Return (live_cache, global_cache) from disk if fresh, else None."""
    try:
        data   = json.loads(_PRICING_CACHE_PATH.read_text())
        entry  = data[region]
        if time.time() - entry["timestamp"] > _PRICING_CACHE_TTL:
            return None
        load = lambda d: {k: tuple(v) for k, v in d.items()}
        return load(entry["live"]), load(entry["global"])
    except Exception:
        return None


def _save_pricing_cache(region: str, live: dict, global_: dict) -> None:
    try:
        try:
            data = json.loads(_PRICING_CACHE_PATH.read_text())
        except Exception:
            data = {}
        data[region] = {"timestamp": time.time(), "live": live, "global": global_}
        _PRICING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PRICING_CACHE_PATH.write_text(json.dumps(data))
    except Exception:
        pass


def init_pricing(region: str | None, bedrock_client=None) -> None:
    """Populate all pricing caches. Call once at startup before any lookup()."""
    global _live_cache, _global_cache, _live_sorted, _global_sorted, _overrides, _model_names, _cross_region_prefixes

    resolved = region or "us-east-1"
    cached   = _load_pricing_cache(resolved)

    if cached is not None:
        _live_cache, _global_cache = cached
        # model names and prefixes are fast and volatile — always fetch fresh
        if bedrock_client is not None:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_names    = pool.submit(_fetch_model_names, bedrock_client)
                f_prefixes = pool.submit(_fetch_cross_region_prefixes, bedrock_client)
                _model_names           = f_names.result()
                _cross_region_prefixes = f_prefixes.result()
    else:
        # All four calls are IO-bound and independent — run concurrently
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_csv      = pool.submit(_fetch_live_csv, resolved)
            f_products = pool.submit(_fetch_live_products, resolved)
            f_names    = pool.submit(_fetch_model_names, bedrock_client) if bedrock_client else None
            f_prefixes = pool.submit(_fetch_cross_region_prefixes, bedrock_client) if bedrock_client else None

            csv_regional, csv_global = f_csv.result()
            products                 = f_products.result()
            if f_names    is not None: _model_names           = f_names.result()
            if f_prefixes is not None: _cross_region_prefixes = f_prefixes.result()

        _live_cache   = {**products, **csv_regional}
        _global_cache = csv_global
        _save_pricing_cache(resolved, _live_cache, _global_cache)

    _live_sorted   = sorted(_live_cache.items(),   key=lambda x: -len(x[0]))
    _global_sorted = sorted(_global_cache.items(), key=lambda x: -len(x[0]))
    _overrides = load_overrides()
    cleanup_overrides()


def get_cross_region_prefixes() -> tuple[str, ...]:
    return _cross_region_prefixes


def _derive_display_name(model_id: str) -> str:
    """Derive a human-readable name from a model ID when no API name is available.

    "anthropic.claude-sonnet-4-5-20250929-v1:0" → "Claude Sonnet 4.5"
    "meta.llama3-2-90b-instruct-v1:0"           → "Llama3 2 90B Instruct"
    """
    name = re.sub(r'^[^.]+\.', '', model_id)       # strip provider
    name = re.sub(r'[-_]\d{6,}.*$', '', name)       # strip date + trailing
    name = re.sub(r'[-_]v\d+[:\d]*$', '', name)     # strip version suffix

    parts  = name.split('-')
    merged: list[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i].isdigit() and parts[i + 1].isdigit():
            merged.append(f"{parts[i]}.{parts[i + 1]}")
            i += 2
        else:
            merged.append(parts[i].capitalize())
            i += 1
    return ' '.join(merged)


def get_model_display_name(model_id: str) -> str:
    return _model_names.get(model_id) or _derive_display_name(model_id)


def lookup(model_id: str, prefer_global: bool = False) -> ModelPricing:
    """Return per-1M prices for a model ID.

    prefer_global=True  → try global-profile cache first (for global. prefix model IDs)
    needs_pricing=True  → no price found; caller should prompt and call save_override()
    """
    norm    = _normalize(model_id.lower())
    primary  = _global_sorted if prefer_global else _live_sorted
    fallback = _live_sorted   if prefer_global else _global_sorted

    for cache in (primary, fallback):
        for key, (in_p, out_p, cw_p, cr_p, name) in cache:
            if key in norm:
                return ModelPricing(in_p, out_p, cw_p, cr_p, name, False)

    if _overrides:
        for mid, (in_p, out_p, cw_p, cr_p, name) in _overrides.items():
            if _normalize(mid) in norm:
                return ModelPricing(in_p, out_p, cw_p, cr_p, name, False)

    return ModelPricing(0.0, 0.0, 0.0, 0.0, get_model_display_name(model_id), True)


def calculate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
    prefer_global: bool = False,
) -> float:
    p = lookup(model_id, prefer_global=prefer_global)
    return (
        input_tokens         * p.input_per_1m
        + output_tokens      * p.output_per_1m
        + cache_write_tokens * p.cache_write_per_1m
        + cache_read_tokens  * p.cache_read_per_1m
    ) / 1_000_000
