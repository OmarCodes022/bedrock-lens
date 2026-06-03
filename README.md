# bedrock-lens

Real-time token usage and cost monitoring for AWS Bedrock — because Cost Explorer won't tell you until tomorrow.

![bedrock-lens preview](assets/image.png)

## The problem

When running Bedrock agents, you're flying blind. AWS Cost Explorer has a 24–48 hour lag. CloudWatch has live data, but getting to it requires knowing the log group name, converting dates to epoch milliseconds, parsing deeply nested JSON, and doing the cost math yourself.

There's no `bedrock usage` command. This is that command.

## Install

```bash
pip install bedrock-lens
```

Or with uv:

```bash
uv tool install bedrock-lens
```

## Usage

```bash
bedrock-lens                        # today's usage (default)
bedrock-lens --yesterday
bedrock-lens --week
bedrock-lens --since 2h             # last 2 hours
bedrock-lens --since 30m            # last 30 minutes
bedrock-lens --since 1d             # last 1 day
bedrock-lens --live                 # tail mode, refreshes every 5s
bedrock-lens --since 1h --live      # live tail for the last hour
bedrock-lens --live --threshold 2   # alert when spend crosses $2
bedrock-lens --setup                # one-time setup wizard
```

```bash
# different profile / region
bedrock-lens --profile my-profile --region us-west-2
```

## First-time setup

Bedrock doesn't log invocations by default. Run the setup wizard once per AWS account:

```bash
bedrock-lens --setup
```

This creates the CloudWatch log group, an IAM role for Bedrock to write to it, and enables model invocation logging. Takes about 10 seconds. After that, every Bedrock call shows up within ~30 seconds.

By default, the log group is created with no retention policy (AWS keeps logs forever). Use `--retention` to control this:

```bash
bedrock-lens --setup --retention 90   # expire logs after 90 days
bedrock-lens --setup --retention 0    # remove any existing retention policy
```

Omitting `--retention` leaves any existing policy untouched.

If you don't have IAM permissions to create roles, the wizard prints the exact policies and CLI command to hand off to your admin.

## How it works

Bedrock writes a JSON record to `/aws/bedrock/model-invocations` in CloudWatch for every model call. Each record contains the model ID, input token counts, and output token count. `bedrock-lens` reads those records, applies per-model pricing, and renders the table.

**Prompt cache tracking:** Bedrock logs cache writes and cache reads as separate token counts from regular input tokens, each billed at a different rate. `bedrock-lens` tracks all three buckets independently and prices them correctly. Cache Write and Cache Read columns appear in the table automatically when any model in the current window has cache activity.

**Live mode** (`--live`) polls every 5 seconds with a 90-second overlap window to handle CloudWatch's ingestion delay, deduplicating events by ID so nothing gets double-counted.

## Pricing

Prices are fetched live at startup from two AWS sources and merged:

**`AmazonBedrockFoundationModels` price list CSV** — the primary source for Anthropic/Claude models (including the latest 4.x releases), Cohere, AI21, and legacy models. Downloaded via `list_price_lists` + `get_price_list_file_url` for your specific region. Uses the standard on-demand Global rate rather than the geo-CRIS regional premium.

**`AmazonBedrock` Price List API** — covers all other providers: Meta (Llama), Mistral, DeepSeek, Google (Gemma), Amazon Nova, Nvidia, Qwen, and 50+ more. Prices are region-specific and update automatically.

For any model not yet in either source — typically new releases in the days before AWS adds them to the catalogue — the tool prompts you to enter the price once and saves it to `~/.config/bedrock-lens/overrides.json`. The entry is removed automatically the next time the model appears in the live pricing data.

Token counts are always accurate for every model regardless of pricing status — they come directly from CloudWatch logs written by Bedrock itself. Unknown models show `N/A` for cost but token counts are never affected.

## Requirements

- Python 3.9+
- AWS credentials with:
  - `logs:FilterLogEvents` on `/aws/bedrock/model-invocations`
  - `bedrock:ListFoundationModels` and `bedrock:ListInferenceProfiles` for live model discovery
  - `pricing:ListPriceLists` and `pricing:GetPriceListFileUrl` for live pricing
- Bedrock model invocation logging enabled (run `--setup` if not)
