# Claude forecasting bot for Metaculus FutureEval

A fully autonomous forecasting bot for the Metaculus FutureEval bot tournaments
(the seasonal tournament and the two-week MiniBench). No human is involved in any
forecast.

## How it works

Every 20 minutes a GitHub Actions run:

1. Finds the bot tournaments that are running now (the configured ones plus any
   new FutureEval season or MiniBench round it discovers on Metaculus), lists
   their open questions and skips any the bot has already forecast (FutureEval
   takes one forecast per question).
2. For each new question, runs one research pass with Claude using its web search
   and web fetch tools: latest news and official data, the status quo, base rates,
   scheduled events before resolution, and fine-print risks.
3. Asks Claude for an independent forecast three times, using the Metaculus
   template prompts (status quo weighting, wide intervals for numeric questions).
4. Aggregates: median probability for binary questions, median CDF for numeric
   and discrete questions, mean probabilities for multiple choice.
5. Posts the forecast and a private comment containing the research summary and
   the reasoning of each run.

The model is Claude Sonnet, called through the Claude Code command-line tool in
non-interactive mode and authenticated with the owner's Claude subscription
(`CLAUDE_CODE_OAUTH_TOKEN`, created with `claude setup-token`). No paid API key is
used. If the subscription's usage limit is reached, the run stops politely and the
next run retries; a daily cap limits how many questions are forecast per day.

Question handling, prompts, answer parsing and CDF construction come from the
official Metaculus template (`main_with_no_framework.py` in
[metac-bot-template](https://github.com/Metaculus/metac-bot-template)). The model
calls, research step, run loop, limits and logging are new.

## Setup

Repository secrets (Settings > Secrets and variables > Actions):

| Secret | What it is |
|---|---|
| `METACULUS_TOKEN` | API token of the bot account (metaculus.com > Settings > My Forecasting Bots) |
| `CLAUDE_CODE_OAUTH_TOKEN` | Output of `claude setup-token` (valid for one year) |

Repository variables (optional unless noted):

| Variable | Default | Meaning |
|---|---|---|
| `BOT_ENABLED` | not set | **Required for scheduled runs.** Set to `true` after a successful test run. |
| `TOURNAMENTS` | `fall-futureeval-2026,minibench` | Comma separated tournament slugs or IDs that are always checked. |
| `AUTO_DISCOVER_TOURNAMENTS` | `true` | Also check running bot tournaments found on Metaculus (FutureEval, MiniBench, AI Benchmarking), so a new season is picked up without changes. Set to `false` to use only `TOURNAMENTS`. |
| `CLAUDE_MODEL` | `sonnet` | Model alias passed to Claude Code. |
| `NUM_RUNS_PER_QUESTION` | `3` | Independent forecast runs per question. |
| `MAX_QUESTIONS_PER_DAY` | `30` | Daily cap (UTC) to protect subscription usage. |

Test first: Actions > Forecast bot > Run workflow > mode `test`. This forecasts a
few questions in the unscored `bot-testing-area` tournament.

## Files

- `bot.py` - question handling, prompts, aggregation, run loop
- `claude_cli.py` - Claude Code calls, limit and login error handling
- `.github/workflows/forecast.yaml` - scheduled and manual runs
- `logs/forecasts.csv` - created by the bot: one row per question processed
  (time, tournament, question, status). Forecast values are not logged.
