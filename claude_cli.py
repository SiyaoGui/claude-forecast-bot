"""
Thin wrapper around the Claude Code command-line tool, used as the bot's LLM.

The bot never calls a paid API key. Each call runs `claude -p` (non-interactive
mode) and authenticates with the owner's Claude subscription through the
CLAUDE_CODE_OAUTH_TOKEN environment variable (created once with
`claude setup-token`). Usage counts against that subscription's limits.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex

# Command used to start Claude Code. Tests point this at a stand-in script.
CLAUDE_CMD = shlex.split(os.getenv("CLAUDE_BIN") or "claude")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet") or "sonnet"
RESEARCH_EFFORT = os.getenv("RESEARCH_EFFORT", "")  # empty = model default
FORECAST_EFFORT = os.getenv("FORECAST_EFFORT", "")  # empty = model default
MAX_PARALLEL_CALLS = int(os.getenv("MAX_PARALLEL_CLAUDE_CALLS", "2") or 2)
RESEARCH_TIMEOUT_SECONDS = int(os.getenv("RESEARCH_TIMEOUT_SECONDS", "900") or 900)
FORECAST_TIMEOUT_SECONDS = int(os.getenv("FORECAST_TIMEOUT_SECONDS", "420") or 420)
RESEARCH_TOOLS = ["WebSearch", "WebFetch"]

_semaphore: asyncio.Semaphore | None = None

# Messages Claude Code prints when a subscription session/weekly limit is hit,
# or when the service is rate limiting. These stop the current run politely;
# the next scheduled run tries again.
_LIMIT_PATTERN = re.compile(
    r"hit your (session|weekly|usage|daily) limit|usage limit|limit reached|"
    r"rate[_ -]?limit|\b429\b|too many requests|out of (usage|credits)|"
    r"spend limit|exhausted",
    re.IGNORECASE,
)
# Messages that mean the login/token is wrong. These fail the run loudly.
_AUTH_PATTERN = re.compile(
    r"invalid api key|invalid (oauth )?token|authentication|unauthori[sz]ed|\b401\b|"
    r"please run /login|not logged in|oauth_org_not_allowed|"
    r"organization has disabled|subscription access",
    re.IGNORECASE,
)


class UsageLimitReached(RuntimeError):
    """The Claude subscription limit was hit. Stop for now, retry on the next run."""


class ClaudeAuthError(RuntimeError):
    """The subscription token is missing, expired or not allowed."""


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_PARALLEL_CALLS)
    return _semaphore


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    # An API key would take precedence over the subscription token in -p mode.
    # Remove it so the subscription is always what gets used.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


def _parse_cli_json(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the last line that looks like a JSON object.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
    return None


def classify_failure(message: str) -> type[Exception]:
    if _AUTH_PATTERN.search(message):
        return ClaudeAuthError
    if _LIMIT_PATTERN.search(message):
        return UsageLimitReached
    return RuntimeError


async def run_claude(
    prompt: str,
    *,
    system_prompt: str,
    tools: list[str] | None,
    effort: str = "",
    timeout_seconds: int = 420,
    retries: int = 1,
) -> str:
    """Run one non-interactive Claude Code call and return the final text."""
    args = [
        *CLAUDE_CMD,
        "-p",
        "--model",
        CLAUDE_MODEL,
        "--output-format",
        "json",
        "--no-session-persistence",
        "--system-prompt",
        system_prompt,
        "--tools",
        ",".join(tools) if tools else "",
    ]
    if tools:
        args += ["--allowedTools", ",".join(tools)]
    if effort:
        args += ["--effort", effort]

    last_error = "unknown error"
    async with _get_semaphore():
        for attempt in range(retries + 1):
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_child_env(),
            )
            try:
                out_b, err_b = await asyncio.wait_for(
                    proc.communicate(prompt.encode("utf-8")), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                last_error = f"Claude call timed out after {timeout_seconds}s"
                continue
            stdout = out_b.decode("utf-8", errors="replace")
            stderr = err_b.decode("utf-8", errors="replace")
            data = _parse_cli_json(stdout)
            if (
                proc.returncode == 0
                and data is not None
                and not data.get("is_error")
                and data.get("subtype", "success") == "success"
                and isinstance(data.get("result"), str)
                and data["result"].strip()
            ):
                return data["result"]

            detail = ""
            if data is not None:
                detail = f"{data.get('subtype', '')} {data.get('result', '')}"
            message = f"{detail} {stderr} {stdout[-800:] if data is None else ''}".strip()
            failure_type = classify_failure(message)
            if failure_type is not RuntimeError:
                raise failure_type(message[:1000])
            last_error = message[:1000] or f"exit code {proc.returncode}"
            if attempt < retries:
                await asyncio.sleep(15)
    raise RuntimeError(f"Claude call failed: {last_error}")


FORECAST_SYSTEM_PROMPT = (
    "You are an expert superforecaster with excellent calibration. "
    "Reason step by step, then end with the final answer in exactly the format requested."
)

RESEARCH_SYSTEM_PROMPT = (
    "You are a meticulous research assistant for a professional forecaster. "
    "Use web search and fetch primary sources to find the most recent relevant facts. "
    "Be concise and factual, give dates and source names, and never give a probability."
)


async def forecast_text(prompt: str) -> str:
    """Pure reasoning call: no tools, short system prompt."""
    return await run_claude(
        prompt,
        system_prompt=FORECAST_SYSTEM_PROMPT,
        tools=None,
        effort=FORECAST_EFFORT,
        timeout_seconds=FORECAST_TIMEOUT_SECONDS,
    )


async def web_research(prompt: str) -> str:
    """Research call with web search and web fetch enabled."""
    return await run_claude(
        prompt,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        tools=RESEARCH_TOOLS,
        effort=RESEARCH_EFFORT,
        timeout_seconds=RESEARCH_TIMEOUT_SECONDS,
    )
