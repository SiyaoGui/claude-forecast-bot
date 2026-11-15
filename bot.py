"""
Autonomous forecasting bot for Metaculus FutureEval bot tournaments.

How it works
- Every run (GitHub Actions, every 20 minutes) it finds the running bot
  tournaments (configured ones plus newly discovered seasons), lists their
  open questions, skips any it has already forecast, and for each new
  question:
    1. runs one web-research pass with Claude (WebSearch + WebFetch tools),
    2. asks Claude for an independent forecast NUM_RUNS_PER_QUESTION times,
    3. aggregates (median for binary and numeric, mean for multiple choice),
    4. posts the forecast and a private comment with the reasoning.
- The LLM is Claude, called through the Claude Code CLI with the owner's
  subscription token (no paid API key). See claude_cli.py.
- No human is involved in any forecast.

Based on the official Metaculus template (main_with_no_framework.py from
https://github.com/Metaculus/metac-bot-template). The question handling,
prompts, parsing and CDF code below come from that file; the model calls,
research step, run loop, limits and logging were rewritten.
"""
from __future__ import annotations

import asyncio
import csv
import datetime
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from pydantic import BaseModel, Field, model_validator

try:  # optional, for local runs with a .env file
    import dotenv

    dotenv.load_dotenv()
except ImportError:
    pass

from claude_cli import (
    CLAUDE_MODEL,
    ClaudeAuthError,
    UsageLimitReached,
    forecast_text,
    web_research,
)

######################### SETTINGS #########################
# BOT_MODE:
#   live = real tournaments, posts forecasts, skips questions already forecast
#   test = Metaculus "bot-testing-area" (unscored), posts forecasts, re-forecasts
#   dry  = no posting at all (local checks)
BOT_MODE = (os.getenv("BOT_MODE") or "live").strip().lower()
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")

# Current seasonal tournament + the rolling MiniBench. Override with the
# TOURNAMENTS variable (comma separated slugs or IDs) when a new season starts.
DEFAULT_LIVE_TOURNAMENTS = ["fall-futureeval-2026", "minibench"]
BOT_TESTING_AREA_ID = "bot-testing-area"
TOURNAMENT_ID = DEFAULT_LIVE_TOURNAMENTS[0]


def _int_env(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    try:
        return int(value) if value else default
    except ValueError:
        return default


NUM_RUNS_PER_QUESTION = _int_env("NUM_RUNS_PER_QUESTION", 3)
MAX_QUESTIONS_PER_RUN = _int_env("MAX_QUESTIONS_PER_RUN", 4 if BOT_MODE == "test" else 15)
MAX_QUESTIONS_PER_DAY = _int_env("MAX_QUESTIONS_PER_DAY", 30)
MAX_COMMENT_CHARS = _int_env("MAX_COMMENT_CHARS", 8000)
LOG_PATH = Path(os.getenv("FORECAST_LOG_PATH", "logs/forecasts.csv"))

# Shared state for one run.
RESEARCH_CACHE: dict[int, str] = {}
LIMIT_STATE = {"hit": False}

######################### MODEL CALLS #########################


async def call_llm(prompt: str, model: str = "", temperature: float = 0.3) -> str:
    """Forecast call. Replaces the template's OpenAI call with Claude Code."""
    if LIMIT_STATE["hit"]:
        raise UsageLimitReached("Usage limit already reached earlier in this run")
    try:
        return await forecast_text(prompt)
    except UsageLimitReached:
        LIMIT_STATE["hit"] = True
        raise


RESEARCH_PROMPT_TEMPLATE = """Today is {today} (UTC). Research the forecasting question below so that a professional forecaster can answer it well.

Question: {title}

Resolution criteria:
{resolution_criteria}

Fine print:
{fine_print}

Background:
{background}

Forecasting closes at {close_time}. The question is scheduled to resolve around {resolve_time}.

Do this:
1. Search for the latest news and official data relevant to the resolution criteria. Prefer primary sources such as official statistics, government or company announcements, and market or exchange data.
2. Establish the current status quo and the most recent value of any quantity mentioned, with its date.
3. Find base rates or historical comparisons: how often this kind of event has happened, and typical changes over a similar time window.
4. List scheduled events before the resolution date that could change the outcome.
5. Point out anything in the resolution criteria or fine print that could make the question resolve in a surprising way.

Write a brief of at most 600 words with dated facts and source names. End with a short list titled "Key uncertainties". Do not give a probability or a forecast."""


async def run_research(question_details: dict) -> str:
    """One web-research pass per question, shared by all forecast runs."""
    question_id = question_details.get("id")
    if question_id in RESEARCH_CACHE:
        return RESEARCH_CACHE[question_id]
    if LIMIT_STATE["hit"]:
        raise UsageLimitReached("Usage limit already reached earlier in this run")
    prompt = RESEARCH_PROMPT_TEMPLATE.format(
        today=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        title=question_details.get("title", ""),
        resolution_criteria=question_details.get("resolution_criteria", ""),
        fine_print=question_details.get("fine_print", ""),
        background=question_details.get("description", ""),
        close_time=question_details.get("scheduled_close_time", "unknown"),
        resolve_time=question_details.get("scheduled_resolve_time", "unknown"),
    )
    try:
        research = await web_research(prompt)
    except UsageLimitReached:
        LIMIT_STATE["hit"] = True
        raise
    except ClaudeAuthError:
        raise
    except Exception as exc:  # research is helpful but not essential
        print(f"Research failed for question {question_id}: {type(exc).__name__}: {str(exc)[:300]}")
        research = (
            "Online research was unavailable for this question. "
            "Rely on the question text and your background knowledge."
        )
    RESEARCH_CACHE[question_id] = research
    print(f"Research for question {question_id}: {len(research)} characters")
    return research


def build_comment(question_details: dict, model_comment: str) -> str:
    """Private comment posted with every forecast (required for prizes)."""
    research = RESEARCH_CACHE.get(question_details.get("id"), "")
    header = (
        f"Automated forecast by a fully autonomous bot: Claude ({CLAUDE_MODEL}) via the "
        f"Claude Code CLI, one web-research pass, {NUM_RUNS_PER_QUESTION} independent "
        "forecast runs, aggregated. No human was involved in this forecast.\n\n"
    )
    research_part = f"## Research summary\n{research[:2000]}\n\n" if research else ""
    text = header + research_part + model_comment
    if len(text) > MAX_COMMENT_CHARS:
        text = text[: MAX_COMMENT_CHARS - 20] + "\n\n[truncated]"
    return text


######################### TEMPLATE CODE (Metaculus API, prompts, parsing) #########################

AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"


def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS, timeout=60,  # type: ignore
    )
    if not response.ok:
        raise RuntimeError(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                "source": "api",
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS, timeout=60,  # type: ignore
    )
    print(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int | str = TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
                "discrete",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams, timeout=60)  # type: ignore
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


def get_open_question_ids_from_tournament(
    tournament_id: int | str = TOURNAMENT_ID,
) -> list[tuple[int, int]]:
    posts = list_posts_from_tournament(tournament_id)

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(
        url,
        **AUTH_HEADERS, timeout=60,  # type: ignore
    )
    if not response.ok:
        raise Exception(response.text)
    details = json.loads(response.content)
    return details




BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Question background:
{background}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = float(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]

    summary_report = await run_research(question_details)

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )

    async def get_rationale_and_probability(content: str) -> tuple[float, str]:
        rationale = await call_llm(content)

        probability = extract_probability_from_response_as_percentage_not_decimal(
            rationale
        )
        comment = (
            f"Extracted Probability: {probability}%\n\nModel reasoning: "
            f"{rationale}\n\n\n"
        )
        return probability, comment

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100

    final_comment = f"Median Probability: {median_probability}\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_probability, final_comment


####################### NUMERIC ###############
# @title Numeric prompt & functions

NUMERIC_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Background:
{background}

{resolution_criteria}

{fine_print}

Units for answer: {units}

Your research assistant says:
{summary_report}

Today is {today}.

{lower_bound_message}
{upper_bound_message}


Formatting Instructions:
- Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unkowns.

The last thing you write is your final answer as:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""


def extract_percentiles_from_response(forecast_text: str) -> dict:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text) -> dict:
        pattern = r"^.*(?:P|p)ercentile.*$"
        number_pattern = (
            r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        )
        results = []

        for line in text.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers = [
                    float(num) if "." in num else int(num) for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    # Check if the original line had a negative sign before the last number
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
    cdf_size: int,
) -> list[float]:
    """
    Returns: list[float]: A list of 201 float values representing the CDF.
    """

    percentiles = []
    for percentile, value in percentile_values.items():
        percentiles.append(Percentile(percentile=percentile/100, value=value))
    numeric_distribution = NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=open_upper_bound,
        open_lower_bound=open_lower_bound,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        zero_point=zero_point,
        cdf_size=cdf_size,
    )
    cdf_as_objects = numeric_distribution.get_cdf()
    cdf_as_floats = [percentile.percentile for percentile in cdf_as_objects]

    return cdf_as_floats


class NumericDefaults:
    DEFAULT_CDF_SIZE = (
        201  # Discrete questions have fewer points, Numeric will have 201 points
    )
    DEFAULT_INBOUND_OUTCOME_COUNT = DEFAULT_CDF_SIZE - 1
    MAX_NUMERIC_PMF_VALUE = 0.2

    @classmethod
    def get_max_pmf_value(
        cls, cdf_size: int, include_wiggle_room: bool = True
    ) -> float:
        # cap depends on inboundOutcomeCount (0.2 if it is the default 200)
        inbound_outcome_count = cdf_size - 1
        normal_cap = cls.MAX_NUMERIC_PMF_VALUE * (
            cls.DEFAULT_INBOUND_OUTCOME_COUNT / inbound_outcome_count
        )

        if include_wiggle_room:
            return normal_cap * 0.95
        else:
            return normal_cap


class Percentile(BaseModel):
    percentile: float = Field(
        description="A number between 0 and 1 (e.g. '90% of people are age 60 or younger' translates to '0.9')",
    )
    value: float = Field(
        description="The number matching the percentile (e.g. '90% of people are age 60 or younger' translates to '60')",
    )

    @model_validator(mode="after")
    def validate_percentile(self: Percentile) -> Percentile:
        if self.percentile < 0 or self.percentile > 1:
            raise ValueError(
                f"Percentile must be between 0 and 1, but was {self.percentile}"
            )
        if np.isnan(self.percentile):
            raise ValueError(f"Percentile must be a number, but was {self.percentile}")
        return self


class NumericDistribution(BaseModel):
    declared_percentiles: list[Percentile]
    open_upper_bound: bool
    open_lower_bound: bool
    upper_bound: float
    lower_bound: float
    zero_point: float | None
    cdf_size: int | None = (
        None  # Normal numeric questions have 201 points, but discrete questions have fewer
    )
    standardize_cdf: bool = True
    strict_validation: bool = True
    is_date: bool = False

    @model_validator(mode="after")
    def validate_percentiles(self: NumericDistribution) -> NumericDistribution:
        percentiles = self.declared_percentiles
        self._check_percentiles_increasing()
        self._check_log_scaled_fields()

        if not self.strict_validation:
            return self

        self._check_percentile_spacing()

        if self.standardize_cdf:
            self._check_too_far_from_bounds(percentiles)
        if self.standardize_cdf and len(percentiles) == self.cdf_size:
            self._check_distribution_too_tall(percentiles)

        self.declared_percentiles = self._check_and_update_repeating_values(percentiles)
        return self

    def _check_percentiles_increasing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if percentiles[i].percentile >= percentiles[i + 1].percentile:
                raise ValueError("Percentiles must be in strictly increasing order")
            if percentiles[i].value > percentiles[i + 1].value:
                raise ValueError("Values must be in strictly increasing order")
        if len(percentiles) < 2:
            raise ValueError("NumericDistribution must have at least 2 percentiles")

    def _check_percentile_spacing(self) -> None:
        percentiles = self.declared_percentiles
        for i in range(len(percentiles) - 1):
            if abs(percentiles[i + 1].percentile - percentiles[i].percentile) < 5e-05:
                raise ValueError(
                    f"Percentiles at indices {i} and {i+1} are too close. CDF must be increasing by at least 5e-05 at every step. "
                    f"{percentiles[i].percentile} and {percentiles[i+1].percentile} "
                    f"at values {percentiles[i].value} and {percentiles[i+1].value}. "
                    "One possible reason is that your prediction is mostly or completely out of the upper/lower "
                    "bound range thus assigning very little probability to any one x-axis value."
                )

    def _check_log_scaled_fields(self) -> None:
        if self.zero_point is not None and self.lower_bound <= self.zero_point:
            raise ValueError(
                f"Lower bound {self.lower_bound} is less than or equal to the zero point {self.zero_point}. "
                "Lower bound must be greater than the zero point."
            )

        for percentile in self.declared_percentiles:
            if self.zero_point is not None and percentile.value < self.zero_point:
                raise ValueError(
                    f"Percentile value {percentile.value} is less than the zero point {self.zero_point}. "
                    "Determining probability less than zero point is currently not supported."
                )

    def _check_and_update_repeating_values(
        self, percentiles: list[Percentile]
    ) -> list[Percentile]:
        unique_value_count = Counter(percentile.value for percentile in percentiles)
        final_percentiles = []
        for percentile in percentiles:
            value = percentile.value
            count = unique_value_count[value]
            repeated_value = count > 1
            value_in_bounds = self.lower_bound < value < self.upper_bound
            value_above_bound = value >= self.upper_bound
            value_below_bound = value <= self.lower_bound
            epsilon = 1e-10
            if not repeated_value:
                final_percentiles.append(percentile)
            elif value_in_bounds:
                greater_epsilon = 1e-6  # TODO: Figure out why normal epsilon doesn't work. Could cause brittle behavior.
                modification = (1 - percentile.percentile) * greater_epsilon
                final_percentiles.append(
                    Percentile(
                        value=value - modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_above_bound:
                modification = epsilon * percentile.percentile
                final_percentiles.append(
                    Percentile(
                        value=self.upper_bound + modification,
                        percentile=percentile.percentile,
                    )
                )
            elif value_below_bound:
                modification = epsilon * (1 - percentile.percentile)
                final_percentiles.append(
                    Percentile(
                        value=self.lower_bound - modification,
                        percentile=percentile.percentile,
                    )
                )
            else:
                raise ValueError(
                    f"Unexpected state: value {value} is repeated {count} times. Bound is {self.lower_bound} and {self.upper_bound}"
                )
        return final_percentiles

    def _check_too_far_from_bounds(self, percentiles: list[Percentile]) -> None:
        max_to_min_range = self.upper_bound - self.lower_bound

        # TODO: Better handle log scaled questions (a fixed wiggle room percentage doesn't work well for them)
        wiggle_percent = 0.25
        wiggle_room = max_to_min_range * wiggle_percent
        upper_bound_plus_wiggle_room = self.upper_bound + wiggle_room
        lower_bound_minus_wiggle_room = self.lower_bound - wiggle_room
        percentiles_within_bounds_plus_wiggle_room = [
            percentile
            for percentile in percentiles
            if lower_bound_minus_wiggle_room
            <= percentile.value
            <= upper_bound_plus_wiggle_room
        ]
        if len(percentiles_within_bounds_plus_wiggle_room) == 0:
            raise ValueError(
                f"No declared percentiles are within the range of the question +/- {wiggle_percent * 100}%. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles}"
            )

        max_to_min_range_buffer = max_to_min_range * 2
        percentiles_far_exceeding_bounds = [
            percentile
            for percentile in percentiles
            if percentile.value < self.lower_bound - max_to_min_range_buffer
            or percentile.value > self.upper_bound + max_to_min_range_buffer
        ]
        if len(percentiles_far_exceeding_bounds) > 0:
            raise ValueError(
                "Some declared percentiles are far exceeding the bounds of the question. "
                f"Lower bound: {self.lower_bound}, upper bound: {self.upper_bound}. "
                f"Percentiles: {percentiles_far_exceeding_bounds}"
            )

    def _check_distribution_too_tall(self, cdf: list[Percentile]) -> None:
        if len(cdf) != self.cdf_size:
            raise ValueError(
                f"CDF size is not the same as the declared percentiles. CDF size: {len(cdf)}, declared percentiles: {self.cdf_size}"
            )
        cap = NumericDefaults.get_max_pmf_value(len(cdf), include_wiggle_room=False)

        for i in range(len(cdf) - 1):
            pmf_value = cdf[i + 1].percentile - cdf[i].percentile
            if pmf_value > cap:
                raise ValueError(
                    f"Distribution is too concentrated. The probability mass between "
                    f"values {cdf[i].value} and {cdf[i + 1].value} is {pmf_value:.4f}, "
                    f"which exceeds the maximum allowed of {cap:.4f}."
                )

    def get_cdf(self) -> list[Percentile]:
        """
        Turns a list of percentiles into a full distribution (201 points, if numeric, otherwise based on discrete values)
        between upper and lower bound (taking into account probability assigned above and below the bounds)
        that is compatible with Metaculus questions.

        cdf stands for 'continuous distribution function'

        At Metaculus CDFs are often represented with 201 points. Each point has:
        - percentile ("X% of values are below this point". This is the y axis of the cdf graph)
        - 'value' or 'nominal location' (The real world number that answers the question)
        - cdf location (a number between 0 and 1 representing where the point is on the cdf x axis, where 0 is range min, and 1 is range max)
        """

        cdf_size = self.cdf_size or NumericDefaults.DEFAULT_CDF_SIZE
        continuous_cdf = []
        cdf_xaxis = []
        cdf_eval_locations = [i / (cdf_size - 1) for i in range(cdf_size)]
        for l in cdf_eval_locations:
            continuous_cdf.append(self._get_cdf_at(l))
            cdf_xaxis.append(self._cdf_location_to_nominal_location(l))

        if self.standardize_cdf:
            continuous_cdf = self._standardize_cdf(continuous_cdf)

        percentiles = [
            Percentile(value=value, percentile=percentile)
            for value, percentile in zip(cdf_xaxis, continuous_cdf)
        ]
        assert len(percentiles) == cdf_size

        validation_distribution = NumericDistribution(
            declared_percentiles=percentiles,
            open_upper_bound=self.open_upper_bound,
            open_lower_bound=self.open_lower_bound,
            upper_bound=self.upper_bound,
            lower_bound=self.lower_bound,
            zero_point=self.zero_point,
            standardize_cdf=self.standardize_cdf,
        )
        NumericDistribution.model_validate(validation_distribution)
        return percentiles

    @classmethod
    def _percentile_list_to_dict(
        cls, percentiles: list[Percentile], multiply_by_100: bool
    ) -> dict[float, float]:
        return {
            (
                percentile.percentile * 100
                if multiply_by_100
                else percentile.percentile
            ): percentile.value
            for percentile in percentiles
        }

    @classmethod
    def _dict_to_percentile_list(
        cls, percentile_dict: dict[float, float], divide_by_100: bool
    ) -> list[Percentile]:
        return [
            Percentile(
                percentile=percentile / 100 if divide_by_100 else percentile,
                value=value,
            )
            for percentile, value in percentile_dict.items()
        ]

    def _add_explicit_upper_lower_bound_percentiles(
        self,
        input_percentiles: list[Percentile],
    ) -> list[Percentile]:
        open_upper_bound = self.open_upper_bound
        open_lower_bound = self.open_lower_bound
        range_max = self.upper_bound
        range_min = self.lower_bound

        return_percentiles = self._percentile_list_to_dict(
            input_percentiles, multiply_by_100=True
        )
        percentile_max = max(percentile for percentile in return_percentiles.keys())
        percentile_min = min(percentile for percentile in return_percentiles.keys())
        range_size = abs(range_max - range_min)
        buffer = 1 if range_size > 100 else 0.01 * range_size

        # Adjust any values that are exactly at the bounds
        for percentile, value in list(return_percentiles.items()):
            # TODO: Handle this more gracefully for log scaled questions
            #  (where buffer could be quite a bit on the lower bound side)
            if not open_lower_bound and value <= range_min + buffer:
                return_percentiles[percentile] = range_min + buffer
            if not open_upper_bound and value >= range_max - buffer:
                return_percentiles[percentile] = range_max - buffer

        # Set cdf values outside range
        if open_upper_bound:
            if range_max > return_percentiles[percentile_max]:
                halfway_between_max_and_100th_percentile = 100 - (
                    0.5 * (100 - percentile_max)
                )
                return_percentiles[halfway_between_max_and_100th_percentile] = range_max
        else:
            return_percentiles[100] = range_max

        # Set cdf values outside range
        if open_lower_bound:
            if range_min < return_percentiles[percentile_min]:
                halfway_between_min_and_0th_percentile = 0.5 * percentile_min
                return_percentiles[halfway_between_min_and_0th_percentile] = range_min
        else:
            return_percentiles[0] = range_min

        sorted_return_percentiles = dict(sorted(return_percentiles.items()))

        return_list = self._dict_to_percentile_list(
            sorted_return_percentiles, divide_by_100=True
        )
        return return_list

    def _nominal_location_to_cdf_location(self, nominal_value: float) -> float:
        """
        Takes the real world value (like $17k - that would answer the forecasting question)
        and converts it to a cdf location between 0 and 1 depending on
        how far it is between the upper and lower bound
        (it can go over 1 or below 0 if beyond the bounds)
        """
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is not None:
            # logarithmically scaled question
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            if nominal_value == zero_point:
                # If nominal = zero point, then you would take the log of 0. Add a small epsilon to avoid this.
                nominal_value += 1e-10
            unscaled_location = (
                np.log(
                    (nominal_value - range_min) * (deriv_ratio - 1)
                    + (range_max - range_min)
                )
                - np.log(range_max - range_min)
            ) / np.log(deriv_ratio)
        else:
            # linearly scaled question
            unscaled_location = (nominal_value - range_min) / (range_max - range_min)
        return float(unscaled_location)

    def _get_cdf_at(self, cdf_location: float) -> float:
        """
        Helper function that takes a cdf location and returns
        the height (percentile) of the cdf at that location, linearly
        interpolating between values
        """
        bounded_percentiles = self._add_explicit_upper_lower_bound_percentiles(
            self.declared_percentiles
        )
        cdf_location_to_percentile_mapping: list[tuple[float, float]] = []
        for percentile in bounded_percentiles:
            height = percentile.percentile
            location = self._nominal_location_to_cdf_location(percentile.value)
            cdf_location_to_percentile_mapping.append((location, height))
        previous = cdf_location_to_percentile_mapping[0]
        for i in range(1, len(cdf_location_to_percentile_mapping)):
            current = cdf_location_to_percentile_mapping[i]
            epsilon = 1e-10
            if previous[0] - epsilon <= cdf_location <= current[0] + epsilon:
                result = previous[1] + (current[1] - previous[1]) * (
                    cdf_location - previous[0]
                ) / (current[0] - previous[0])
                if np.isnan(result):
                    raise ValueError(f"Result is NaN for cdf location {cdf_location}")
                return result
            previous = current
        raise ValueError(f"CDF location Input {cdf_location} cannot be found")

    def _standardize_cdf(self, cdf: list[float] | np.ndarray) -> list[float]:
        """
        See documentation: https://metaculus.com/api/#:~:text=CDF%20generation%20details in the
            "CDF generation details and examples" section

        Takes a cdf and returns a standardized version of it

        - assigns no mass outside of closed bounds (scales accordingly)
        - assigns at least a minimum amount of mass outside of open bounds
        - increasing by at least the minimum amount (0.01 / 200 = 0.0005)
        - caps the maximum growth to 0.2

        Note, thresholds change with different `inbound_outcome_count`s
        """

        lower_open = self.open_lower_bound
        upper_open = self.open_upper_bound

        # apply lower bound & enforce boundary values
        scale_lower_to = 0 if lower_open else cdf[0]
        scale_upper_to = 1.0 if upper_open else cdf[-1]
        rescaled_inbound_mass = scale_upper_to - scale_lower_to

        def apply_minimum(F: float, location: float) -> float:
            # `F` is the height of the cdf at `location` (in range [0, 1])
            # rescale
            rescaled_F = (F - scale_lower_to) / rescaled_inbound_mass
            # offset
            if lower_open and upper_open:
                return 0.988 * rescaled_F + 0.01 * location + 0.001
            elif lower_open:
                return 0.989 * rescaled_F + 0.01 * location + 0.001
            elif upper_open:
                return 0.989 * rescaled_F + 0.01 * location
            return 0.99 * rescaled_F + 0.01 * location

        for i, value in enumerate(cdf):
            cdf[i] = apply_minimum(value, i / (len(cdf) - 1))

        # apply upper bound
        # operate in PMF space
        pmf = np.diff(cdf, prepend=0, append=1)
        cap = NumericDefaults.get_max_pmf_value(len(cdf))

        def cap_pmf(scale: float) -> np.ndarray:
            return np.concatenate(
                [pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]]
            )

        def capped_sum(scale: float) -> float:
            return float(cap_pmf(scale).sum())

        # find the appropriate scale search space
        lo = hi = scale = 1.0
        while capped_sum(hi) < 1.0:
            hi *= 1.2
        # hone in on scale value that makes capped sum 1
        for _ in range(100):
            scale = 0.5 * (lo + hi)
            s = capped_sum(scale)
            if s < 1.0:
                lo = scale
            else:
                hi = scale
            if s == 1.0 or (hi - lo) < 2e-5:
                break
        # apply scale and renormalize
        pmf = cap_pmf(scale)
        pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
        # back to CDF space
        cdf = np.cumsum(pmf)[:-1]

        # round to minimize floating point errors
        cdf = np.round(cdf, 10)
        return cdf.tolist()

    def _cdf_location_to_nominal_location(self, cdf_location: float) -> float:
        range_max = self.upper_bound
        range_min = self.lower_bound
        zero_point = self.zero_point

        if zero_point is None:
            scaled_location = range_min + (range_max - range_min) * cdf_location
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scaled_location = range_min + (range_max - range_min) * (
                deriv_ratio**cdf_location - 1
            ) / (deriv_ratio - 1)
        if np.isnan(scaled_location):
            raise ValueError(f"Scaled location is NaN for cdf location {cdf_location}")
        return scaled_location


async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int
) -> tuple[list[float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    unit_of_measure = (
        question_details["unit"]
        if question_details["unit"]
        else "Not stated (please infer this)"
    )
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling["zero_point"]
    if question_type == "discrete":
        outcome_count = question_details["scaling"]["inbound_outcome_count"]
        cdf_size = outcome_count + 1
    else:
        cdf_size = 201

    # Create messages about the bounds that are passed in the LLM prompt
    if open_upper_bound:
        upper_bound_message = ""
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = ""
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    summary_report = await run_research(question_details)

    content = NUMERIC_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        units=unit_of_measure,
    )

    async def ask_llm_to_get_cdf(content: str) -> tuple[list[float], str]:
        rationale = await call_llm(content)
        percentile_values = extract_percentiles_from_response(rationale)

        comment = (
            f"Extracted Percentile_values: {percentile_values}\n\nModel reasoning: "
            f"{rationale}\n\n\n"
        )

        cdf = generate_continuous_cdf(
            percentile_values,
            question_type,
            open_upper_bound,
            open_lower_bound,
            upper_bound,
            lower_bound,
            zero_point,
            cdf_size,
        )

        return cdf, comment

    cdf_and_comment_pairs = await asyncio.gather(
        *[ask_llm_to_get_cdf(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in cdf_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    cdfs: list[list[float]] = [pair[0] for pair in cdf_and_comment_pairs]
    all_cdfs = np.array(cdfs)
    median_cdf: list[float] = np.median(all_cdfs, axis=0).tolist()

    final_comment = f"Median CDF: `{str(median_cdf)[:100]}...`\n\n" + "\n\n".join(
        final_comment_sections
    )
    return median_cdf, final_comment


########################## MULTIPLE CHOICE ###############
# @title Multiple Choice prompt & functions

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

The options are: {options}


Background:
{background}

{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""


def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text):

        # Number extraction pattern
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        # Iterate through each line in the text
        for line in text.split("\n"):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            # Convert strings to float or int
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """

    # confirm that there is a probability for each option
    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    # Ensure we are using decimals
    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment

        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
) -> tuple[dict[str, float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    options = question_details["options"]

    summary_report = await run_research(question_details)

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )

    async def ask_llm_for_multiple_choice_probabilities(
        content: str,
    ) -> tuple[dict[str, float], str]:
        rationale = await call_llm(content)

        option_probabilities = extract_option_probabilities_from_response(
            rationale, options
        )

        comment = (
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nModel reasoning: "
            f"{rationale}\n\n\n"
        )

        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return probability_yes_per_category, comment

    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: list[dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: list[float] = [
            dict[option] for dict in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    final_comment = (
        f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`\n\n"
        + "\n\n".join(final_comment_sections)
    )
    return average_probability_yes_per_category, final_comment


################### FORECASTING ###################
def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary_of_forecast = ""
    summary_of_forecast += (
        f"-----------------------------------------------\nQuestion: {title}\n"
    )
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions == True
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    if question_type == "binary":
        forecast, comment = await get_binary_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "numeric":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "discrete":
        forecast, comment = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "multiple_choice":
        forecast, comment = await get_multiple_choice_gpt_prediction(
            question_details, num_runs_per_question
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print(
        f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n"
    )
    print(f"Forecast ready for post {post_id} (question {question_id})")
    print(f"Comment for post {post_id} (question {question_id}): {len(comment)} characters")

    if question_type == "numeric" or question_type == "discrete":
        summary_of_forecast += f"Forecast: {str(forecast)[:200]}...\n"
    else:
        summary_of_forecast += f"Forecast: {forecast}\n"

    summary_of_forecast += f"Comment:\n```\n{comment[:200]}...\n```\n\n"

    if submit_prediction == True:
        forecast_payload = create_forecast_payload(forecast, question_type)
        post_question_prediction(question_id, forecast_payload)
        post_question_comment(post_id, build_comment(question_details, comment))
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast




######################### RUN LOOP #########################


async def forecast_questions(
    items: list[tuple[str, int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> list[dict]:
    """Forecast every (tournament, question_id, post_id) item and report a status per item."""

    async def one(item: tuple[str, int, int]) -> dict:
        tournament, question_id, post_id = item
        started = time.time()
        result = {
            "tournament": str(tournament),
            "question_id": question_id,
            "post_id": post_id,
            "status": "error",
            "detail": "",
        }
        try:
            summary = await forecast_individual_question(
                question_id,
                post_id,
                submit_prediction,
                num_runs_per_question,
                skip_previously_forecasted_questions,
            )
            if "Posted: Forecast was posted to Metaculus." in summary:
                result["status"] = "posted"
            elif "Skipped: Forecast already made" in summary:
                result["status"] = "skipped"
            else:
                result["status"] = "dry_run"
        except UsageLimitReached as exc:
            result["status"] = "usage_limit"
            result["detail"] = str(exc)[:300]
        except ClaudeAuthError as exc:
            result["status"] = "auth_error"
            result["detail"] = str(exc)[:300]
        except Exception as exc:
            result["status"] = "error"
            result["detail"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        result["seconds"] = round(time.time() - started)
        return result

    return list(await asyncio.gather(*[one(item) for item in items]))


# Words that mark a Metaculus tournament as a bot tournament (FutureEval seasons,
# MiniBench, older "AI Benchmarking" series). Matched against slug and name.
BOT_TOURNAMENT_PATTERN = re.compile(
    r"futureeval|future-eval|minibench|mini-bench|ai[- ]?benchmark|\baib\b|bot[- ]tournament",
    re.IGNORECASE,
)
AUTO_DISCOVER = (os.getenv("AUTO_DISCOVER_TOURNAMENTS") or "true").strip().lower() != "false"


def _parse_time(value) -> datetime.datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        t = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def discover_bot_tournaments() -> list[str]:
    """Find bot tournaments that are running now, so a new season is picked up
    automatically. Returns slugs (or IDs). Any failure returns an empty list and
    the configured tournaments are used as before."""
    try:
        response = requests.get(f"{API_BASE_URL}/projects/tournaments/", **AUTH_HEADERS, timeout=60)  # type: ignore
        if not response.ok:
            print(f"Tournament discovery skipped: HTTP {response.status_code}")
            return []
        data = response.json()
    except Exception as exc:
        print(f"Tournament discovery skipped: {type(exc).__name__}: {str(exc)[:200]}")
        return []
    items = data.get("results", []) if isinstance(data, dict) else data
    now = datetime.datetime.now(datetime.timezone.utc)
    found: list[str] = []
    for t in items if isinstance(items, list) else []:
        if not isinstance(t, dict):
            continue
        slug = str(t.get("slug") or "")
        name = str(t.get("name") or "")
        if not BOT_TOURNAMENT_PATTERN.search(f"{slug} {name}"):
            continue
        if slug == BOT_TESTING_AREA_ID or "testing" in slug:
            continue
        if t.get("is_ongoing") is False:
            continue
        start = _parse_time(t.get("start_date"))
        end = _parse_time(t.get("forecasting_end_date") or t.get("close_date"))
        if start and start > now:
            continue
        if end and end < now:
            continue
        found.append(slug or str(t.get("id")))
    return found


def parse_tournaments() -> list[str]:
    if BOT_MODE == "test":
        return [BOT_TESTING_AREA_ID]
    override = (os.getenv("TOURNAMENTS") or "").strip()
    configured = (
        [t.strip() for t in override.split(",") if t.strip()]
        if override
        else list(DEFAULT_LIVE_TOURNAMENTS)
    )
    if not AUTO_DISCOVER:
        return configured
    discovered = discover_bot_tournaments()
    print(f"Running bot tournaments found on Metaculus: {discovered}")
    new = [t for t in discovered if t not in configured]
    if new:
        print(f"Discovered running bot tournaments not in the configured list: {new}")
    return configured + new


def count_posted_today() -> int:
    if not LOG_PATH.exists():
        return 0
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    count = 0
    with LOG_PATH.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (
                row.get("mode") == "live"
                and row.get("status") == "posted"
                and row.get("utc_time", "").startswith(today)
            ):
                count += 1
    return count


LOG_FIELDS = ["utc_time", "mode", "tournament", "post_id", "question_id", "status", "seconds", "model", "runs"]


def write_log(results: list[dict]) -> None:
    """Append one row per processed question. No forecast values are logged."""
    if not results or BOT_MODE == "dry":
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_PATH.exists()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new_file:
            writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "utc_time": now,
                    "mode": BOT_MODE,
                    "tournament": r["tournament"],
                    "post_id": r["post_id"],
                    "question_id": r["question_id"],
                    "status": r["status"],
                    "seconds": r.get("seconds", ""),
                    "model": CLAUDE_MODEL,
                    "runs": NUM_RUNS_PER_QUESTION,
                }
            )


def check_environment() -> None:
    problems: list[str] = []
    if BOT_MODE not in {"live", "test", "dry"}:
        problems.append(f"BOT_MODE must be live, test or dry (got '{BOT_MODE}').")
    if not METACULUS_TOKEN:
        problems.append(
            "METACULUS_TOKEN is missing. Create a bot at "
            "https://www.metaculus.com/accounts/settings/bots/ and add its token as a repository secret."
        )
    if BOT_MODE != "dry" and not os.getenv("CLAUDE_CODE_OAUTH_TOKEN") and not os.getenv("CLAUDE_BIN"):
        problems.append(
            "CLAUDE_CODE_OAUTH_TOKEN is missing. Run `claude setup-token` on your computer "
            "and add the token as a repository secret."
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY is set but ignored; the subscription token is used.")
    if problems:
        print("Setup problems:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


def main() -> int:
    check_environment()
    tournaments = parse_tournaments()
    submit = BOT_MODE in {"live", "test"}
    skip_previous = BOT_MODE == "live"
    print(
        f"Mode={BOT_MODE} tournaments={tournaments} model={CLAUDE_MODEL} "
        f"runs_per_question={NUM_RUNS_PER_QUESTION} submit={submit}"
    )

    room = MAX_QUESTIONS_PER_RUN
    if BOT_MODE == "live":
        posted_today = count_posted_today()
        room = min(room, max(0, MAX_QUESTIONS_PER_DAY - posted_today))
        print(f"Forecasts posted today (UTC): {posted_today}; daily cap {MAX_QUESTIONS_PER_DAY}")
        if room == 0:
            print("::warning::Daily cap reached. New questions wait for the next UTC day.")
            return 0

    candidates: list[tuple[str, int, int]] = []
    seen: set[int] = set()
    listing_failed = False
    for tournament in tournaments:
        try:
            pairs = get_open_question_ids_from_tournament(tournament)
        except Exception as exc:
            listing_failed = True
            print(f"Could not list questions for {tournament}: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        for question_id, post_id in pairs:
            if question_id not in seen:
                seen.add(question_id)
                candidates.append((tournament, question_id, post_id))

    todo: list[tuple[str, int, int]] = []
    for tournament, question_id, post_id in candidates:
        if skip_previous:
            try:
                if forecast_is_already_made(get_post_details(post_id)):
                    continue
            except Exception as exc:
                print(f"Could not read post {post_id}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
        todo.append((tournament, question_id, post_id))

    if len(todo) > room:
        print(f"{len(todo)} new questions, processing {room} this run (limits).")
        todo = todo[:room]
    if not todo:
        print("Nothing new to forecast right now.")
        return 1 if listing_failed else 0

    results = asyncio.run(
        forecast_questions(todo, submit, NUM_RUNS_PER_QUESTION, skip_previous)
    )
    write_log(results)

    counts = Counter(r["status"] for r in results)
    print("\n" + "=" * 70)
    print("Run summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for r in results:
        line = f"  [{r['status']}] {r['tournament']} https://www.metaculus.com/questions/{r['post_id']}/ ({r.get('seconds', '?')}s)"
        if r["detail"]:
            line += f" :: {r['detail'][:200]}"
        print(line)
    print("=" * 70)

    if counts.get("auth_error"):
        print("::error::Claude login failed. Check the CLAUDE_CODE_OAUTH_TOKEN secret (run `claude setup-token` again).")
        return 1
    if counts.get("error"):
        print("::error::Some questions failed. See the run summary above.")
        return 1
    if counts.get("usage_limit"):
        print("::warning::Claude subscription limit reached. Remaining questions will be retried on the next run.")
    return 1 if listing_failed else 0


if __name__ == "__main__":
    sys.exit(main())
