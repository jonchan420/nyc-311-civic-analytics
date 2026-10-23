"""Deterministic routing: answer with a tool, ask a question back, decline, or explain.

These rules run before any model. A question they fully understand is
answered with no model call at all; a question that is out of scope is
declined with a reason; an ambiguous one gets a clarifying question.
Only what the rules can't place is handed to a model planner (agent.py).
Every assumption the rules make is recorded so the answer can state it.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from analyst.catalog import spelling_key
from analyst.data import DataSource, month_name, shift_month
from config.settings import DATA_DIR, LIVE_DIR

MONTHS = {"jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4, "april": 4,
          "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "sept": 9,
          "september": 9, "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12}
MON = "(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2)}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
BOROUGHS = {"staten island": "STATEN ISLAND", "the bronx": "BRONX", "bronx": "BRONX", "brooklyn": "BROOKLYN",
            "manhattan": "MANHATTAN", "queens": "QUEENS"}
CITYWIDE_WORDS = r"\b(citywide|city-wide|city wide|nyc|new york city|the whole city|the city|all (?:five |5 )?boroughs)\b"
NEIGHBORHOOD_FILE = DATA_DIR / "equity" / "results" / "neighborhood_reporting_gap.csv"

# Category names that are also everyday words; they only match through ALIASES.
AMBIGUOUS_NAMES = {"GENERAL", "SAFETY", "LEAD", "DRINKING", "STORM", "UNSPECIFIED", "BENCH", "NOISE",
                   "SNW", "SRDE", "SRGOVG", "ZTESTINT", "PLANT"}
# Everyday words for categories: pattern -> category names (several = all of them)
ALIASES: dict[str, list[str]] = {
    r"\b(no )?heat\b|\bhot water\b": ["HEAT/HOT WATER"],
    r"\bpotholes?\b": ["Street Condition"],
    r"\b(rats?|mice|rodents?)\b": ["Rodent"],
    r"\bsnow\b|\bicy\b": ["Snow or Ice"],
    r"\b(double|illegal(ly)?) park(ing|ed)\b": ["Illegal Parking"],
    r"\bmissed (garbage|trash|recycling|pick ?ups?|collections?)\b": ["Missed Collection"],
    r"\b(residential|apartment|neighbou?r) noise\b|\bnoisy neighbou?rs?\b": ["Noise - Residential"],
    r"\b(street|sidewalk) noise\b": ["Noise - Street/Sidewalk"],
    r"\b(vehicle|car|traffic) noise\b": ["Noise - Vehicle"],
    r"\bcommercial noise\b|\bnoisy (bars?|stores?|restaurants?)\b": ["Noise - Commercial"],
    r"\bhelicopters?\b": ["Noise - Helicopter"],
}
FAMILIES = {"noise": "NOISE", "trees?": "TREE"}  # a family word means every category named with it
UNCLEAR_WORDS = {r"\b(trash|garbage|litter)\b": ["Dirty Condition", "Missed Collection", "Illegal Dumping",
                                                 "Litter Basket Complaint"]}

OFF_TOPIC = ("I can only answer questions about recorded NYC 311 complaints: how many were filed, where "
             "(by borough or citywide), when, and of which type. That question isn't about them.")
# a sentence that tries to change the rules is dropped before routing; only those sentences
INJECTION = (r"\b(ignore|disregard|forget|override)\b[^.?!;]{0,60}\b(rules?|instructions?|prompts?|guidelines?|"
             r"previous|above|everything)\b|\byou are now\b|\bdeveloper mode\b|\bjailbreak\b|\bsystem prompt\b"
             r"|\b(reveal|print|show|tell me|give me|output)\b[^.?!;]{0,40}\b(api ?keys?|secrets?|passwords?|tokens?)\b")
INJECTION_REPLY = ("I can't change how I work, follow instructions written into a question, or share settings or "
                   "keys. I can answer questions about recorded NYC 311 complaints.")
INJECTION_NOTE = "Part of the question asked me to change how I work; that part was ignored."
DECLINES = [
    (r"\b(write|compose|create|generate|draft)\b[^.?!]*\b(poems?|story|stories|songs?|jokes?|essays?|haikus?|"
     r"limericks?|raps?|lyrics)\b"
     r"|\b(tell|sing|make up|give)\b[^.?!]{0,20}\b(a|an|some)\s+(\w+\s+){0,2}(poems?|story|stories|songs?|jokes?|"
     r"riddles?|haikus?|limericks?|raps?)\b", OFF_TOPIC),
    (r"\b(select|insert|update|delete|drop)\b.+\b(from|into|table|where)\b|\bsql\b",
     "I can't run SQL or code someone supplies. Ask the question in words and I'll pick one of the "
     "checked calculations."),
    (r"\bresponse times?\b|\bhow long\b|\btime to (close|resolve|respond|fix)\b|\b(closing|resolution) times?\b"
     r"|\bget (fixed|resolved|closed)\b",
     "Response and closing times aren't in the tables this assistant reads yet, so I can't answer "
     "that. I can show how many complaints were filed."),
    (r"\bcrimes?\b|\bsafe(r|st)?\b|\bdangerous\b|\bshootings?\b|\bmurders?\b|\brobber(y|ies)\b",
     "311 records complaints and service requests, not crimes, so it can't measure crime or safety. "
     "NYPD crime data is a separate dataset."),
    (r"\b(my complaint|complaint (number|status)|status of my|my request|who (filed|reported|complained))\b"
     r"|\baddress(es)?\b|\bwhich buildings?\b",
     "I only work with monthly totals, not individual complaints, addresses, or who filed them."),
    (r"\b(zip ?codes?|1[01]\d{3})\b|\bcensus tracts?\b|\bcommunity (boards?|districts?)\b|\bblocks?\b",
     "This assistant works by borough and citywide; ZIP codes, community districts, and blocks aren't "
     "supported yet."),
    (r"\bagenc(y|ies)\b.*\b(perform|best|worst|fastest|slowest|doing)\b|\bhow well\b",
     "Agency performance isn't something these complaint counts can measure."),
]
VAGUE_TIME = r"\b(recently|lately|these days|nowadays)\b"
DEFINITION = (r"\b(mean|means|meaning|defined|definition|count as|counts as|counted as|calculated|computed|"
              r"scored|measured)\b")
# words that show a question is about the complaint data, when no place, period, or type is named
DATA_WORDS = (r"\b(complaints?|requests?|311|calls?|reports?|reported|reporting|filed|counts?|numbers?|totals?|"
              r"trends?|changes?|changed|chang\w*|rise|rose|risen|rising|increase\w*|decrease\w*|drop\w*|fell|"
              r"fall(en|ing)|grew|grow\w*|compar\w*|vs|versus|brief\w*|summar\w*|overview|recap|happen\w*|"
              r"busiest|quietest|peak|common|top|charts?|graphs?|monthly|data)\b")
EXPLAIN = (r"^(what|which|how|why|where|when|is|are|does|do|can)\b.*\b(mean|means|meaning|defin|calculat|computed|"
           r"scor|wape|come from|source|methodolog|limit|caveat|flag|renam|catalog|per[- ]capita|partial|"
           r"unusual|spike|missing|accurate|accuracy|reliable|trust|count as|reporting bias|interpret|"
           r"add up|differ)|^why (do|does|is|are|did)(n'?t| not)\b")


@dataclass
class RouterContext:
    complete_through: str
    first_month: str
    categories: list[str]
    neighborhoods: dict[str, set[str]]
    renamed: dict[str, str] = field(default_factory=dict)   # old label -> current category name

    @classmethod
    def load(cls, data_dir: Path = LIVE_DIR, neighborhood_file: Path = NEIGHBORHOOD_FILE) -> RouterContext:
        with DataSource(data_dir) as source:
            categories = source.catalog.names()
            renamed = source.catalog.renamed_labels()
            complete_through, first_month = source.complete_through, source.first_month
        return cls(complete_through, first_month, categories, load_neighborhoods(neighborhood_file), renamed)


def load_neighborhoods(path: Path = NEIGHBORHOOD_FILE) -> dict[str, set[str]]:
    """Neighborhood name -> borough(s), from NYC Health's neighborhood names."""
    if not Path(path).exists():
        return {}
    names: dict[str, set[str]] = {}
    table = pd.read_csv(path, usecols=["neighborhood", "borough"])
    for neighborhood, borough in zip(table["neighborhood"], table["borough"]):
        for part in str(neighborhood).split("/"):
            name = re.sub(r"\s*\(.*?\)", "", part).strip()
            if len(name) > 3:
                names.setdefault(name.lower(), set()).add(borough)
    return names


@dataclass
class Route:
    kind: str                                  # tool | clarify | decline | explain | model
    tool: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    message: str = ""                          # the question back, the reason, or what's missing
    options: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


@dataclass
class Period:
    start: str
    end: str
    label: str
    position: int
    endpoints: bool = False                    # written as "<month> to <month>"

    @property
    def months(self) -> int:
        (y1, m1), (y2, m2) = (map(int, s.split("-")) for s in (self.start, self.end))
        return (y2 - y1) * 12 + m2 - m1 + 1

    def as_param(self) -> dict[str, str]:
        return {"start": self.start, "end": self.end}

    def shifted(self, months: int, label: str) -> Period:
        return Period(shift_month(self.start, months), shift_month(self.end, months), label, self.position)


def strip_injection(question: str) -> tuple[str, bool]:
    """The question without any sentence that tries to change the rules, and whether one was dropped."""
    sentences = [s for s in re.split(r"(?<=[.!?;:])\s+", question.strip()) if s]
    kept = [s for s in sentences if not re.search(INJECTION, s.lower())]
    return " ".join(kept), len(kept) < len(sentences)


def has_words(text: str) -> bool:
    return bool(re.search(r"[a-z]{3}", text.lower()))


def route(question: str, context: RouterContext) -> Route:
    cleaned, dropped = strip_injection(question)
    if dropped:
        if not has_words(cleaned):
            return Route("decline", message=INJECTION_REPLY)
        plan = route(cleaned, context)
        plan.assumptions.insert(0, INJECTION_NOTE)
        return plan
    text = " ".join(question.lower().split())
    for pattern, reason in DECLINES:
        if re.search(pattern, text):
            return Route("decline", message=reason)

    place = _find_place(text, context)
    if place.kind != "tool":
        return place
    geography = place.params["geography"]

    if re.search(r"^(how|what|which)\b", text) and re.search(DEFINITION, text):  # how a number is made, not the number
        return Route("explain", message="method question")
    forecast = bool(re.search(r"\b(forecasts?|predict(s|ed|ion|ions)?|projections?|next month|expect(ed)? next)\b", text))
    if forecast:
        assumptions = []
        if geography != "CITYWIDE":
            assumptions.append("The forecasts cover the whole city; there's no separate borough forecast.")
        return Route("tool", "forecast_status", {}, assumptions=assumptions)

    periods, assumptions = _find_periods(text, context)
    if re.search(EXPLAIN, text) and (not periods or re.search(DEFINITION, text)):
        return Route("explain", message="method question")

    if re.search(r"\b(this|current) month\b|\btoday\b|\bthis week\b", text):
        return Route("decline", message=(
            f"The data is complete through {month_name(context.complete_through)}, and the current "
            "month isn't finished, so its numbers would look misleadingly low."))

    for period in periods:
        if period.end > context.complete_through:
            return Route("decline", message=(
                f"The data is complete through {month_name(context.complete_through)}, so "
                f"{period.label} can't be answered yet."))
        if period.start < context.first_month:
            return Route("decline", message=f"The data begins in {month_name(context.first_month)}.")

    categories, category_notes, unclear = _find_categories(text, context.categories, context.renamed)
    if unclear:
        return Route("clarify", message=f'"{unclear[0]}" could mean several complaint types. Which one?',
                     options=unclear[1])
    if not categories:
        unknown = _unrecognized_topic(text)
        if unknown:
            suggestions = difflib.get_close_matches(unknown, context.categories, n=4, cutoff=0.5)
            return Route("model", message=f'No complaint type called "{unknown}" was found.',
                         options=suggestions)
    named_place = geography != "CITYWIDE" or re.search(CITYWIDE_WORDS, text)
    if text.startswith("why"):
        if not (named_place or periods or categories):  # counts can't say why; the notes explain what they can
            return Route("explain", message="why question")
        assumptions.append("The counts show what changed, not why it changed.")
    if not (named_place or periods or categories or re.search(DATA_WORDS, text)):
        return Route("decline", message=OFF_TOPIC)
    assumptions += category_notes
    if place.assumptions:
        assumptions = place.assumptions + assumptions

    briefing = bool(re.search(r"\b(brief(ing)?|summar(y|ize|ise)|overview|recap|what happened|report on)\b", text))
    trend = bool(re.search(r"\b(trends?|over time|monthly|each month|per month|month by month|chart|graph|"
                           r"history|since|past \d+ months|last \d+ months)\b", text))
    compare = bool(re.search(r"\b(compare[ds]?|comparison|versus|vs\.?|between|change[ds]?|increase[ds]?|"
                             r"decrease[ds]?|rise|rose|risen|drop(ped)?|fell|fallen|grew|grow(n|th)|"
                             r"contribut\w*|dr(o|i)ve\w*|driving|movers?|(went|gone|go|going) (up|down)|"
                             r"up or down|more or fewer)\b",
                             text))

    if briefing:
        return _summary_route(geography, periods, assumptions, context)
    if compare and not trend and len(periods) == 1 and periods[0].endpoints:
        whole = periods[0]  # "how did X change from June 2025 to June 2026" compares the two months
        periods = [Period(whole.start, whole.start, month_name(whole.start), whole.position),
                   Period(whole.end, whole.end, month_name(whole.end), whole.position + 1)]
        assumptions.append(f"This compares {month_name(whole.start)} with {month_name(whole.end)}, "
                           "not the months in between.")
    if len(periods) >= 2:
        return _compare_route(geography, periods, categories, assumptions, context)
    if trend:
        return _trend_route(geography, periods, categories, assumptions, context, text)
    if compare:
        if not periods and re.search(VAGUE_TIME, text):
            return Route("clarify", message="Which periods should I compare?", options=_period_suggestions(context))
        return _compare_route(geography, periods, categories, assumptions, context)
    if len(periods) == 1 and periods[0].months == 1 and not categories:
        return _summary_route(geography, periods, assumptions, context)
    return _trend_route(geography, periods, categories, assumptions, context, text)


# ------------------------------------------------------------------ places

def _find_place(text: str, context: RouterContext) -> Route:
    masked = text
    for name in sorted(context.neighborhoods, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", masked):
            boroughs = sorted(context.neighborhoods[name])
            options = [b.title() for b in boroughs]
            if len(boroughs) > 1:
                return Route("clarify", message=(
                    f"{name.title()} is the name of neighborhoods in {' and '.join(options)}, and I work "
                    "at the borough level. Which borough should I use?"), options=options)
            return Route("clarify", message=(
                f"I work at the borough level, not by neighborhood. {name.title()} is in {options[0]}; "
                f"should I answer for {options[0]}?"), options=options)
    if re.search(r"\bthe heights\b|^heights\b|\bheights\b", masked):
        return Route("clarify", message=(
            "\"The Heights\" could be Washington Heights (Manhattan), Brooklyn Heights or Crown Heights "
            "(Brooklyn), or Jackson Heights (Queens), and I work at the borough level. Which borough?"),
            options=["Manhattan", "Brooklyn", "Queens"])
    found = sorted({code for alias, code in BOROUGHS.items() if re.search(rf"\b{alias}\b", masked)})
    if len(found) > 1:
        return Route("clarify", message=(
            "I compare periods within one place at a time. Which should I start with?"),
            options=[b.title() for b in found])
    if found:
        return Route("tool", params={"geography": found[0]})
    if re.search(r"\b(neighbou?rhoods?|my area|my block|my street)\b", masked):
        return Route("clarify", message="I work at the borough level. Which borough, or citywide?",
                     options=["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island", "Citywide"])
    note = [] if re.search(CITYWIDE_WORDS, masked) else ["No borough was named, so this covers the whole city."]
    return Route("tool", params={"geography": "CITYWIDE"}, assumptions=note)


# ------------------------------------------------------------------ periods

def _find_periods(text: str, context: RouterContext) -> tuple[list[Period], list[str]]:
    found: list[Period] = []
    taken: list[tuple[int, int]] = []
    assumptions: list[str] = []

    def add(match: re.Match, start: str, end: str, label: str, note: str | None = None,
            endpoints: bool = False) -> None:
        span = match.span()
        if any(span[0] < b and a < span[1] for a, b in taken):
            return
        taken.append(span)
        found.append(Period(start, end, label, span[0], endpoints))
        if note:
            assumptions.append(note)

    def ym(year: int, month: int) -> str:
        return f"{year:04d}-{month:02d}"

    for m in re.finditer(rf"\b{MON}\s+(\d{{4}})\s*(?:to|through|thru|until|-|–)\s*{MON}\s+(\d{{4}})\b", text):
        start, end = ym(int(m[2]), MONTHS[m[1]]), ym(int(m[4]), MONTHS[m[3]])
        add(m, start, end, f"{month_name(start)} to {month_name(end)}", endpoints=True)
    for m in re.finditer(rf"\b{MON}\s*(?:to|through|thru|until|-|–)\s*{MON}\s+(\d{{4}})\b", text):
        first, last, year = MONTHS[m[1]], MONTHS[m[2]], int(m[3])
        start, end = ym(year - 1 if first > last else year, first), ym(year, last)
        add(m, start, end, f"{month_name(start)} to {month_name(end)}", endpoints=True)
    for m in re.finditer(r"\b(spring|summer|fall|autumn|winter)\s+(?:of\s+)?(\d{4})\b", text):
        first, last = SEASONS[m[1]]
        year = int(m[2])
        start = ym(year - 1, 12) if m[1] == "winter" else ym(year, first)
        add(m, start, ym(year, last), m[0],
            f"{m[0].title()} means {month_name(start)} to {month_name(ym(year, last))}.")
    for m in re.finditer(r"\bq([1-4])\s*(?:of\s+)?(\d{4})\b|\b(first|second|third|fourth) quarter (?:of )?(\d{4})\b", text):
        quarter = int(m[1]) if m[1] else ORDINALS[m[3]]
        year = int(m[2] or m[4])
        add(m, ym(year, 3 * quarter - 2), ym(year, 3 * quarter), f"Q{quarter} {year}")
    for m in re.finditer(r"\b(first|second) half (?:of )?(\d{4})\b", text):
        half = ORDINALS[m[1]]
        add(m, ym(int(m[2]), 6 * half - 5), ym(int(m[2]), 6 * half), f"the {m[1]} half of {m[2]}")
    latest = context.complete_through
    for m in re.finditer(r"\bsince\s+(?:(" + MON[1:-4] + r")\s+)?(\d{4})\b", text):
        start = ym(int(m[2]), MONTHS[m[1]] if m[1] else 1)
        add(m, start, latest, m[0], f"Since {month_name(start)} means through {month_name(latest)}, the latest complete month.")
    for m in re.finditer(rf"\b{MON}\s+(\d{{4}})\b", text):
        month = ym(int(m[2]), MONTHS[m[1]])
        add(m, month, month, month_name(month))
    for m in re.finditer(r"\b(\d{4})-(0[1-9]|1[0-2])\b", text):
        add(m, m[0], m[0], month_name(m[0]))
    for m in re.finditer(r"\b(?:last|past|previous) (\d{1,2}) months\b", text):
        n = int(m[1])
        add(m, shift_month(latest, -(n - 1)), latest, m[0],
            f"The last {n} months means {month_name(shift_month(latest, -(n - 1)))} to {month_name(latest)}.")
    for m in re.finditer(r"\blast month\b", text):
        add(m, latest, latest, "last month",
            f"Last month means {month_name(latest)}, the latest complete month.")
    for m in re.finditer(r"\b(this|last|past|most recent) (spring|summer|fall|autumn|winter)\b", text):
        first, last = SEASONS[m[2]]
        year = int(latest[:4])
        while ym(year, last) > latest:  # the most recent season that is complete
            year -= 1
        start = ym(year - 1, 12) if m[2] == "winter" else ym(year, first)
        add(m, start, ym(year, last), m[0],
            f"{m[0].capitalize()} means {month_name(start)} to {month_name(ym(year, last))}, "
            f"the most recent complete {m[2]}.")
    year_to_date = re.search(r"\b(this year|so far this year|year to date|ytd)\b", text)
    since_last_year = re.search(r"\b(since|than|vs\.?|versus|from|(compared |)(with|to)) last year\b", text)
    if since_last_year and not year_to_date and found:
        since_last_year = None  # another period was named; it is compared with a year earlier
    if year_to_date or since_last_year:
        start = f"{latest[:4]}-01"
        span_match = year_to_date or since_last_year
        add(span_match, start, latest, "this year so far",
            f"This year so far means {month_name(start)} to {month_name(latest)}.")
        if since_last_year:
            this_year = found[-1]
            last = this_year.shifted(-12, "the same months last year")
            found.append(Period(last.start, last.end, last.label, this_year.position + 1))
    for m in re.finditer(r"\b(20[12]\d)\b", text):
        if m[1] == latest[:4] and latest[5:] != "12":  # the current year isn't over yet
            add(m, f"{m[1]}-01", latest, f"{m[1]} so far",
                f"{m[1]} means {month_name(m[1] + '-01')} to {month_name(latest)}, the months complete so far.")
        else:
            add(m, f"{m[1]}-01", f"{m[1]}-12", m[1])
    found.sort(key=lambda p: (p.start, p.position))
    unique: list[Period] = []
    for period in found:
        if not any((p.start, p.end) == (period.start, period.end) for p in unique):
            unique.append(period)
    return unique, list(dict.fromkeys(assumptions))


def _capitalized(label: str) -> str:
    return label[:1].upper() + label[1:]


def _period_suggestions(context: RouterContext) -> list[str]:
    latest = context.complete_through
    year = int(latest[:4])
    return [f"{month_name(latest)} vs {month_name(shift_month(latest, -12))}",
            f"summer {year - 1} vs summer {year}" if latest[5:] >= "08" else f"{year - 2} vs {year - 1}",
            f"the last 3 months vs the same months a year before"]


# ------------------------------------------------------------------ categories

def _loose_key(text: str) -> str:
    """spelling_key with plural s removed, so "street conditions" matches "Street Condition"."""
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("S") and not w.endswith("SS") else w
                    for w in spelling_key(text).split())


def _find_categories(text: str, names: list[str], renamed: dict[str, str] | None = None
                     ) -> tuple[list[str], list[str], tuple[str, list[str]] | None]:
    """(category names in the order the question names them, notes, unclear (word, options) or None).

    Multi-word names and old labels match first, then everyday aliases, then
    one-word names, so "traffic noise" is read as vehicle noise, not the
    "Traffic" category. Matched words are blanked out (keeping their length)
    so a later step can't match them again and positions stay comparable.
    """
    hits: list[tuple[int, str]] = []
    notes: list[str] = []
    padded = f" {_loose_key(text)} "
    current = {_loose_key(name): name for name in names if spelling_key(name) not in AMBIGUOUS_NAMES}
    old = {_loose_key(label): name for label, name in (renamed or {}).items()
           if name in names and spelling_key(label) not in AMBIGUOUS_NAMES and _loose_key(label) not in current}
    old_spelling = {_loose_key(label): label for label in (renamed or {})}

    def take(start: int, end: int, name: str) -> None:
        nonlocal padded
        padded = padded[:start] + " " * (end - start) + padded[end:]
        if name not in (n for _, n in hits):
            hits.append((start, name))

    def match(keys: dict[str, str], multi_word: bool, was_renamed: bool = False) -> None:
        for key in sorted(keys, key=len, reverse=True):
            if (" " in key) != multi_word:
                continue
            for found in re.finditer(rf"(?<= ){re.escape(key)}(?= )", padded):
                take(found.start(), found.end(), keys[key])
                if was_renamed:
                    note = f'"{old_spelling[key]}" complaints are now recorded as "{keys[key]}"; both are counted.'
                    if note not in notes:
                        notes.append(note)

    match(current, multi_word=True)
    match(old, multi_word=True, was_renamed=True)
    for pattern, targets in ALIASES.items():
        for found in re.finditer(pattern, padded.lower()):
            for target in targets:
                if target in names:
                    take(found.start(), found.end(), target)
            if pattern == r"\bpotholes?\b" and "Potholes are recorded under Street Condition." not in notes:
                notes.append("Potholes are recorded under Street Condition.")
    match(current, multi_word=False)
    match(old, multi_word=False, was_renamed=True)
    for word, family in FAMILIES.items():
        found = re.search(rf"\b{word}\b", padded.lower())
        if found:
            members = [n for n in names if re.search(rf"\b{family}\b", spelling_key(n))]
            for member in members:
                take(found.start(), found.end(), member)
            notes.append(f'"{family.lower()}" includes every {family.lower()} category: {", ".join(members)}.')
    if not hits:
        for pattern, options in UNCLEAR_WORDS.items():
            found = re.search(pattern, padded.lower())
            if found:
                return [], [], (found[0], [o for o in options if o in names])
    return [name for _, name in sorted(hits, key=lambda hit: hit[0])], notes, None


GENERIC_WORDS = set("""the a an all total overall 311 service many how which what more fewer most number of in for
about with and or any top biggest largest main common type types kind kinds category categories complaint
complaints request requests call calls report reports issue issues problem problems new recent these those
different other city citywide borough boroughs nyc my our their its this that last next past year years month
months season quarter people residents filed made did does do is are was were there here per each every
rise rising increase decrease change changes trend trends compare comparison show tell give me us
have has had been being gone go going went get got be will would could should can may might why when where
who whom whose so as at on to by from than then also just only even still up down out over any
recorded reported monthly yearly annual seasonal weekly daily average typical usual unusual share percent
percentage rate rates count counts volume level levels pattern patterns history historical latest current
previous same similar
""".split())


def _unrecognized_topic(text: str) -> str | None:
    """The topic in "<topic> complaints": the words just before the noun, back to the
    first place, time, number, or generic word ("bike lane" in "how did bike lane complaints")."""
    place_and_time = set(BOROUGHS) | set(MONTHS) | set(SEASONS) | {"bronx", "staten", "island"}
    for match in re.finditer(r"\b(complaints?|requests?|calls|reports|issues|problems)\b", text):
        topic: list[str] = []
        for word in reversed(re.findall(r"[a-z']+|\d+", text[:match.start()])):
            if word in GENERIC_WORDS or word in place_and_time or word.isdigit():
                break
            topic.insert(0, word)
        if topic:
            return " ".join(topic[-3:])
    return None


# ------------------------------------------------------------------ routes

def _compare_route(geography: str, periods: list[Period], categories: list[str], assumptions: list[str],
                   context: RouterContext) -> Route:
    if len(periods) > 2:
        return Route("clarify", message="I compare two periods at a time. Which two?",
                     options=[p.label for p in periods])
    if len(periods) == 1:
        later = periods[0]
        periods = [later.shifted(-12, f"the same months a year earlier"), later]
        assumptions.append(f"{_capitalized(later.label)} is compared with the same months a year earlier.")
    if not periods:
        latest = context.complete_through
        periods = [Period(shift_month(latest, -12), shift_month(latest, -12), "a year earlier", 0),
                   Period(latest, latest, "the latest month", 1)]
        assumptions.append(f"No period was named, so {month_name(latest)} is compared with "
                           f"{month_name(shift_month(latest, -12))}.")
    a, b = sorted(periods, key=lambda p: p.start)
    if a.months != b.months:
        return Route("clarify", message=(
            f"Those periods have different lengths ({a.months} and {b.months} months), so the totals "
            "wouldn't be comparable. Which matching periods should I use?"),
            options=[f"{a.label} vs the same months a year later", f"{b.label} vs the same months a year earlier"])
    if a.end >= b.start:
        return Route("clarify", message="Those periods overlap. Which two separate periods should I compare?")
    if a.start < context.first_month:
        return Route("decline", message=f"The data begins in {month_name(context.first_month)}.")
    params: dict[str, Any] = {"geography": geography, "period_a": a.as_param(), "period_b": b.as_param()}
    if categories:
        params["categories"] = categories[:25]
    return Route("tool", "compare_periods", params, assumptions=assumptions)


def _summary_route(geography: str, periods: list[Period], assumptions: list[str], context: RouterContext) -> Route:
    if periods and periods[-1].months > 1:
        month = periods[-1].end
        assumptions.append(f"A briefing covers one month, so this uses {month_name(month)}.")
    elif periods:
        month = periods[-1].start
    else:
        month = context.complete_through
        assumptions.append(f"No month was named, so this covers {month_name(month)}, the latest complete month.")
    return Route("tool", "month_summary", {"geography": geography, "month": month}, assumptions=assumptions)


def _trend_route(geography: str, periods: list[Period], categories: list[str], assumptions: list[str],
                 context: RouterContext, text: str) -> Route:
    if periods:
        period = {"start": periods[0].start, "end": periods[-1].end}
    else:
        latest = context.complete_through
        period = {"start": shift_month(latest, -11), "end": latest}
        assumptions.append(f"No period was named, so this shows the last 12 months "
                           f"({month_name(period['start'])} to {month_name(latest)}).")
    params: dict[str, Any] = {"geography": geography, "period": period}
    if categories:
        params["categories"] = categories[:10]
        if len(categories) > 10:
            assumptions.append(f"Only the first 10 of {len(categories)} matching complaint types are shown.")
    return Route("tool", "monthly_trend", params, assumptions=assumptions)
