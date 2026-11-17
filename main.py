"""Hour-by-hour quantile forecast of a launch day's Twitch viewers.

Approach: every comparable past launch contributes the shape of its own
release day, measured in log space against its first hour.  Launches
that opened at a similar scale to this case vote loudest (a log-scale
similarity kernel on the hour-1 print), launches flagged as
early-access-contaminated or weak analogues vote softer, and each
hour's forecast is the weighted median shape with a spread blended
from two views of how much the shapes disagree.  A short moving
average steadies the median shape hour to hour.  The result hangs off
this case's own hour-1 print: the median shape carries the unlock
geography (the early-hours dip, the US-unlock jumps, the evening
peak) and the spread carries honest cross-launch uncertainty.  No
randomness anywhere, so two runs produce identical bytes.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

# Where the grader leaves the case files, and where forecasts go.
INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Fallback levels if a case ever arrived without its own.
DEFAULT_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# A comparable whose release day is missing any hour cannot teach a shape.
DAY_HOURS = 24
# How fast a comparable's vote fades as its first hour drifts from ours,
# in log units: launches that opened at a similar scale teach the most
# about how this day unfolds.
KERNEL_TAU = 1.2
# The raw cross-launch spread runs a little wide; shrink it toward the
# honest band the backtests support.
SPREAD_SHRINK = 0.9
# Smallest spread we ever quote, in log units: even the most predictable
# hour of a launch day moves.
SPREAD_FLOOR = 0.12
# Largest spread we ever quote: beyond this the band stops informing.
SPREAD_CAP = 1.5


def parse_instant(text):
    """An ISO-8601 UTC stamp like 2026-11-19T00:00:00Z as a datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def normal_quantile(prob):
    """Inverse of the standard normal CDF (Acklam's rational approximation).

    Accurate to about 1e-9 across (0, 1); good enough that the choice of
    approximation never moves a quoted quantile by a whole viewer.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError("probability outside (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02,
         -2.759285984469534e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02,
         -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00)
    edge = 0.02425
    if prob < edge:
        r = math.sqrt(-2.0 * math.log(prob))
        return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
            ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
    if prob > 1.0 - edge:
        r = math.sqrt(-2.0 * math.log(1.0 - prob))
        return -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
            ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
    r = prob - 0.5
    s = r * r
    return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / \
        (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1.0)


def weighted_quantile(pairs, prob):
    """Quantile of values carrying weights; pairs are (value, weight)."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        raise ValueError("no weight")
    target = prob * total
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= target:
            return value
    return ordered[-1][0]


def tier_weight(unlock_note):
    """How much a comparable's shape should count, from its research tier.

    Launches whose release day was contaminated by early access, and
    launches flagged as weak analogues, still inform the shape but count
    for less than clean comparable launches.
    """
    note = (unlock_note or "").lower()
    if "early-access-contaminated" in note:
        return 0.4
    if "tier: weak" in note or "tier:weak" in note:
        return 0.25
    return 1.0


def release_day_curve(comparable):
    """The 24 hourly viewer counts of a comparable's release day, or None.

    Only complete days teach a shape: a day with a missing hour would
    silently shift every later hour's lesson.
    """
    try:
        start = parse_instant(comparable["day_start"])
    except (KeyError, ValueError, TypeError):
        return None
    by_hour = {}
    for point in comparable.get("hourly", []):
        try:
            at = parse_instant(point["from"])
            by_hour[at] = point["viewers"]
        except (KeyError, ValueError, TypeError):
            continue
    curve = []
    for step in range(DAY_HOURS):
        value = by_hour.get(start + timedelta(hours=step))
        if value is None:
            return None
        curve.append(max(1.0, float(value)))
    if curve[0] <= 0:
        return None
    return curve


def comparable_lessons(comparables):
    """Each usable comparable as log-curves plus its standing weight.

    A lesson is (log hourly curve, log first hour, tier weight).  The
    similarity kernel is applied later, once this case's anchor is known.
    """
    lessons = []
    for comparable in comparables or []:
        curve = release_day_curve(comparable)
        if curve is None:
            continue
        log_curve = [math.log(value) for value in curve]
        lessons.append((log_curve, log_curve[0],
                        tier_weight(comparable.get("unlock", ""))))
    return lessons


def hour_lesson_pairs(lessons, hour, anchor_log):
    """This hour's shape votes: (log ratio to hour 1, kernel weight)."""
    pairs = []
    for log_curve, log_first, standing in lessons:
        gap = log_first - anchor_log
        weight = standing * math.exp(-(gap * gap) / (2.0 * KERNEL_TAU ** 2))
        pairs.append((log_curve[hour - 1] - log_first, weight))
    return pairs


def blended_spread(pairs):
    """One spread from two views of the same disagreement.

    The interquartile range and the 10-90 range each estimate the log
    spread of the shape votes; averaging them is steadier than either
    alone when only a dozen or so launches carry real weight.
    """
    iqr = (weighted_quantile(pairs, 0.75) - weighted_quantile(pairs, 0.25)) / 1.349
    wide = (weighted_quantile(pairs, 0.90) - weighted_quantile(pairs, 0.10)) / 2.563
    return min(max(0.5 * (iqr + wide) * SPREAD_SHRINK, SPREAD_FLOOR), SPREAD_CAP)


def smooth_centres(centres):
    """A 3-hour moving average over the median shape, edges included.

    Launch-day shapes move smoothly hour to hour; averaging neighbours
    removes single-hour wobble from whichever launches happen to dominate
    a given hour's vote.
    """
    smoothed = {}
    for hour in range(2, DAY_HOURS + 1):
        neighbours = [centres[h] for h in (hour - 1, hour, hour + 1)
                      if 2 <= h <= DAY_HOURS]
        smoothed[hour] = sum(neighbours) / len(neighbours)
    return smoothed


def round_half_up_cents(value):
    """Round to 2 decimals, halves away from zero, as a plain float."""
    return math.floor(value * 100.0 + 0.5) / 100.0


def tidy_quantiles(values):
    """Clamp to the legal band and force non-decreasing order."""
    tidy = []
    floor = 0.0
    for value in values:
        clamped = min(max(value, 0.01), 1e9)
        clamped = max(clamped, floor)
        tidy.append(round_half_up_cents(clamped))
        floor = tidy[-1]
    return tidy


def shape_forecast(first_print, lessons, levels):
    """Median-shape forecast with a dispersion spread, per hour."""
    z_scores = [normal_quantile(level) for level in levels]
    anchor_log = math.log(first_print)
    centres = {}
    spreads = {}
    for h in range(2, DAY_HOURS + 1):
        pairs = hour_lesson_pairs(lessons, h, anchor_log)
        if pairs:
            centres[h] = weighted_quantile(pairs, 0.5)
            spreads[h] = blended_spread(pairs)
        else:
            centres[h] = 0.0
            spreads[h] = 0.30 * math.sqrt(h - 1)
    centres = smooth_centres(centres)
    rows = []
    for h in range(2, DAY_HOURS + 1):
        centre_value = first_print * math.exp(centres[h])
        rows.append({"hour": h, "q": tidy_quantiles(
            [centre_value * math.exp(z * spreads[h]) for z in z_scores])})
    return {"hours": rows}


def fallback_forecast(first_print, levels):
    """Persistence with a widening band, used only when nothing else works."""
    z_scores = [normal_quantile(level) for level in levels]
    rows = []
    for h in range(2, DAY_HOURS + 1):
        spread = 0.30 * math.sqrt(h - 1)
        rows.append({"hour": h, "q": tidy_quantiles(
            [first_print * math.exp(z * spread) for z in z_scores])})
    return {"hours": rows}


def first_hour_print(case):
    """Hour 1's print: the known row, else the latest history point, else 1."""
    for row in case.get("known", []) or []:
        if row.get("hour") == 1:
            try:
                return max(1.0, float(row["viewers"]))
            except (KeyError, TypeError, ValueError):
                pass
    history = case.get("history_hourly") or case.get("history") or []
    if history:
        try:
            return max(1.0, float(history[-1]["viewers"]))
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    return 1.0


def forecast_case(case):
    """One case's forecast; never raises, so a case is never invalid."""
    levels = case.get("levels") or list(DEFAULT_LEVELS)
    try:
        levels = [float(level) for level in levels]
        if len(levels) != 5 or any(not 0.0 < lv < 1.0 for lv in levels):
            levels = list(DEFAULT_LEVELS)
    except (TypeError, ValueError):
        levels = list(DEFAULT_LEVELS)
    anchor = first_hour_print(case)
    try:
        lessons = comparable_lessons(case.get("comparables"))
        if lessons:
            return shape_forecast(anchor, lessons, levels)
    except Exception:
        pass
    return fallback_forecast(anchor, levels)


def run(input_dir, output_dir):
    """Forecast every case file in name order, one output per case.

    A file that cannot be read or parsed still gets a well-formed
    fallback forecast under its own stem: a missing output is scored
    far worse than a cautious one, and one bad file must never take
    down the rest of the run.
    """
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir) if name.endswith(".in"))
    for name in names:
        stem = name[:-3]
        try:
            with open(os.path.join(input_dir, name), encoding="utf-8") as handle:
                case = json.load(handle)
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_forecast(1.0, list(DEFAULT_LEVELS))
        payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        with open(os.path.join(output_dir, stem + ".out"), "w", encoding="utf-8") as handle:
            handle.write(payload)


def main():
    run(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()
