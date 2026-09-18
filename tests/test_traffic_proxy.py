"""
Tests for src/build_traffic_proxy.py's synthetic congestion score.

This proxy stands in for real traffic data (see README > Data decisions)
and is used throughout the project's correlation study and every
forecasting model, so its basic shape needs to actually match the
documented Delhi commute pattern it claims to encode -- a weekday
rush-hour score higher than the same hour on a weekend -- and it must
stay within the stated [0, 1] range, since several downstream models and
plots implicitly assume a bounded score.
"""

import pandas as pd

from build_traffic_proxy import (
    traffic_proxy_score,
    WEEKDAY_HOURLY_WEIGHTS,
    WEEKEND_HOURLY_WEIGHTS,
)

DOCUMENTED_RUSH_HOURS = [8, 9, 18, 19]  # per README: AM peak ~8-10, PM peak ~6-9


def test_weekday_rush_hour_score_exceeds_weekend_same_hour():
    """The whole point of a weekday/weekend split is that rush hours are
    sharper on weekdays; if a weekday rush hour ever scored *below* the
    same hour on a weekend, the proxy would contradict the documented
    pattern it's supposed to encode."""
    monday = pd.Timestamp("2023-01-02")  # confirmed Monday (dayofweek == 0)
    saturday = pd.Timestamp("2023-01-07")  # confirmed Saturday (dayofweek == 5)
    assert monday.dayofweek == 0
    assert saturday.dayofweek == 5

    for hour in DOCUMENTED_RUSH_HOURS:
        weekday_score = traffic_proxy_score(monday + pd.Timedelta(hours=hour))
        weekend_score = traffic_proxy_score(saturday + pd.Timedelta(hours=hour))
        assert weekday_score > weekend_score, (
            f"hour {hour}: weekday score {weekday_score} should exceed "
            f"weekend score {weekend_score}"
        )


def test_all_scores_are_within_unit_interval():
    """Every hourly weight in both the weekday and weekend tables -- the
    full domain traffic_proxy_score() can return -- must lie in [0, 1],
    since it's documented and used everywhere as a 0-1 congestion score."""
    all_weights = list(WEEKDAY_HOURLY_WEIGHTS.values()) + list(WEEKEND_HOURLY_WEIGHTS.values())
    assert len(all_weights) == 48  # 24 hours x 2 day-types, sanity-check completeness
    for w in all_weights:
        assert 0.0 <= w <= 1.0


def test_score_dispatches_correctly_by_actual_weekday_flag():
    """traffic_proxy_score() must pick the table matching the timestamp's
    own weekday, not e.g. always default to the weekday table -- checked
    here against the module's own weight dicts rather than hardcoded
    numbers, so this test doesn't silently drift if the weights are
    retuned later."""
    monday_9am = pd.Timestamp("2023-01-02 09:00:00")
    saturday_9am = pd.Timestamp("2023-01-07 09:00:00")

    assert traffic_proxy_score(monday_9am) == WEEKDAY_HOURLY_WEIGHTS[9]
    assert traffic_proxy_score(saturday_9am) == WEEKEND_HOURLY_WEIGHTS[9]
