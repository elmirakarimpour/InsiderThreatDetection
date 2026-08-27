"""
test_feature_engineering.py

فقط توابع «خالص» (pure) تست می‌شوند — چون توابع I/O به فایل واقعی روی
دیسک نیاز دارند و در این مرحله دیتاست واقعی هنوز به پروژه وصل نشده.
"""

import numpy as np
import pandas as pd

from src.data.feature_engineering import (
    add_after_hours_indicator,
    add_temporal_baseline_features,
    aggregate_daily_log_chunk,
    complete_daily_calendar,
    to_binary_indicator,
)


def test_after_hours_flag_true_before_start_hour():
    """رویداد ساعت ۳ بامداد باید is_after_hours=1 بگیرد."""
    events = pd.DataFrame({"date": ["2011-01-03 03:00:00"]})  # ۲۰۱۱-۰۱-۰۳ دوشنبه است
    result = add_after_hours_indicator(events)
    assert result["is_after_hours"].iloc[0] == 1


def test_after_hours_flag_false_during_work_hours():
    """رویداد ساعت ۱۰ صبح یک روز کاری باید is_after_hours=0 بگیرد."""
    events = pd.DataFrame({"date": ["2011-01-03 10:00:00"]})
    result = add_after_hours_indicator(events)
    assert result["is_after_hours"].iloc[0] == 0


def test_after_hours_flag_true_on_weekend():
    """رویداد ساعت ۱۰ صبح یک شنبه (تعطیل، weekday=5) هم باید ۱ باشد، چون آخر هفته است."""
    events = pd.DataFrame({"date": ["2011-01-08 10:00:00"]})  # ۲۰۱۱-۰۱-۰۸ شنبه است
    result = add_after_hours_indicator(events)
    assert result["is_after_hours"].iloc[0] == 1


def test_binary_indicator_recognizes_common_true_values():
    values = pd.Series(["1", "True", "yes", "0", "no", None])
    result = to_binary_indicator(values)
    assert result.tolist() == [1, 1, 1, 0, 0, 0]


def test_aggregate_daily_log_chunk_counts_events_per_user_per_day():
    """دو رویداد از یک کاربر در یک روز باید در یک سطر با total=2 جمع شوند."""
    chunk = pd.DataFrame({
        "user": ["alice", "alice", "bob"],
        "date": ["2011-01-03 09:00:00", "2011-01-03 20:00:00", "2011-01-03 09:00:00"],
    })
    result = aggregate_daily_log_chunk(chunk, log_type="logon")
    alice_row = result[result["user"] == "alice"].iloc[0]
    assert alice_row["logon_total"] == 2
    assert alice_row["logon_after_hours"] == 1  # فقط رویداد ساعت ۲۰ خارج از ساعت اداری است


def test_complete_daily_calendar_fills_missing_days_with_zero():
    """اگر کاربری روز ۱ و روز ۳ فعالیت داشته، روز ۲ باید با صفر اضافه شود."""
    profile = pd.DataFrame({
        "user": ["alice", "alice"],
        "date_only": pd.to_datetime(["2011-01-01", "2011-01-03"]),
        "logon_total": [5, 7],
    })
    result = complete_daily_calendar(profile)
    assert len(result) == 3  # روز ۱، ۲، ۳
    middle_day = result[result["date_only"] == pd.Timestamp("2011-01-02").date()]
    assert middle_day["logon_total"].iloc[0] == 0.0


def test_temporal_baseline_excludes_current_day_no_leakage():
    """
    تست حیاتی ضد-leakage: مقدار baseline روز دوم نباید شامل مقدار
    خود روز دوم باشد، فقط باید بر اساس روز اول (گذشته) حساب شود.
    """
    profile = pd.DataFrame({
        "user": ["alice", "alice"],
        "date_only": [1, 2],
        "logon_total": [10, 100],  # جهش بزرگ در روز دوم
    })
    result = add_temporal_baseline_features(profile)
    day_two = result[result["date_only"] == 2].iloc[0]
    # baseline روز دوم باید فقط بر اساس روز اول (۱۰) باشد، نه میانگین ۱۰ و ۱۰۰
    assert day_two["logon_baseline_deviation"] == abs(100 - 10)
