"""
feature_engineering.py

مسئولیت این فایل: تبدیل لاگ‌های خام رویدادمحور (هر سطر = یک رویداد، مثلاً
یک لاگین) به پروفایل رفتاری روزانه‌ی هر کاربر (هر سطر = یک کاربر در یک روز).

طراحی این فایل عمداً بین دو نوع کد فرق می‌گذارد:
    - توابع "خالص" (pure): فقط محاسبه انجام می‌دهند، هیچ فایلی نمی‌خوانند.
      این‌ها بدون نیاز به دیتاست واقعی، با یک DataFrame کوچک قابل تست‌اند.
    - توابع "I/O": فایل از دیسک می‌خوانند (این‌ها را نمی‌شود بدون فایل
      واقعی تست کرد، پس منطق سنگین را داخل‌شان نمی‌نویسیم).
"""

import os

import numpy as np
import pandas as pd

USER_COLUMN = "user"
TIMESTAMP_COLUMN = "date"
DAILY_DATE_COLUMN = "date_only"
EPSILON = 1e-5

LOG_FILES = {
    "logon": "logon.csv",
    "file": "file.csv",
    "http": "http.csv",
    "email": "email.csv",
    "device": "device.csv",
}
OCEAN_COLUMNS = ["O", "C", "E", "A", "N"]
LDAP_COLUMNS = ["department", "role"]


# ============================================================
# بخش ۱: توابع خالص (pure) — قابل تست بدون فایل واقعی
# ============================================================

def add_after_hours_indicator(
    event_records: pd.DataFrame,
    date_column: str = TIMESTAMP_COLUMN,
    start_hour: int = 8,
    end_hour: int = 18,
) -> pd.DataFrame:
    """
    به هر رویداد سه ستون اضافه می‌کند: ساعت وقوع، روز هفته، و اینکه
    آیا خارج از ساعت اداری (قبل از ۸ صبح، بعد از ۶ عصر، یا آخر هفته) بوده.

    این تابع یک کپی جدید برمی‌گرداند و ورودی را دست‌نخورده نگه می‌دارد —
    این کار عمدی است تا فراخوانی این تابع اثر جانبی غیرمنتظره نداشته باشد.
    """
    events = event_records.copy()
    events[date_column] = pd.to_datetime(events[date_column], errors="coerce")
    events["hour"] = events[date_column].dt.hour
    events["day_of_week"] = events[date_column].dt.dayofweek  # 5=شنبه، 6=یکشنبه در pandas پیش‌فرض میلادی

    is_outside_working_hours = (events["hour"] < start_hour) | (events["hour"] >= end_hour)
    is_weekend = events["day_of_week"] >= 5
    events["is_after_hours"] = (is_outside_working_hours | is_weekend).astype(np.int8)
    return events


def to_binary_indicator(text_values: pd.Series) -> pd.Series:
    """
    مقادیر متنی رایج برای «بله/خیر» (مثل '1', 'true', 'yes', 'y') را
    به ۰ و ۱ عددی تبدیل می‌کند. برای ستون‌هایی مثل to_removable_media استفاده می‌شود.
    """
    normalized = text_values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"}).astype(np.int8)


def aggregate_daily_log_chunk(event_chunk: pd.DataFrame, log_type: str) -> pd.DataFrame:
    """
    یک تکه (chunk) از رویدادهای خام یک نوع لاگ خاص را می‌گیرد و
    به سطح «کاربر-روز» جمع می‌بندد.

    خروجی: یک DataFrame با یک سطر به ازای هر (کاربر, روز) که در این
    chunk دیده شده، به همراه ستون‌های شمارشی مخصوص همان نوع لاگ.

    این تابع «خالص» است: هیچ فایلی نمی‌خواند، فقط روی ورودی محاسبه می‌کند —
    برای همین می‌شود آن را با یک DataFrame کوچک ساختگی تست کرد.
    """
    chunk = add_after_hours_indicator(event_chunk)
    chunk = chunk.dropna(subset=[TIMESTAMP_COLUMN]).copy()
    chunk[DAILY_DATE_COLUMN] = chunk[TIMESTAMP_COLUMN].dt.normalize()

    aggregation_rules = {
        f"{log_type}_total": (TIMESTAMP_COLUMN, "count"),
        f"{log_type}_after_hours": ("is_after_hours", "sum"),
    }

    # هر نوع لاگ یک سیگنال رفتاری اضافه‌ی مخصوص خودش دارد
    if log_type == "file" and "to_removable_media" in chunk:
        chunk["is_usb_transfer"] = to_binary_indicator(chunk["to_removable_media"])
        aggregation_rules["file_usb_count"] = ("is_usb_transfer", "sum")

    elif log_type == "http" and "url" in chunk:
        chunk["url_length"] = chunk["url"].fillna("").astype(str).str.len()
        chunk["url_was_observed"] = chunk["url"].notna().astype(np.int8)
        aggregation_rules["http_url_length_sum"] = ("url_length", "sum")
        aggregation_rules["http_url_observation_count"] = ("url_was_observed", "sum")

    elif log_type == "email" and "attachments" in chunk:
        chunk["has_attachment"] = (
            chunk["attachments"].fillna("").astype(str).str.strip().ne("").astype(np.int8)
        )
        aggregation_rules["email_attachments_count"] = ("has_attachment", "sum")

    return chunk.groupby([USER_COLUMN, DAILY_DATE_COLUMN]).agg(**aggregation_rules).reset_index()


def finalize_log_aggregation(chunk_aggregations: list, log_type: str) -> pd.DataFrame | None:
    """
    خروجی‌های جمع‌بندی‌شده‌ی همه‌ی chunkها را با هم ترکیب می‌کند و
    نسبت‌های نهایی (مثل درصد فعالیت خارج از ساعت اداری) را می‌سازد.
    """
    if not chunk_aggregations:
        return None

    daily_aggregation = (
        pd.concat(chunk_aggregations, ignore_index=True)
        .groupby([USER_COLUMN, DAILY_DATE_COLUMN], as_index=False)
        .sum(numeric_only=True)
    )
    daily_aggregation[f"{log_type}_after_hours_ratio"] = (
        daily_aggregation[f"{log_type}_after_hours"] / (daily_aggregation[f"{log_type}_total"] + EPSILON)
    )

    if log_type == "http":
        observed = daily_aggregation["http_url_observation_count"].replace(0, np.nan)
        daily_aggregation["http_avg_url_length"] = (
            daily_aggregation["http_url_length_sum"].div(observed).fillna(0.0)
        )
        daily_aggregation = daily_aggregation.drop(
            columns=["http_url_length_sum", "http_url_observation_count"]
        )
    return daily_aggregation


def complete_daily_calendar(profile_data: pd.DataFrame) -> pd.DataFrame:
    """
    بین اولین و آخرین روزی که هر کاربر فعالیت ثبت‌شده دارد، روزهای خالی
    (بدون هیچ فعالیتی) را هم با مقدار صفر اضافه می‌کند — تا سری زمانی
    هر کاربر پیوسته باشد و sequence_builder بتواند پنجره‌های ۷روزه‌ی
    درست بسازد (بدون این کار، «روز بعدی» در داده لزوماً یک روز واقعی
    تقویمی نیست).
    """
    numeric_columns = profile_data.select_dtypes(include=np.number).columns.tolist()
    profiles = profile_data.copy()
    profiles[DAILY_DATE_COLUMN] = pd.to_datetime(profiles[DAILY_DATE_COLUMN], errors="coerce")
    profiles = profiles.dropna(subset=[DAILY_DATE_COLUMN])

    completed_per_user = []
    for user_id, user_profile in profiles.groupby(USER_COLUMN, sort=False):
        ordered = user_profile.sort_values(DAILY_DATE_COLUMN)
        full_calendar = pd.date_range(
            start=ordered[DAILY_DATE_COLUMN].min(),
            end=ordered[DAILY_DATE_COLUMN].max(),
            freq="D",
        )
        completed = ordered.set_index(DAILY_DATE_COLUMN).reindex(full_calendar)
        completed.index.name = DAILY_DATE_COLUMN
        completed[USER_COLUMN] = user_id
        completed[numeric_columns] = completed[numeric_columns].fillna(0.0)
        completed_per_user.append(completed.reset_index())

    result = pd.concat(completed_per_user, ignore_index=True)
    result[DAILY_DATE_COLUMN] = pd.to_datetime(result[DAILY_DATE_COLUMN]).dt.date
    return result


def add_temporal_baseline_features(profile_data: pd.DataFrame) -> pd.DataFrame:
    """
    برای هر کاربر، میانگین تاریخی رفتارش را حساب می‌کند و انحراف امروز
    از آن میانگین را می‌سازد.

    نکته‌ی حیاتی جلوگیری از leakage: از .shift(1) استفاده می‌شود، یعنی
    میانگین «تا دیروز» محاسبه می‌شود، نه شامل خود امروز. اگر این shift
    نبود، مدل هنگام ارزیابی به مقداری از «آینده‌ی» خودش دسترسی پیدا
    می‌کرد که در دنیای واقعی در آن لحظه هنوز وجود نداشت.
    """
    ordered = profile_data.sort_values([USER_COLUMN, DAILY_DATE_COLUMN]).reset_index(drop=True)
    baseline_targets = {
        "logon_total": "logon_baseline_deviation",
        "file_usb_count": "file_usb_baseline_deviation",
        "email_attachments_count": "email_attachments_baseline_deviation",
    }

    for source_column, deviation_column in baseline_targets.items():
        if source_column not in ordered:
            continue
        historical_mean = ordered.groupby(USER_COLUMN)[source_column].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
        ordered[deviation_column] = (
            ordered[source_column].sub(historical_mean.fillna(ordered[source_column])).abs()
        )

    if "logon_total" in ordered:
        rolling_mean = ordered.groupby(USER_COLUMN)["logon_total"].transform(
            lambda values: values.shift(1).rolling(window=7, min_periods=1).mean()
        )
        ordered["logon_rolling_mean_7d"] = rolling_mean.fillna(0.0)

    return ordered


# ============================================================
# بخش ۲: توابع I/O — خواندن فایل از دیسک (chunk-based)
# ============================================================

def process_large_log_file(file_path, log_type, chunk_size=500_000, max_rows=None):
    """
    یک فایل CSV بزرگ را تکه‌تکه می‌خواند (تا در رم ۱۶ گیگ جا شود) و هر
    تکه را با aggregate_daily_log_chunk جمع می‌بندد.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"فایل لاگ پیدا نشد: {file_path}")

    print(f"  - در حال پردازش {log_type} به‌صورت تکه‌تکه...")
    chunk_aggregations = []
    processed_row_count = 0
    read_chunk_size = min(chunk_size, max_rows) if max_rows is not None else chunk_size

    for event_chunk in pd.read_csv(file_path, chunksize=read_chunk_size, low_memory=False):
        if max_rows is not None:
            remaining = max_rows - processed_row_count
            if remaining <= 0:
                break
            event_chunk = event_chunk.iloc[:remaining].copy()

        processed_row_count += len(event_chunk)
        chunk_aggregations.append(aggregate_daily_log_chunk(event_chunk, log_type))

    return finalize_log_aggregation(chunk_aggregations, log_type)


def build_multisource_behavioral_profiles(base_path, sample_nrows=None) -> pd.DataFrame:
    """
    نقطه‌ی ورود اصلی این فایل: هر ۵ فایل لاگ را می‌خواند، پروفایل روزانه‌ی
    هرکدام را می‌سازد، همه را برای هر (کاربر, روز) با هم ادغام می‌کند،
    تقویم را کامل می‌کند، و در نهایت ویژگی‌های خط‌مبنای تاریخی را اضافه می‌کند.
    """
    merged_profiles = None
    for log_type, file_name in LOG_FILES.items():
        daily_features = process_large_log_file(
            file_path=os.path.join(base_path, file_name),
            log_type=log_type,
            max_rows=sample_nrows,
        )
        if daily_features is None:
            continue
        merged_profiles = (
            daily_features
            if merged_profiles is None
            else pd.merge(merged_profiles, daily_features, on=[USER_COLUMN, DAILY_DATE_COLUMN], how="outer")
        )

    if merged_profiles is None:
        raise ValueError("هیچ پروفایل رفتاری‌ای از فایل‌های لاگ ساخته نشد.")

    numeric_columns = merged_profiles.select_dtypes(include=np.number).columns.tolist()
    merged_profiles[numeric_columns] = merged_profiles[numeric_columns].fillna(0.0)
    return add_temporal_baseline_features(complete_daily_calendar(merged_profiles))


def _load_unique_user_metadata(metadata_file_path, required_columns, source_name):
    """تابع خصوصی: فایل متادیتای سطح-کاربر (نه سطح-روز) را می‌خواند و یک ردیف به ازای هر کاربر برمی‌گرداند."""
    if not os.path.exists(metadata_file_path):
        raise FileNotFoundError(f"فایل {source_name} پیدا نشد: {metadata_file_path}")

    metadata = pd.read_csv(metadata_file_path)
    if "user_id" in metadata:
        metadata = metadata.rename(columns={"user_id": USER_COLUMN})

    missing = set(required_columns).difference(metadata.columns)
    if missing:
        raise ValueError(f"فایل {source_name} این ستون‌ها را کم دارد: {sorted(missing)}")
    return metadata[required_columns].drop_duplicates(USER_COLUMN)


def add_psychometric_features(profile_df, psychometric_file_path):
    """نمرات روان‌شناختی OCEAN را به پروفایل روزانه اضافه می‌کند (مقدار گم‌شده = میانه)."""
    psychometric_data = _load_unique_user_metadata(psychometric_file_path, [USER_COLUMN, *OCEAN_COLUMNS], "Psychometric")
    enriched = pd.merge(profile_df, psychometric_data, on=USER_COLUMN, how="left")
    for column in OCEAN_COLUMNS:
        median_value = enriched[column].median()
        enriched[column] = enriched[column].fillna(0.0 if pd.isna(median_value) else median_value)
    return enriched


def add_ldap_features(profile_df, ldap_file_path):
    """اطلاعات سازمانی (دپارتمان و نقش شغلی) را از LDAP اضافه می‌کند."""
    ldap_data = _load_unique_user_metadata(ldap_file_path, [USER_COLUMN, *LDAP_COLUMNS], "LDAP")
    enriched = pd.merge(profile_df, ldap_data, on=USER_COLUMN, how="left")
    for column in LDAP_COLUMNS:
        enriched[column] = enriched[column].fillna("Unknown")
    return enriched
