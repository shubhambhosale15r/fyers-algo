"""
FYERS NIFTY ATM CE/PE 2x Option Backtest
=========================================

Strategy
--------
For every NIFTY trading day:

1. Get NIFTY underlying 09:15 open.
2. Find the NIFTY strike closest to that open.
3. Find the nearest weekly expiry available for that day.
4. Get the ATM CE and ATM PE for that expiry.
5. Record each option's 09:15 open.
6. Independently monitor CE and PE:
       Entry  = 2.0 x option_open
       SL     = 1.5 x option_open
       Target = 3.0 x option_open

7. When price reaches Entry:
       Enter LONG option.

8. After entry:
       SL     = 1.5 x option_open
       Target = 3.0 x option_open

9. CE and PE are traded independently.
10. If both SL and Target are touched in the same 1-minute candle,
    SL is assumed to occur first (conservative assumption).
11. No new entry after ENTRY_CUTOFF.
12. Any open position is force-closed at EOD.

Output
------
Excel workbook containing:
    - Trades
    - Daily Summary
    - Statistics
    - Parameters
    - Errors

Requirements
------------
pip install fyers-apiv3 pandas openpyxl requests python-dateutil

Authentication
--------------
Set environment variables:

Windows PowerShell:
    $env:FYERS_CLIENT_ID="YOUR_CLIENT_ID"
    $env:FYERS_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"

Linux/macOS:
    export FYERS_CLIENT_ID="YOUR_CLIENT_ID"
    export FYERS_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"

Then:
    python nifty_2x_backtest.py

NOTE
----
FYERS API details can change. If your installed fyers-apiv3 version has
different method names, update the two wrapper functions:

    get_expiry_dates()
    get_expired_contracts()

The rest of the backtest engine is independent of the API.
"""

from __future__ import annotations

import os
import sys
import time
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, date, time as dt_time, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print("ERROR: fyers-apiv3 is not installed.")
    print("Run: pip install fyers-apiv3")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")

if not CLIENT_ID or not ACCESS_TOKEN:
    raise RuntimeError(
        "\nMissing FYERS credentials.\n\n"
        "Set:\n"
        "  FYERS_CLIENT_ID\n"
        "  FYERS_ACCESS_TOKEN\n\n"
        "Do NOT hard-code your access token in this script.\n"
    )


# Backtest period
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# Underlying
UNDERLYING = "NSE:NIFTY50-INDEX"

# Exchange/symbol components used when constructing option symbols
OPTION_PREFIX = "NSE:NIFTY"

# Candle resolution
RESOLUTION = "1"

# Strategy parameters
ENTRY_MULTIPLE = 2.0
SL_MULTIPLE = 1.5
TARGET_MULTIPLE = 3.0

# Market times
MARKET_OPEN = dt_time(9, 15)

# We use the 09:15 candle's open as the option open.
# Entry monitoring begins from the 09:15 candle onward.
ENTRY_START = dt_time(9, 15)

# Do not initiate a new position after this time.
ENTRY_CUTOFF = dt_time(15, 15)

# Force-close open positions at this time.
FORCE_EXIT_TIME = dt_time(15, 29)

# Conservative same-candle assumption:
# if both SL and target are hit, SL wins.
SAME_CANDLE_POLICY = "SL_FIRST"

# Number of days requested per API call.
# FYERS documents a 100-day maximum for intraday historical requests.
API_CHUNK_DAYS = 90

# Pause between API requests to reduce the chance of rate limiting.
API_SLEEP_SECONDS = 0.20

# Output
OUTPUT_FILE = "NIFTY_2X_Backtest_2024_2025.xlsx"

# Optional:
# If True, save raw downloaded candles as CSV files.
SAVE_RAW_DATA = False
RAW_DATA_DIR = "fyers_raw_data"


# ============================================================
# FYERS CLIENT
# ============================================================

fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    is_async=False,
    token=ACCESS_TOKEN,
    log_path=""
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Trade:
    trade_date: str
    option_type: str
    symbol: str
    expiry: str
    strike: float

    option_open: float

    entry_price: float
    stop_loss: float
    target: float

    entry_time: Optional[str]
    exit_time: Optional[str]

    exit_price: float
    exit_reason: str

    quantity: int
    gross_pnl: float
    pnl_percent: float

    underlying_open: float


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def parse_epoch_to_datetime(ts: int) -> datetime:
    """
    Convert epoch seconds returned by FYERS into local datetime.

    FYERS NSE timestamps are expected to represent Indian market time.
    """
    return datetime.fromtimestamp(ts)


def safe_float(value) -> Optional[float]:
    try:
        x = float(value)

        if not math.isfinite(x):
            return None

        return x

    except (TypeError, ValueError):
        return None


def daterange(start: date, end: date):
    current = start

    while current <= end:
        yield current
        current += timedelta(days=1)


def is_weekday(d: date) -> bool:
    return d.weekday() < 5


def nearest_strike(price: float, strikes: List[float]) -> float:
    """
    Find strike closest to underlying open.

    In the extremely unlikely event of an exact tie, the lower
    strike is selected.
    """
    return min(
        strikes,
        key=lambda x: (abs(x - price), x)
    )


def make_chunks(
    start_date: date,
    end_date: date,
    chunk_days: int = API_CHUNK_DAYS
):
    """
    Split historical requests into chunks.
    """

    current = start_date

    while current <= end_date:
        chunk_end = min(
            current + timedelta(days=chunk_days - 1),
            end_date
        )

        yield current, chunk_end

        current = chunk_end + timedelta(days=1)


# ============================================================
# FYERS API HELPERS
# ============================================================

def api_call_with_retry(
    function,
    data: dict,
    retries: int = 3
):
    """
    Generic retry wrapper.
    """

    last_exception = None

    for attempt in range(1, retries + 1):

        try:

            response = function(data)

            if response is None:
                raise RuntimeError("FYERS returned None")

            return response

        except Exception as exc:

            last_exception = exc

            print(
                f"API error attempt {attempt}/{retries}: {exc}"
            )

            if attempt < retries:
                time.sleep(attempt * 2)

    raise last_exception


# ============================================================
# GET NIFTY UNDERLYING DATA
# ============================================================

def get_underlying_history(
    start_date: date,
    end_date: date
) -> pd.DataFrame:

    """
    Download NIFTY 1-minute candles.

    We only need the daily 09:15 opening candle for determining
    the ATM strike, but downloading 1-minute data also makes it
    possible to verify trading days.
    """

    all_rows = []

    for chunk_start, chunk_end in make_chunks(
        start_date,
        end_date
    ):

        print(
            f"Downloading NIFTY underlying: "
            f"{chunk_start} -> {chunk_end}"
        )

        data = {
            "symbol": UNDERLYING,
            "resolution": RESOLUTION,
            "date_format": "1",
            "range_from": chunk_start.strftime("%Y-%m-%d"),
            "range_to": chunk_end.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }

        response = api_call_with_retry(
            fyers.history,
            data
        )

        if response.get("s") != "ok":

            print(
                "Underlying response:",
                response
            )

            time.sleep(API_SLEEP_SECONDS)
            continue

        for candle in response.get("candles", []):

            if len(candle) < 5:
                continue

            ts = int(candle[0])

            dt = parse_epoch_to_datetime(ts)

            all_rows.append({
                "timestamp": ts,
                "datetime": dt,
                "date": dt.date(),
                "time": dt.time(),
                "open": safe_float(candle[1]),
                "high": safe_float(candle[2]),
                "low": safe_float(candle[3]),
                "close": safe_float(candle[4]),
                "volume": safe_float(candle[5]) if len(candle) > 5 else None
            })

        time.sleep(API_SLEEP_SECONDS)

    if not all_rows:
        raise RuntimeError(
            "No NIFTY underlying data returned."
        )

    df = pd.DataFrame(all_rows)

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values("timestamp")

    # We only need 09:15.
    opening = df[
        df["time"].apply(
            lambda x: x.hour == 9 and x.minute == 15
        )
    ].copy()

    opening = opening[
        opening["open"].notna()
    ]

    opening = opening.drop_duplicates(
        subset=["date"],
        keep="first"
    )

    return opening.reset_index(drop=True)


# ============================================================
# EXPIRY DISCOVERY
# ============================================================

def get_expiry_dates(
    symbol: str,
    start_date: date,
    end_date: date
) -> List[date]:

    """
    Get available expiry dates from FYERS.

    IMPORTANT:
    FYERS API versions have changed endpoint wrappers over time.

    This function tries the common FYERS v3 method first.

    If your installed SDK exposes expiry information differently,
    this is the function to adapt.
    """

    expiry_dates = []

    # --------------------------------------------------------
    # Attempt 1: history_expiry_dates
    # --------------------------------------------------------

    if hasattr(fyers, "get_expiry_dates"):

        for chunk_start, chunk_end in make_chunks(
            start_date,
            end_date
        ):

            data = {
                "symbol": symbol,
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d")
            }

            try:

                response = api_call_with_retry(
                    fyers.get_expiry_dates,
                    data
                )

                expiry_dates.extend(
                    extract_expiry_dates(response)
                )

            except Exception as exc:

                print(
                    "get_expiry_dates failed:",
                    exc
                )

    # --------------------------------------------------------
    # Attempt 2: history_underlying_symbols
    #
    # We discover expiries from actual contracts if needed.
    # --------------------------------------------------------

    if not expiry_dates:

        print(
            "\nWARNING: Direct expiry-date endpoint was not "
            "available through this SDK."
        )

        print(
            "Expiry dates will instead be discovered from "
            "contract metadata."
        )

    return sorted(set(expiry_dates))


def extract_expiry_dates(response) -> List[date]:

    """
    Extract dates from several possible FYERS response shapes.
    """

    results = []

    if not isinstance(response, dict):
        return results

    # Common possible structures
    candidates = []

    data = response.get("data")

    if isinstance(data, dict):

        for key in [
            "expiry_dates",
            "expiryDates",
            "expiry",
            "expiries",
            "dates"
        ]:

            value = data.get(key)

            if isinstance(value, list):
                candidates.extend(value)

    if isinstance(response.get("expiry_dates"), list):
        candidates.extend(
            response["expiry_dates"]
        )

    if isinstance(response.get("expiryDates"), list):
        candidates.extend(
            response["expiryDates"]
        )

    for value in candidates:

        if isinstance(value, dict):

            for key in [
                "expiry_date",
                "expiryDate",
                "date"
            ]:

                if key in value:
                    value = value[key]
                    break

        try:

            if isinstance(value, (int, float)):

                # Handle epoch milliseconds/seconds
                v = float(value)

                if v > 10_000_000_000:
                    v /= 1000

                d = datetime.fromtimestamp(v).date()

            else:

                text = str(value)[:10]

                d = datetime.strptime(
                    text,
                    "%Y-%m-%d"
                ).date()

            results.append(d)

        except Exception:
            pass

    return results


# ============================================================
# CONTRACT DISCOVERY
# ============================================================

def get_contracts_for_expiry(
    trade_date: date,
    expiry_date: date
) -> List[str]:

    """
    Ask FYERS for NIFTY option contracts for the given expiry.

    Uses history_underlying_symbols when available.
    """

    data = {
        "symbol": UNDERLYING,
        "expiry_date": expiry_date.strftime("%Y-%m-%d")
    }

    if not hasattr(
        fyers,
        "history_underlying_symbols"
    ):
        raise RuntimeError(
            "Installed fyers-apiv3 does not expose "
            "history_underlying_symbols(). "
            "Please update fyers-apiv3."
        )

    response = api_call_with_retry(
        fyers.history_underlying_symbols,
        data
    )

    if response.get("s") not in ("ok", None):

        # Some SDK versions use "code" instead.
        if response.get("code") != 200:
            raise RuntimeError(
                f"Contract discovery failed: {response}"
            )

    data_response = response.get(
        "data",
        {}
    )

    contracts = data_response.get(
        "contracts",
        {}
    )

    options = contracts.get(
        "options",
        []
    )

    return options


def parse_option_symbol(symbol: str):
    """
    Parse NIFTY option symbol.

    Examples:
        NSE:NIFTY25MAR23000CE
        NSE:NIFTY25MAR23000PE
    """

    text = symbol.split(":")[-1]

    if not text.startswith("NIFTY"):
        return None

    if text.endswith("CE"):
        option_type = "CE"
        body = text[:-2]

    elif text.endswith("PE"):
        option_type = "PE"
        body = text[:-2]

    else:
        return None

    # Find trailing strike digits.
    digits = ""

    for char in reversed(body):

        if char.isdigit():
            digits = char + digits

        else:
            break

    if not digits:
        return None

    strike = float(digits)

    return {
        "symbol": symbol,
        "option_type": option_type,
        "strike": strike
    }


def build_contract_map(
    contracts: List[str]
) -> Dict[float, Dict[str, str]]:

    """
    Returns:

        {
            22000: {
                "CE": "NSE:NIFTY...",
                "PE": "NSE:NIFTY..."
            }
        }
    """

    result = {}

    for symbol in contracts:

        parsed = parse_option_symbol(symbol)

        if parsed is None:
            continue

        strike = parsed["strike"]
        option_type = parsed["option_type"]

        if strike not in result:
            result[strike] = {}

        result[strike][option_type] = symbol

    return result


# ============================================================
# NEAREST EXPIRY
# ============================================================

def find_nearest_weekly_expiry(
    trade_date: date,
    expiry_dates: List[date]
) -> Optional[date]:

    """
    Select the nearest expiry on or after the trading date.

    The returned expiry must be no more than 7 calendar days
    from the trading date for this strategy's weekly-expiry
    assumption.

    If FYERS returns all NIFTY expiries, this naturally selects
    the next available expiry.
    """

    candidates = [
        x
        for x in expiry_dates
        if x >= trade_date
    ]

    if not candidates:
        return None

    candidates.sort()

    for expiry in candidates:

        days = (expiry - trade_date).days

        if days <= 7:
            return expiry

    return None


# ============================================================
# OPTION HISTORICAL DATA
# ============================================================

def get_option_history(
    symbol: str,
    trade_date: date
) -> pd.DataFrame:

    """
    Download one trading day's 1-minute option data.
    """

    data = {
        "symbol": symbol,
        "resolution": RESOLUTION,
        "date_format": "1",
        "range_from": trade_date.strftime("%Y-%m-%d"),
        "range_to": trade_date.strftime("%Y-%m-%d"),
        "include_oi": "1"
    }

    response = api_call_with_retry(
        fyers.fno_historical_data,
        data
    )

    if response.get("s") != "ok":

        return pd.DataFrame()

    rows = []

    for candle in response.get(
        "candles",
        []
    ):

        if len(candle) < 5:
            continue

        timestamp = int(candle[0])

        dt = parse_epoch_to_datetime(
            timestamp
        )

        rows.append({
            "timestamp": timestamp,
            "datetime": dt,
            "date": dt.date(),
            "time": dt.time(),
            "open": safe_float(candle[1]),
            "high": safe_float(candle[2]),
            "low": safe_float(candle[3]),
            "close": safe_float(candle[4]),
            "volume": safe_float(candle[5]) if len(candle) > 5 else None,
            "oi": safe_float(candle[6]) if len(candle) > 6 else None
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "timestamp"
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    return df.reset_index(drop=True)


# ============================================================
# FIND 09:15 OPTION OPEN
# ============================================================

def get_0915_open(
    df: pd.DataFrame
) -> Optional[float]:

    if df.empty:
        return None

    rows = df[
        df["time"].apply(
            lambda x:
                x.hour == 9 and x.minute == 15
        )
    ]

    if rows.empty:
        return None

    value = rows.iloc[0]["open"]

    return safe_float(value)


# ============================================================
# BACKTEST ONE OPTION
# ============================================================

def backtest_option(
    df: pd.DataFrame,
    trade_date: date,
    option_type: str,
    symbol: str,
    expiry: date,
    strike: float,
    underlying_open: float
) -> Optional[Trade]:

    """
    Backtest one option for one day.

    Rules:

        option_open = 09:15 option open

        entry = option_open * 2

        SL = option_open * 1.5

        target = option_open * 3

    Entry is triggered when candle high reaches entry.

    After entry:

        if low <= SL and high >= target:
            SL wins

        elif low <= SL:
            SL

        elif high >= target:
            TARGET

    Any open trade is force-closed at FORCE_EXIT_TIME
    using that candle's close, or the last available close.
    """

    if df.empty:
        return None

    option_open = get_0915_open(df)

    if option_open is None:
        return None

    if option_open <= 0:
        return None

    entry_price = option_open * ENTRY_MULTIPLE
    stop_loss = option_open * SL_MULTIPLE
    target = option_open * TARGET_MULTIPLE

    # --------------------------------------------------------
    # Only candles from 09:15 onward
    # --------------------------------------------------------

    intraday = df[
        df["time"] >= ENTRY_START
    ].copy()

    if intraday.empty:
        return None

    # --------------------------------------------------------
    # Entry search
    # --------------------------------------------------------

    entry_row = None

    for _, row in intraday.iterrows():

        candle_time = row["time"]

        if candle_time > ENTRY_CUTOFF:
            break

        high = safe_float(row["high"])

        if high is None:
            continue

        if high >= entry_price:

            entry_row = row
            break

    # No entry
    if entry_row is None:
        return None

    entry_time = entry_row["datetime"]

    # --------------------------------------------------------
    # Exit search
    # --------------------------------------------------------

    after_entry = intraday[
        intraday["timestamp"]
        >= entry_row["timestamp"]
    ].copy()

    exit_price = None
    exit_time = None
    exit_reason = None

    for _, row in after_entry.iterrows():

        candle_time = row["time"]

        high = safe_float(row["high"])
        low = safe_float(row["low"])

        if high is None or low is None:
            continue

        # Force exit
        if candle_time >= FORCE_EXIT_TIME:

            close = safe_float(row["close"])

            if close is not None:
                exit_price = close
                exit_time = row["datetime"]
                exit_reason = "EOD"
                break

        hit_sl = low <= stop_loss
        hit_target = high >= target

        # Both touched within same candle
        if hit_sl and hit_target:

            if SAME_CANDLE_POLICY == "SL_FIRST":

                exit_price = stop_loss
                exit_time = row["datetime"]
                exit_reason = "SL"

            else:

                exit_price = target
                exit_time = row["datetime"]
                exit_reason = "TARGET"

            break

        # SL only
        if hit_sl:

            exit_price = stop_loss
            exit_time = row["datetime"]
            exit_reason = "SL"
            break

        # Target only
        if hit_target:

            exit_price = target
            exit_time = row["datetime"]
            exit_reason = "TARGET"
            break

    # --------------------------------------------------------
    # No explicit exit found -> use last available candle
    # --------------------------------------------------------

    if exit_price is None:

        last = after_entry.iloc[-1]

        close = safe_float(last["close"])

        if close is None:
            return None

        exit_price = close
        exit_time = last["datetime"]
        exit_reason = "LAST_CANDLE"

    # --------------------------------------------------------
    # P&L
    # --------------------------------------------------------

    gross_pnl = exit_price - entry_price

    pnl_percent = (
        gross_pnl / entry_price
    ) * 100.0

    return Trade(
        trade_date=trade_date.strftime("%Y-%m-%d"),
        option_type=option_type,
        symbol=symbol,
        expiry=expiry.strftime("%Y-%m-%d"),
        strike=strike,

        option_open=option_open,

        entry_price=entry_price,
        stop_loss=stop_loss,
        target=target,

        entry_time=entry_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ) if entry_time else None,

        exit_time=exit_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ) if exit_time else None,

        exit_price=exit_price,
        exit_reason=exit_reason,

        quantity=1,

        gross_pnl=gross_pnl,
        pnl_percent=pnl_percent,

        underlying_open=underlying_open
    )


# ============================================================
# DISCOVER EXPIRY FROM CONTRACT DATA
# ============================================================

def discover_expiry_for_day(
    trade_date: date,
    expiry_cache: Dict[date, date],
    known_expiries: List[date]
) -> Optional[date]:

    """
    Use cached expiry dates where possible.

    If direct expiry discovery did not return anything, we make
    a best-effort search by trying the next several Thursdays.

    This fallback exists because FYERS SDK versions differ in
    how expiry-date discovery is exposed.
    """

    if trade_date in expiry_cache:
        return expiry_cache[trade_date]

    expiry = find_nearest_weekly_expiry(
        trade_date,
        known_expiries
    )

    if expiry is not None:

        expiry_cache[trade_date] = expiry

        return expiry

    # --------------------------------------------------------
    # Fallback:
    # Try next 7 calendar days for an expiry.
    # --------------------------------------------------------

    for offset in range(0, 8):

        candidate = (
            trade_date +
            timedelta(days=offset)
        )

        # Weekly NIFTY expiry is normally Thursday.
        # We don't hard-code this as the only condition because
        # exchange calendars/holidays can move the actual expiry.
        if candidate.weekday() != 3:
            continue

        try:

            contracts = get_contracts_for_expiry(
                trade_date,
                candidate
            )

            if contracts:

                expiry_cache[trade_date] = candidate

                return candidate

        except Exception as exc:

            print(
                f"Expiry fallback failed for "
                f"{candidate}: {exc}"
            )

    return None


# ============================================================
# EXCEL REPORT
# ============================================================

def make_statistics(
    trades_df: pd.DataFrame
) -> pd.DataFrame:

    if trades_df.empty:

        return pd.DataFrame({
            "Metric": [
                "Total Trades"
            ],
            "Value": [
                0
            ]
        })

    pnl = trades_df["gross_pnl"]

    wins = trades_df[
        trades_df["gross_pnl"] > 0
    ]

    losses = trades_df[
        trades_df["gross_pnl"] < 0
    ]

    total_trades = len(trades_df)

    win_rate = (
        len(wins) / total_trades * 100
        if total_trades
        else 0
    )

    gross_profit = wins["gross_pnl"].sum()

    gross_loss = abs(
        losses["gross_pnl"].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    cumulative = pnl.cumsum()

    running_max = cumulative.cummax()

    drawdown = cumulative - running_max

    max_drawdown = drawdown.min()

    stats = [
        ("Total Trades", total_trades),

        ("Winning Trades", len(wins)),

        ("Losing Trades", len(losses)),

        ("Win Rate %", win_rate),

        ("Total P&L", pnl.sum()),

        ("Average Trade P&L", pnl.mean()),

        ("Median Trade P&L", pnl.median()),

        ("Average Winner",
         wins["gross_pnl"].mean()
         if not wins.empty else 0),

        ("Average Loser",
         losses["gross_pnl"].mean()
         if not losses.empty else 0),

        ("Largest Winner",
         pnl.max()),

        ("Largest Loser",
         pnl.min()),

        ("Gross Profit",
         gross_profit),

        ("Gross Loss",
         -gross_loss),

        ("Profit Factor",
         profit_factor),

        ("Max Drawdown",
         max_drawdown),

        ("Total Return on Entry Premium",
         (
             pnl.sum() /
             trades_df["entry_price"].sum() *
             100
         )
         if trades_df["entry_price"].sum() != 0
         else 0
        )
    ]

    return pd.DataFrame(
        stats,
        columns=["Metric", "Value"]
    )


def make_daily_summary(
    trades_df: pd.DataFrame
) -> pd.DataFrame:

    if trades_df.empty:
        return pd.DataFrame()

    summary = (
        trades_df
        .groupby("trade_date")
        .agg(
            Trades=("gross_pnl", "count"),
            PnL=("gross_pnl", "sum"),
            Winners=(
                "gross_pnl",
                lambda x: (x > 0).sum()
            ),
            Losers=(
                "gross_pnl",
                lambda x: (x < 0).sum()
            ),
            CE_PnL=(
                "gross_pnl",
                lambda x: 0
            )
        )
        .reset_index()
    )

    # Proper CE/PE breakdown
    ce = (
        trades_df[
            trades_df["option_type"] == "CE"
        ]
        .groupby("trade_date")["gross_pnl"]
        .sum()
        .rename("CE_PnL")
    )

    pe = (
        trades_df[
            trades_df["option_type"] == "PE"
        ]
        .groupby("trade_date")["gross_pnl"]
        .sum()
        .rename("PE_PnL")
    )

    summary = summary.drop(
        columns=["CE_PnL"]
    )

    summary = summary.merge(
        ce,
        on="trade_date",
        how="left"
    )

    summary = summary.merge(
        pe,
        on="trade_date",
        how="left"
    )

    summary[
        ["CE_PnL", "PE_PnL"]
    ] = summary[
        ["CE_PnL", "PE_PnL"]
    ].fillna(0)

    summary["Cumulative_PnL"] = (
        summary["PnL"].cumsum()
    )

    return summary


def make_exit_summary(
    trades_df: pd.DataFrame
) -> pd.DataFrame:

    if trades_df.empty:
        return pd.DataFrame()

    result = (
        trades_df
        .groupby(
            ["option_type", "exit_reason"]
        )
        .agg(
            Trades=("gross_pnl", "count"),
            PnL=("gross_pnl", "sum"),
            Avg_PnL=("gross_pnl", "mean")
        )
        .reset_index()
    )

    return result


def write_excel_report(
    trades: List[Trade],
    daily_summary: pd.DataFrame,
    errors: List[dict],
    output_file: str
):

    trades_df = pd.DataFrame(
        [t.__dict__ for t in trades]
    )

    statistics = make_statistics(
        trades_df
    )

    exit_summary = make_exit_summary(
        trades_df
    )

    parameters = pd.DataFrame([
        ("Instrument", "NIFTY"),
        ("Underlying", UNDERLYING),
        ("Backtest Start", START_DATE),
        ("Backtest End", END_DATE),
        ("Expiry Rule", "Nearest weekly expiry available"),
        ("ATM Rule", "Strike closest to 09:15 NIFTY open"),
        ("Option Open", "09:15 option candle open"),
        ("Entry Multiple", ENTRY_MULTIPLE),
        ("SL Multiple", SL_MULTIPLE),
        ("Target Multiple", TARGET_MULTIPLE),
        ("Entry Cutoff", ENTRY_CUTOFF.strftime("%H:%M")),
        ("Force Exit", FORCE_EXIT_TIME.strftime("%H:%M")),
        ("Same Candle Policy", SAME_CANDLE_POLICY),
        ("Resolution", "1 minute"),
        ("Quantity", 1),
        ("P&L Model", "Points x quantity"),
        ("Brokerage/Taxes", "NOT INCLUDED"),
    ], columns=["Parameter", "Value"])

    errors_df = pd.DataFrame(errors)

    if errors_df.empty:
        errors_df = pd.DataFrame(
            columns=[
                "trade_date",
                "stage",
                "message"
            ]
        )

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        parameters.to_excel(
            writer,
            sheet_name="Parameters",
            index=False
        )

        statistics.to_excel(
            writer,
            sheet_name="Statistics",
            index=False
        )

        if not trades_df.empty:

            trades_df.to_excel(
                writer,
                sheet_name="Trades",
                index=False
            )

        if not daily_summary.empty:

            daily_summary.to_excel(
                writer,
                sheet_name="Daily Summary",
                index=False
            )

        if not exit_summary.empty:

            exit_summary.to_excel(
                writer,
                sheet_name="Exit Summary",
                index=False
            )

        errors_df.to_excel(
            writer,
            sheet_name="Errors",
            index=False
        )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    from openpyxl import load_workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = load_workbook(
        output_file
    )

    for ws in wb.worksheets:

        # Header
        for cell in ws[1]:
            cell.font = Font(
                bold=True
            )

        # Freeze header
        ws.freeze_panes = "A2"

        # Auto-width
        for column_cells in ws.columns:

            max_length = 0

            column_letter = get_column_letter(
                column_cells[0].column
            )

            for cell in column_cells:

                try:
                    value_length = len(
                        str(cell.value)
                    )

                    max_length = max(
                        max_length,
                        value_length
                    )

                except Exception:
                    pass

            ws.column_dimensions[
                column_letter
            ].width = min(
                max_length + 2,
                45
            )

    wb.save(
        output_file
    )


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_backtest():

    start_date = datetime.strptime(
        START_DATE,
        "%Y-%m-%d"
    ).date()

    end_date = datetime.strptime(
        END_DATE,
        "%Y-%m-%d"
    ).date()

    print("=" * 70)
    print("NIFTY 2X OPTION BACKTEST")
    print("=" * 70)

    print(
        f"Period: {start_date} -> {end_date}"
    )

    print(
        f"Entry: {ENTRY_MULTIPLE}x option open"
    )

    print(
        f"SL: {SL_MULTIPLE}x option open"
    )

    print(
        f"Target: {TARGET_MULTIPLE}x option open"
    )

    print(
        f"Same candle: {SAME_CANDLE_POLICY}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: NIFTY opens
    # --------------------------------------------------------

    underlying = get_underlying_history(
        start_date,
        end_date
    )

    print(
        f"\nTrading days found: {len(underlying)}"
    )

    # --------------------------------------------------------
    # Step 2: Discover expiry dates
    # --------------------------------------------------------

    print(
        "\nDiscovering available expiry dates..."
    )

    known_expiries = get_expiry_dates(
        UNDERLYING,
        start_date,
        end_date
    )

    print(
        f"Expiry dates discovered: "
        f"{len(known_expiries)}"
    )

    # --------------------------------------------------------
    # Step 3: Process every day
    # --------------------------------------------------------

    trades: List[Trade] = []

    errors: List[dict] = []

    expiry_cache: Dict[date, date] = {}

    contract_cache: Dict[
        date,
        Dict[float, Dict[str, str]]
    ] = {}

    processed = 0

    for _, day_row in underlying.iterrows():

        trade_date = day_row["date"]

        underlying_open = safe_float(
            day_row["open"]
        )

        if underlying_open is None:
            continue

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"{trade_date} | "
            f"NIFTY Open = {underlying_open:.2f}"
        )

        # ----------------------------------------------------
        # Expiry
        # ----------------------------------------------------

        try:

            expiry = discover_expiry_for_day(
                trade_date,
                expiry_cache,
                known_expiries
            )

            if expiry is None:

                msg = (
                    "Could not determine nearest "
                    "weekly expiry."
                )

                print(
                    "SKIP:",
                    msg
                )

                errors.append({
                    "trade_date": trade_date,
                    "stage": "expiry",
                    "message": msg
                })

                continue

            print(
                f"Expiry = {expiry}"
            )

        except Exception as exc:

            print(
                "Expiry error:",
                exc
            )

            errors.append({
                "trade_date": trade_date,
                "stage": "expiry",
                "message": str(exc)
            })

            continue

        # ----------------------------------------------------
        # Contract discovery
        # ----------------------------------------------------

        try:

            if expiry not in contract_cache:

                print(
                    "Downloading option contracts..."
                )

                contracts = get_contracts_for_expiry(
                    trade_date,
                    expiry
                )

                contract_cache[expiry] = (
                    build_contract_map(
                        contracts
                    )
                )

                time.sleep(
                    API_SLEEP_SECONDS
                )

            contract_map = contract_cache[
                expiry
            ]

        except Exception as exc:

            print(
                "Contract error:",
                exc
            )

            errors.append({
                "trade_date": trade_date,
                "stage": "contracts",
                "message": str(exc)
            })

            continue

        if not contract_map:

            errors.append({
                "trade_date": trade_date,
                "stage": "contracts",
                "message": "No option contracts found."
            })

            continue

        # ----------------------------------------------------
        # ATM strike
        # ----------------------------------------------------

        strikes = sorted(
            contract_map.keys()
        )

        strike = nearest_strike(
            underlying_open,
            strikes
        )

        option_contracts = contract_map.get(
            strike,
            {}
        )

        ce_symbol = option_contracts.get(
            "CE"
        )

        pe_symbol = option_contracts.get(
            "PE"
        )

        print(
            f"ATM Strike = {strike}"
        )

        print(
            f"CE = {ce_symbol}"
        )

        print(
            f"PE = {pe_symbol}"
        )

        # ----------------------------------------------------
        # CE
        # ----------------------------------------------------

        if ce_symbol:

            try:

                print(
                    "Downloading CE..."
                )

                ce_df = get_option_history(
                    ce_symbol,
                    trade_date
                )

                trade = backtest_option(
                    ce_df,
                    trade_date,
                    "CE",
                    ce_symbol,
                    expiry,
                    strike,
                    underlying_open
                )

                if trade:

                    trades.append(
                        trade
                    )

                    print(
                        f"CE TRADE: "
                        f"open={trade.option_open:.2f}, "
                        f"entry={trade.entry_price:.2f}, "
                        f"exit={trade.exit_price:.2f}, "
                        f"{trade.exit_reason}, "
                        f"P&L={trade.gross_pnl:.2f}"
                    )

                else:

                    print(
                        "CE: no trade"
                    )

            except Exception as exc:

                print(
                    "CE error:",
                    exc
                )

                errors.append({
                    "trade_date": trade_date,
                    "stage": "CE",
                    "message": str(exc)
                })

        # ----------------------------------------------------
        # PE
        # ----------------------------------------------------

        if pe_symbol:

            try:

                print(
                    "Downloading PE..."
                )

                pe_df = get_option_history(
                    pe_symbol,
                    trade_date
                )

                trade = backtest_option(
                    pe_df,
                    trade_date,
                    "PE",
                    pe_symbol,
                    expiry,
                    strike,
                    underlying_open
                )

                if trade:

                    trades.append(
                        trade
                    )

                    print(
                        f"PE TRADE: "
                        f"open={trade.option_open:.2f}, "
                        f"entry={trade.entry_price:.2f}, "
                        f"exit={trade.exit_price:.2f}, "
                        f"{trade.exit_reason}, "
                        f"P&L={trade.gross_pnl:.2f}"
                    )

                else:

                    print(
                        "PE: no trade"
                    )

            except Exception as exc:

                print(
                    "PE error:",
                    exc
                )

                errors.append({
                    "trade_date": trade_date,
                    "stage": "PE",
                    "message": str(exc)
                })

        processed += 1

        print(
            f"Progress: {processed}/{len(underlying)}"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    trades_df = pd.DataFrame(
        [t.__dict__ for t in trades]
    )

    daily_summary = make_daily_summary(
        trades_df
    )

    # --------------------------------------------------------
    # Write Excel
    # --------------------------------------------------------

    print(
        "\nWriting Excel report..."
    )

    write_excel_report(
        trades,
        daily_summary,
        errors,
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BACKTEST COMPLETE"
    )

    print(
        "=" * 70
    )

    if trades_df.empty:

        print(
            "No trades were generated."
        )

    else:

        statistics = make_statistics(
            trades_df
        )

        for _, row in statistics.iterrows():

            print(
                f"{row['Metric']}: "
                f"{row['Value']}"
            )

    print(
        "\nExcel report:"
    )

    print(
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    print(
        "\nErrors logged:",
        len(errors)
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_backtest()

    except KeyboardInterrupt:

        print(
            "\nBacktest interrupted by user."
        )

    except Exception as exc:

        print(
            "\nFATAL ERROR:"
        )

        print(
            str(exc)
        )

        traceback.print_exc()

        sys.exit(1)

