import re
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

from fyers_apiv3 import fyersModel


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_ID = ""

# IMPORTANT:
# Paste your CURRENT FYERS access token here locally.
# Do not share the token.
ACCESS_TOKEN = ""

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

UNDERLYING = "NSE:NIFTY50-INDEX"

ENTRY_MULTIPLIER = 2.0
STOP_MULTIPLIER = 1.0
TARGET_MULTIPLIER = 3.0

LAST_ENTRY_TIME = "15:00"
FORCE_EXIT_TIME = "15:05"

QUANTITY = 65

# If entry, SL and target are all touched in the same candle:
# SL wins.
SAME_CANDLE_RULE = "SL_FIRST"

OUTPUT_FILE = "NIFTY_2X_Backtest_2024_2025.xlsx"

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# API / RETRY CONFIG
# ============================================================

# Small delay between normal FYERS calls.
API_SLEEP_SECONDS = 0.35

# Retry settings specifically for HTTP 429.
MAX_429_RETRIES = 6

# Conservative exponential backoff:
# 2, 4, 8, 16, 32, 60 seconds approximately.
INITIAL_429_BACKOFF = 2.0
MAX_429_BACKOFF = 60.0


# ============================================================
# FYERS CLIENT
# ============================================================

if not ACCESS_TOKEN:
    raise RuntimeError(
        "ACCESS_TOKEN is empty. Paste your current FYERS access token "
        "into ACCESS_TOKEN before running the script."
    )

fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    is_async=False,
    log_path=""
)


# ============================================================
# CACHES
# ============================================================

expiry_cache = None

contract_cache = {}
contract_failed_cache = {}

underlying_cache = {}
option_cache = {}

diagnostic_contract_expiries = set()

errors = []


# ============================================================
# BASIC HELPERS
# ============================================================

def print_progress(message):
    print(message, flush=True)


def api_sleep():
    time.sleep(API_SLEEP_SECONDS)


def parse_date(value):
    if isinstance(value, date):
        return value

    return datetime.strptime(
        str(value),
        "%Y-%m-%d"
    ).date()


def format_date(value):
    return parse_date(value).strftime("%Y-%m-%d")


def time_to_minutes(value):
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


LAST_ENTRY_MINUTES = time_to_minutes(LAST_ENTRY_TIME)
FORCE_EXIT_MINUTES = time_to_minutes(FORCE_EXIT_TIME)


def candle_timestamp_to_ist(timestamp):
    """
    Convert FYERS epoch timestamp to naive IST datetime.

    Excel/openpyxl does not accept timezone-aware datetimes,
    so the timezone is removed after conversion to IST.
    """
    return datetime.fromtimestamp(
        int(timestamp),
        tz=IST
    ).replace(tzinfo=None)


def candle_time_minutes(timestamp):
    dt = candle_timestamp_to_ist(timestamp)
    return dt.hour * 60 + dt.minute


def record_error(trading_date, stage, message):
    errors.append({
        "Date": format_date(trading_date),
        "Stage": stage,
        "Error": str(message)
    })


# ============================================================
# FYERS RESPONSE HELPERS
# ============================================================

def is_no_data_response(response):
    if not isinstance(response, dict):
        return False

    return response.get("s") == "no_data"


def is_rate_limit_response(response):
    if not isinstance(response, dict):
        return False

    if response.get("code") == 429:
        return True

    message = str(
        response.get("message", "")
    ).lower()

    return (
        "request limit" in message
        or "rate limit" in message
        or "too many requests" in message
    )


def check_fyers_response(response, context):
    if not isinstance(response, dict):
        raise RuntimeError(
            f"{context}: Invalid FYERS response: {response}"
        )

    if response.get("s") != "ok":
        raise RuntimeError(
            f"{context}: FYERS response not OK: {response}"
        )

    return response


# ============================================================
# 429 RETRY WRAPPER
# ============================================================

def fyers_call_with_429_retry(api_function, data, context):
    """
    Call a FYERS API function.

    If FYERS returns HTTP 429 / request limit reached,
    retry with exponential backoff.

    Other responses are returned immediately.
    """

    backoff = INITIAL_429_BACKOFF

    for attempt in range(MAX_429_RETRIES + 1):

        response = api_function(data)

        if not is_rate_limit_response(response):
            return response

        if attempt >= MAX_429_RETRIES:
            raise RuntimeError(
                f"{context}: FYERS request limit remained active "
                f"after {MAX_429_RETRIES} retries: {response}"
            )

        print_progress(
            f"FYERS 429 | {context} | "
            f"retry {attempt + 1}/{MAX_429_RETRIES} "
            f"in {backoff:.0f}s"
        )

        time.sleep(backoff)

        backoff = min(
            backoff * 2,
            MAX_429_BACKOFF
        )

    raise RuntimeError(
        f"{context}: unexpected retry failure"
    )


# ============================================================
# EXPIRY DATES
# ============================================================

def load_all_option_expiries():
    global expiry_cache

    if expiry_cache is not None:
        return expiry_cache

    print_progress("Loading NIFTY option expiry dates...")

    all_expiries = set()

    start = parse_date(START_DATE)
    end = parse_date(END_DATE)

    for year in range(start.year, end.year + 1):

        range_start = date(year, 1, 1)
        range_end = date(year, 12, 31)

        range_start = max(
            range_start,
            start
        )

        range_end = min(
            range_end,
            end
        )

        data = {
            "symbol": UNDERLYING,
            "range_from": format_date(range_start),
            "range_to": format_date(range_end),
            "date_format": "1"
        }

        try:

            response = fyers_call_with_429_retry(
                fyers.expiry_dates,
                data,
                f"expiry_dates {year}"
            )

            check_fyers_response(
                response,
                f"expiry_dates {year}"
            )

            expiry_data = (
                response
                .get("data", {})
                .get("expiry_dates", {})
            )

            options = expiry_data.get(
                "options",
                []
            )

            if options:
                for expiry in options:

                    try:
                        all_expiries.add(
                            parse_date(expiry)
                        )

                    except Exception:
                        print_progress(
                            f"WARNING: Could not parse expiry: "
                            f"{expiry}"
                        )

        except Exception as exc:

            raise RuntimeError(
                f"Could not load expiry dates for {year}: {exc}"
            ) from exc

        api_sleep()

    expiry_cache = sorted(
        all_expiries
    )

    if not expiry_cache:
        raise RuntimeError(
            "FYERS returned no option expiry dates."
        )

    print_progress(
        f"Loaded {len(expiry_cache)} option expiry dates."
    )

    print_progress(
        f"First expiry: {expiry_cache[0]}"
    )

    print_progress(
        f"Last expiry : {expiry_cache[-1]}"
    )

    return expiry_cache


def get_expiry_for_trading_date(trading_date):
    expiries = load_all_option_expiries()

    candidates = [
        expiry
        for expiry in expiries
        if expiry >= trading_date
    ]

    if not candidates:
        raise RuntimeError(
            f"No option expiry on or after "
            f"{format_date(trading_date)}"
        )

    return candidates[0]


# ============================================================
# CONTRACT DISCOVERY
# ============================================================

def load_contracts_for_expiry(expiry):

    expiry_key = format_date(expiry)

    if expiry_key in contract_cache:
        return contract_cache[expiry_key]

    # If a previous request ultimately failed because of repeated
    # 429 responses, don't endlessly repeat the same request for
    # every trading day.
    if expiry_key in contract_failed_cache:
        raise RuntimeError(
            contract_failed_cache[expiry_key]
        )

    data = {
        "symbol": UNDERLYING,
        "expiry_date": expiry_key
    }

    try:

        response = fyers_call_with_429_retry(
            fyers.history_underlying_symbols,
            data,
            f"history_underlying_symbols {expiry_key}"
        )

        check_fyers_response(
            response,
            f"history_underlying_symbols {expiry_key}"
        )

        contracts = (
            response
            .get("data", {})
            .get("contracts", {})
        )

        options = contracts.get(
            "options",
            []
        )

        if not options:
            raise RuntimeError(
                f"No option contracts returned "
                f"for expiry {expiry_key}"
            )

        contract_cache[expiry_key] = options

        if expiry_key not in diagnostic_contract_expiries:

            print_progress(
                f"Contracts {expiry_key}: "
                f"{len(options)} options"
            )

            print_progress(
                f"Sample contracts {expiry_key}:"
            )

            for contract in options[:10]:
                print_progress(
                    f"    {contract}"
                )

            diagnostic_contract_expiries.add(
                expiry_key
            )

        api_sleep()

        return options

    except Exception as exc:

        message = (
            f"Could not load contracts for expiry "
            f"{expiry_key}: {exc}"
        )

        # Only remember terminal failure.
        # Normal 429 retries have already happened inside
        # fyers_call_with_429_retry().
        contract_failed_cache[expiry_key] = message

        raise RuntimeError(
            message
        ) from exc


# ============================================================
# CONTRACT STRIKE PARSER
# ============================================================

def extract_strike_from_contract(symbol, expiry):
    """
    FYERS NIFTY expired-option symbols observed from the API:

        NSE:NIFTY25D1620450CE
        NSE:NIFTY25D1620450PE

    The important part for ATM selection is the numeric strike
    immediately before CE/PE.

    We deliberately do NOT reconstruct the expiry prefix here.
    The contract list was already returned by FYERS for the
    requested expiry.
    """

    if not isinstance(symbol, str):
        return None

    symbol = symbol.strip().upper()

    if not symbol.startswith(
        "NSE:NIFTY"
    ):
        return None

    if symbol.endswith("CE"):
        body = symbol[:-2]

    elif symbol.endswith("PE"):
        body = symbol[:-2]

    else:
        return None

    match = re.search(
        r"(\d+(?:\.\d+)?)$",
        body
    )

    if not match:
        return None

    try:
        return float(
            match.group(1)
        )

    except Exception:
        return None


# ============================================================
# ATM SELECTION
# ============================================================

def get_atm_option_symbols(
    contracts,
    expiry,
    underlying_open
):

    parsed = []

    for symbol in contracts:

        strike = extract_strike_from_contract(
            symbol,
            expiry
        )

        if strike is None:
            continue

        if symbol.endswith("CE"):
            option_type = "CE"

        elif symbol.endswith("PE"):
            option_type = "PE"

        else:
            continue

        parsed.append({
            "symbol": symbol,
            "strike": strike,
            "option_type": option_type
        })

    if not parsed:

        print_progress("")
        print_progress("=" * 48)
        print_progress("ATM PARSE FAILED")
        print_progress(
            f"Expiry         : {expiry}"
        )
        print_progress(
            f"Underlying open: {underlying_open}"
        )
        print_progress(
            f"Contracts      : {len(contracts)}"
        )
        print_progress(
            "Actual FYERS contracts:"
        )

        for contract in contracts[:30]:
            print_progress(
                f"    {contract}"
            )

        print_progress("=" * 48)

        raise RuntimeError(
            f"Could not parse any NIFTY option "
            f"contracts for expiry {expiry}"
        )

    strikes = sorted(
        set(
            item["strike"]
            for item in parsed
        )
    )

    atm_strike = min(
        strikes,
        key=lambda strike: (
            abs(
                strike - underlying_open
            ),
            strike
        )
    )

    ce_symbol = None
    pe_symbol = None

    for item in parsed:

        if item["strike"] != atm_strike:
            continue

        if item["option_type"] == "CE":
            ce_symbol = item["symbol"]

        elif item["option_type"] == "PE":
            pe_symbol = item["symbol"]

    if ce_symbol is None or pe_symbol is None:

        print_progress(
            f"ATM PAIR INCOMPLETE | "
            f"Expiry={expiry} | "
            f"Strike={atm_strike} | "
            f"CE={ce_symbol} | "
            f"PE={pe_symbol}"
        )

        raise RuntimeError(
            f"Incomplete ATM pair for expiry "
            f"{expiry}, strike {atm_strike}"
        )

    return (
        atm_strike,
        ce_symbol,
        pe_symbol
    )


# ============================================================
# UNDERLYING DATA
# ============================================================

def get_underlying_day_data(trading_date):

    key = format_date(
        trading_date
    )

    if key in underlying_cache:
        return underlying_cache[key]

    data = {
        "symbol": UNDERLYING,
        "resolution": "1",
        "date_format": "1",
        "range_from": key,
        "range_to": key,
        "cont_flag": "1"
    }

    try:

        response = fyers_call_with_429_retry(
            fyers.history,
            data,
            f"NIFTY history {key}"
        )

        # FYERS can legitimately return:
        #
        # {
        #   "candles": [],
        #   "s": "no_data"
        # }
        #
        # This is NOT an API failure.
        if is_no_data_response(response):

            underlying_cache[key] = []

            api_sleep()

            return []

        check_fyers_response(
            response,
            f"NIFTY history {key}"
        )

        candles = response.get(
            "candles",
            []
        )

        underlying_cache[key] = candles

        api_sleep()

        return candles

    except Exception as exc:

        raise RuntimeError(
            f"NIFTY history {key}: {exc}"
        ) from exc


# ============================================================
# OPTION DATA
# ============================================================

def get_option_day_data(
    symbol,
    trading_date
):

    key = (
        symbol,
        format_date(trading_date)
    )

    if key in option_cache:
        return option_cache[key]

    data = {
        "symbol": symbol,
        "resolution": "1",
        "date_format": "1",
        "range_from": format_date(trading_date),
        "range_to": format_date(trading_date),
        "include_greeks": "1",
        "include_oi": "1"
    }

    try:

        response = fyers_call_with_429_retry(
            fyers.fno_historical_data,
            data,
            f"FNO history {symbol} "
            f"{format_date(trading_date)}"
        )

        # IMPORTANT:
        # no_data is a valid FYERS response.
        if is_no_data_response(response):

            option_cache[key] = []

            api_sleep()

            return []

        check_fyers_response(
            response,
            f"FNO history {symbol} "
            f"{format_date(trading_date)}"
        )

        candles = response.get(
            "candles",
            []
        )

        option_cache[key] = candles

        api_sleep()

        return candles

    except Exception as exc:

        raise RuntimeError(
            f"FNO history {symbol} "
            f"{format_date(trading_date)}: {exc}"
        ) from exc


# ============================================================
# CANDLE FILTERING
# ============================================================

def filter_session_candles(candles):

    result = []

    for candle in candles:

        if not candle:
            continue

        timestamp = candle[0]

        dt = candle_timestamp_to_ist(
            timestamp
        )

        minutes = (
            dt.hour * 60
            + dt.minute
        )

        if (
            minutes >= 9 * 60 + 15
            and minutes <= FORCE_EXIT_MINUTES
        ):
            result.append({
                "timestamp": timestamp,
                "datetime": dt,
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": (
                    float(candle[5])
                    if len(candle) > 5
                    else None
                )
            })

    result.sort(
        key=lambda x: x["timestamp"]
    )

    return result


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    trading_date,
    expiry,
    underlying_open,
    atm_strike,
    option_type,
    symbol,
    candles
):

    session = filter_session_candles(
        candles
    )

    if not session:
        return None

    first_candle = session[0]

    option_open = first_candle["open"]

    if option_open <= 0:
        return None

    entry_price = (
        option_open
        * ENTRY_MULTIPLIER
    )

    stop_price = (
        option_open
        * STOP_MULTIPLIER
    )

    target_price = (
        option_open
        * TARGET_MULTIPLIER
    )

    entry_candle = None
    entry_index = None

    for index, candle in enumerate(session):

        candle_minutes = (
            candle["datetime"].hour * 60
            + candle["datetime"].minute
        )

        if candle_minutes > LAST_ENTRY_MINUTES:
            break

        if candle["high"] >= entry_price:

            entry_candle = candle
            entry_index = index
            break

    if entry_candle is None:
        return None

    entry_time = entry_candle["datetime"]

    exit_price = None
    exit_time = None
    exit_reason = None

    # --------------------------------------------------------
    # Check candles from entry onward.
    # --------------------------------------------------------

    for candle in session[entry_index:]:

        candle_time = candle["datetime"]

        candle_minutes = (
            candle_time.hour * 60
            + candle_time.minute
        )

        hit_sl = (
            candle["low"] <= stop_price
        )

        hit_target = (
            candle["high"] >= target_price
        )

        # Same-candle rule.
        if hit_sl and hit_target:

            if SAME_CANDLE_RULE == "SL_FIRST":

                exit_price = stop_price
                exit_time = candle_time
                exit_reason = "SL"

                break

            else:

                exit_price = target_price
                exit_time = candle_time
                exit_reason = "TARGET"

                break

        if hit_sl:

            exit_price = stop_price
            exit_time = candle_time
            exit_reason = "SL"

            break

        if hit_target:

            exit_price = target_price
            exit_time = candle_time
            exit_reason = "TARGET"

            break

        if candle_minutes >= FORCE_EXIT_MINUTES:

            exit_price = candle["close"]
            exit_time = candle_time
            exit_reason = "EOD"

            break

    if exit_price is None:

        last_candle = session[-1]

        exit_price = last_candle["close"]
        exit_time = last_candle["datetime"]
        exit_reason = "EOD"

    points = (
        exit_price
        - entry_price
    )

    pnl = (
        points
        * QUANTITY
    )

    if pnl > 0:
        outcome = "WIN"

    elif pnl < 0:
        outcome = "LOSS"

    else:
        outcome = "BREAKEVEN"

    return {
        "Date": trading_date,
        "Expiry": expiry,
        "ATM Strike": atm_strike,
        "Underlying Open": underlying_open,
        "Option Type": option_type,
        "Symbol": symbol,
        "Option Open": option_open,
        "Entry Price": entry_price,
        "Stop Price": stop_price,
        "Target Price": target_price,
        "Entry Time": entry_time,
        "Exit Time": exit_time,
        "Exit Price": exit_price,
        "Exit Reason": exit_reason,
        "Outcome": outcome,
        "Points": points,
        "Quantity": QUANTITY,
        "P&L": pnl
    }


# ============================================================
# TRADE BUILDER
# ============================================================

def build_trade_for_day(
    trading_date,
    expiry,
    underlying_candles
):

    if not underlying_candles:
        return None

    session = filter_session_candles(
        underlying_candles
    )

    if not session:
        return None

    underlying_open = session[0]["open"]

    contracts = load_contracts_for_expiry(
        expiry
    )

    atm_strike, ce_symbol, pe_symbol = (
        get_atm_option_symbols(
            contracts,
            expiry,
            underlying_open
        )
    )

    ce_candles = get_option_day_data(
        ce_symbol,
        trading_date
    )

    pe_candles = get_option_day_data(
        pe_symbol,
        trading_date
    )

    # No-data from either leg means we cannot construct
    # the required CE/PE trade for that day.
    if not ce_candles:
        return None

    if not pe_candles:
        return None

    ce_trade = simulate_trade(
        trading_date=trading_date,
        expiry=expiry,
        underlying_open=underlying_open,
        atm_strike=atm_strike,
        option_type="CE",
        symbol=ce_symbol,
        candles=ce_candles
    )

    pe_trade = simulate_trade(
        trading_date=trading_date,
        expiry=expiry,
        underlying_open=underlying_open,
        atm_strike=atm_strike,
        option_type="PE",
        symbol=pe_symbol,
        candles=pe_candles
    )

    # Existing strategy evaluates the two option legs
    # independently. Return both trades that actually trigger.
    trades = []

    if ce_trade is not None:
        trades.append(ce_trade)

    if pe_trade is not None:
        trades.append(pe_trade)

    return trades


# ============================================================
# DATAFRAME BUILDERS
# ============================================================

def build_trades_dataframe(trades):

    columns = [
        "Date",
        "Expiry",
        "ATM Strike",
        "Underlying Open",
        "Option Type",
        "Symbol",
        "Option Open",
        "Entry Price",
        "Stop Price",
        "Target Price",
        "Entry Time",
        "Exit Time",
        "Exit Price",
        "Exit Reason",
        "Outcome",
        "Points",
        "Quantity",
        "P&L"
    ]

    if not trades:
        return pd.DataFrame(
            columns=columns
        )

    df = pd.DataFrame(
        trades
    )

    for column in [
        "Date",
        "Expiry",
        "Entry Time",
        "Exit Time"
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column]
            )

    return df[columns]


def build_daily_summary(trades_df):

    if trades_df.empty:
        return pd.DataFrame()

    grouped = (
        trades_df
        .groupby("Date", as_index=False)
        .agg(
            Trades=("P&L", "count"),
            Wins=(
                "Outcome",
                lambda x: (x == "WIN").sum()
            ),
            Losses=(
                "Outcome",
                lambda x: (x == "LOSS").sum()
            ),
            Breakeven=(
                "Outcome",
                lambda x: (x == "BREAKEVEN").sum()
            ),
            PnL=("P&L", "sum")
        )
    )

    grouped["Win Rate %"] = (
        grouped["Wins"]
        / grouped["Trades"]
        * 100
    )

    return grouped


def build_monthly_summary(trades_df):

    if trades_df.empty:
        return pd.DataFrame()

    temp = trades_df.copy()

    temp["Month"] = (
        pd.to_datetime(
            temp["Date"]
        ).dt.to_period("M")
        .astype(str)
    )

    grouped = (
        temp
        .groupby("Month", as_index=False)
        .agg(
            Trades=("P&L", "count"),
            Wins=(
                "Outcome",
                lambda x: (x == "WIN").sum()
            ),
            Losses=(
                "Outcome",
                lambda x: (x == "LOSS").sum()
            ),
            Breakeven=(
                "Outcome",
                lambda x: (x == "BREAKEVEN").sum()
            ),
            PnL=("P&L", "sum")
        )
    )

    grouped["Win Rate %"] = (
        grouped["Wins"]
        / grouped["Trades"]
        * 100
    )

    return grouped


def build_exit_summary(trades_df):

    if trades_df.empty:
        return pd.DataFrame()

    grouped = (
        trades_df
        .groupby(
            "Exit Reason",
            as_index=False
        )
        .agg(
            Trades=("P&L", "count"),
            Wins=(
                "Outcome",
                lambda x: (x == "WIN").sum()
            ),
            Losses=(
                "Outcome",
                lambda x: (x == "LOSS").sum()
            ),
            PnL=("P&L", "sum")
        )
    )

    grouped["Win Rate %"] = (
        grouped["Wins"]
        / grouped["Trades"]
        * 100
    )

    return grouped


def build_statistics(trades_df):

    if trades_df.empty:

        return pd.DataFrame({
            "Metric": [
                "Trades",
                "Wins",
                "Losses",
                "Breakeven",
                "Win Rate %",
                "Gross Profit",
                "Gross Loss",
                "Net P&L",
                "Average Win",
                "Average Loss"
            ],
            "Value": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]
        })

    wins = (
        trades_df[
            trades_df["Outcome"] == "WIN"
        ]
    )

    losses = (
        trades_df[
            trades_df["Outcome"] == "LOSS"
        ]
    )

    total = len(trades_df)

    win_count = len(wins)
    loss_count = len(losses)
    breakeven_count = (
        trades_df["Outcome"] == "BREAKEVEN"
    ).sum()

    gross_profit = (
        wins["P&L"].sum()
        if not wins.empty
        else 0
    )

    gross_loss = (
        losses["P&L"].sum()
        if not losses.empty
        else 0
    )

    net_pnl = trades_df["P&L"].sum()

    win_rate = (
        win_count
        / total
        * 100
        if total
        else 0
    )

    avg_win = (
        wins["P&L"].mean()
        if not wins.empty
        else 0
    )

    avg_loss = (
        losses["P&L"].mean()
        if not losses.empty
        else 0
    )

    return pd.DataFrame({
        "Metric": [
            "Trades",
            "Wins",
            "Losses",
            "Breakeven",
            "Win Rate %",
            "Gross Profit",
            "Gross Loss",
            "Net P&L",
            "Average Win",
            "Average Loss"
        ],
        "Value": [
            total,
            win_count,
            loss_count,
            int(breakeven_count),
            win_rate,
            gross_profit,
            gross_loss,
            net_pnl,
            avg_win,
            avg_loss
        ]
    })


def build_parameters():

    return pd.DataFrame({
        "Parameter": [
            "Start Date",
            "End Date",
            "Underlying",
            "Entry Multiplier",
            "Stop Multiplier",
            "Target Multiplier",
            "Last Entry Time",
            "Force Exit Time",
            "Quantity",
            "Same Candle Rule",
            "API Sleep Seconds",
            "Max 429 Retries",
            "Initial 429 Backoff",
            "Max 429 Backoff"
        ],
        "Value": [
            START_DATE,
            END_DATE,
            UNDERLYING,
            ENTRY_MULTIPLIER,
            STOP_MULTIPLIER,
            TARGET_MULTIPLIER,
            LAST_ENTRY_TIME,
            FORCE_EXIT_TIME,
            QUANTITY,
            SAME_CANDLE_RULE,
            API_SLEEP_SECONDS,
            MAX_429_RETRIES,
            INITIAL_429_BACKOFF,
            MAX_429_BACKOFF
        ]
    })


def build_error_dataframe():

    if not errors:
        return pd.DataFrame(
            columns=[
                "Date",
                "Stage",
                "Error"
            ]
        )

    return pd.DataFrame(
        errors,
        columns=[
            "Date",
            "Stage",
            "Error"
        ]
    )


def build_daily_metadata():

    rows = []

    for key, candles in underlying_cache.items():

        rows.append({
            "Date": key,
            "Underlying Candles": len(candles)
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "Date",
                "Underlying Candles"
            ]
        )

    return pd.DataFrame(rows)


# ============================================================
# EXCEL WRITER
# ============================================================

def write_excel(
    trades_df,
    statistics_df,
    daily_summary_df,
    monthly_summary_df,
    exit_summary_df,
    daily_metadata_df,
    errors_df
):

    print_progress("")
    print_progress("WRITING EXCEL")

    parameters_df = build_parameters()

    wb = Workbook()

    default_sheet = wb.active
    wb.remove(default_sheet)

    sheets = [
        (
            "Parameters",
            parameters_df
        ),
        (
            "Statistics",
            statistics_df
        ),
        (
            "Trades",
            trades_df
        ),
        (
            "Daily Summary",
            daily_summary_df
        ),
        (
            "Monthly Summary",
            monthly_summary_df
        ),
        (
            "Exit Summary",
            exit_summary_df
        ),
        (
            "Daily Metadata",
            daily_metadata_df
        ),
        (
            "Errors",
            errors_df
        )
    ]

    for sheet_name, df in sheets:

        ws = wb.create_sheet(
            title=sheet_name
        )

        if df is None or df.empty:
            ws.append(
                list(df.columns)
                if df is not None
                else []
            )
            continue

        ws.append(
            list(df.columns)
        )

        for cell in ws[1]:
            cell.font = Font(
                bold=True
            )

        for row in df.itertuples(
            index=False,
            name=None
        ):
            ws.append(
                list(row)
            )

        ws.freeze_panes = "A2"

        # Basic column sizing.
        for column_cells in ws.columns:

            max_length = 0

            column_letter = (
                column_cells[0]
                .column_letter
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
                max(max_length + 2, 10),
                60
            )

    wb.save(
        OUTPUT_FILE
    )

    print_progress(
        f"Excel saved: {OUTPUT_FILE}"
    )


# ============================================================
# MAIN BACKTEST
# ============================================================

def main():

    start = parse_date(
        START_DATE
    )

    end = parse_date(
        END_DATE
    )

    # Load expiry data once.
    expiries = load_all_option_expiries()

    all_trades = []

    current = start

    total_days = (
        end - start
    ).days + 1

    counter = 0

    while current <= end:

        counter += 1

        print_progress(
            f"[{counter}/{total_days}] "
            f"{current}"
        )

        # Monday-Friday only.
        if current.weekday() >= 5:
            current += timedelta(
                days=1
            )
            continue

        # ----------------------------------------------------
        # UNDERLYING
        # ----------------------------------------------------

        try:

            underlying_candles = (
                get_underlying_day_data(
                    current
                )
            )

        except Exception as exc:

            record_error(
                current,
                "UNDERLYING",
                str(exc)
            )

            current += timedelta(
                days=1
            )

            continue

        # FYERS no_data.
        # This can happen on exchange holidays or days where
        # data isn't available.
        if not underlying_candles:

            current += timedelta(
                days=1
            )

            continue

        # ----------------------------------------------------
        # EXPIRY
        # ----------------------------------------------------

        try:

            expiry = get_expiry_for_trading_date(
                current
            )

        except Exception as exc:

            record_error(
                current,
                "EXPIRY",
                str(exc)
            )

            current += timedelta(
                days=1
            )

            continue

        # ----------------------------------------------------
        # CONTRACTS
        # ----------------------------------------------------

        try:

            contracts = load_contracts_for_expiry(
                expiry
            )

        except Exception as exc:

            record_error(
                current,
                "CONTRACTS",
                str(exc)
            )

            current += timedelta(
                days=1
            )

            continue

        # ----------------------------------------------------
        # UNDERLYING OPEN
        # ----------------------------------------------------

        underlying_session = (
            filter_session_candles(
                underlying_candles
            )
        )

        if not underlying_session:

            current += timedelta(
                days=1
            )

            continue

        underlying_open = (
            underlying_session[0]["open"]
        )

        # ----------------------------------------------------
        # ATM
        # ----------------------------------------------------

        try:

            (
                atm_strike,
                ce_symbol,
                pe_symbol
            ) = get_atm_option_symbols(
                contracts,
                expiry,
                underlying_open
            )

        except Exception as exc:

            record_error(
                current,
                "ATM",
                str(exc)
            )

            current += timedelta(
                days=1
            )

            continue

        # ----------------------------------------------------
        # CE DATA
        # ----------------------------------------------------

        try:

            ce_candles = get_option_day_data(
                ce_symbol,
                current
            )

        except Exception as exc:

            record_error(
                current,
                "CE_DATA",
                str(exc)
            )

            ce_candles = []

        # ----------------------------------------------------
        # PE DATA
        # ----------------------------------------------------

        try:

            pe_candles = get_option_day_data(
                pe_symbol,
                current
            )

        except Exception as exc:

            record_error(
                current,
                "PE_DATA",
                str(exc)
            )

            pe_candles = []

        # ----------------------------------------------------
        # SIMULATE CE
        # ----------------------------------------------------

        if ce_candles:

            try:

                ce_trade = simulate_trade(
                    trading_date=current,
                    expiry=expiry,
                    underlying_open=underlying_open,
                    atm_strike=atm_strike,
                    option_type="CE",
                    symbol=ce_symbol,
                    candles=ce_candles
                )

                if ce_trade is not None:
                    all_trades.append(
                        ce_trade
                    )

            except Exception as exc:

                record_error(
                    current,
                    "CE_TRADE",
                    str(exc)
                )

        # ----------------------------------------------------
        # SIMULATE PE
        # ----------------------------------------------------

        if pe_candles:

            try:

                pe_trade = simulate_trade(
                    trading_date=current,
                    expiry=expiry,
                    underlying_open=underlying_open,
                    atm_strike=atm_strike,
                    option_type="PE",
                    symbol=pe_symbol,
                    candles=pe_candles
                )

                if pe_trade is not None:
                    all_trades.append(
                        pe_trade
                    )

            except Exception as exc:

                record_error(
                    current,
                    "PE_TRADE",
                    str(exc)
                )

        current += timedelta(
            days=1
        )

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    trades_df = build_trades_dataframe(
        all_trades
    )

    statistics_df = build_statistics(
        trades_df
    )

    daily_summary_df = build_daily_summary(
        trades_df
    )

    monthly_summary_df = build_monthly_summary(
        trades_df
    )

    exit_summary_df = build_exit_summary(
        trades_df
    )

    daily_metadata_df = build_daily_metadata()

    errors_df = build_error_dataframe()

    # ========================================================
    # ERROR SUMMARY
    # ========================================================

    print_progress("")
    print_progress("ERROR SUMMARY")
    print_progress("----------------")

    if errors_df.empty:

        print_progress(
            "No errors."
        )

    else:

        error_counts = (
            errors_df["Stage"]
            .value_counts()
        )

        for stage, count in error_counts.items():

            print_progress(
                f"{stage}: {count}"
            )

        print_progress(
            f"Total errors: {len(errors_df)}"
        )

    # ========================================================
    # BACKTEST RESULTS
    # ========================================================

    print_progress("")
    print_progress("BACKTEST RESULTS")
    print_progress("----------------")

    if trades_df.empty:

        print_progress(
            "Trades       : 0"
        )

    else:

        wins = int(
            (
                trades_df["Outcome"]
                == "WIN"
            ).sum()
        )

        losses = int(
            (
                trades_df["Outcome"]
                == "LOSS"
            ).sum()
        )

        breakeven = int(
            (
                trades_df["Outcome"]
                == "BREAKEVEN"
            ).sum()
        )

        total_trades = len(
            trades_df
        )

        win_rate = (
            wins
            / total_trades
            * 100
        )

        gross_profit = trades_df.loc[
            trades_df["P&L"] > 0,
            "P&L"
        ].sum()

        gross_loss = trades_df.loc[
            trades_df["P&L"] < 0,
            "P&L"
        ].sum()

        net_pnl = trades_df[
            "P&L"
        ].sum()

        print_progress(
            f"Trades       : {total_trades}"
        )

        print_progress(
            f"Wins         : {wins}"
        )

        print_progress(
            f"Losses       : {losses}"
        )

        print_progress(
            f"Breakeven    : {breakeven}"
        )

        print_progress(
            f"Win Rate     : {win_rate:.2f}%"
        )

        print_progress(
            f"Gross Profit : {gross_profit:.2f}"
        )

        print_progress(
            f"Gross Loss   : {gross_loss:.2f}"
        )

        print_progress(
            f"Net P&L      : {net_pnl:.2f}"
        )

    # ========================================================
    # WRITE EXCEL
    # ========================================================

    write_excel(
        trades_df=trades_df,
        statistics_df=statistics_df,
        daily_summary_df=daily_summary_df,
        monthly_summary_df=monthly_summary_df,
        exit_summary_df=exit_summary_df,
        daily_metadata_df=daily_metadata_df,
        errors_df=errors_df
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print_progress("")
    print_progress("=" * 48)
    print_progress("BACKTEST COMPLETE")
    print_progress("=" * 48)

    print_progress(
        f"Trades: {len(trades_df)}"
    )

    print_progress(
        f"Errors: {len(errors_df)}"
    )

    print_progress(
        f"Expiry dates: {len(expiries)}"
    )

    print_progress(
        f"Output: {OUTPUT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:

        print_progress("")
        print_progress(
            "BACKTEST INTERRUPTED BY USER"
        )

    except Exception as exc:

        print_progress("")
        print_progress(
            "FATAL ERROR"
        )
        print_progress(
            str(exc)
        )
        raise
