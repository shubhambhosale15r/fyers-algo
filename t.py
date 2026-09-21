# ============================================================
# FYERS NIFTY CE/PE SYNTHETIC UNDERLYING LOGGER
# ============================================================
#
# PURPOSE
# -------
# Collect live NIFTY option-chain data so we can later analyse:
#
#     Synthetic Underlying = Strike + CE - PE
#
# for:
#
#     ATM-2
#     ATM-1
#     ATM
#     ATM+1
#     ATM+2
#
# The script also records:
#
#     NIFTY spot
#     CE LTP
#     PE LTP
#     CE bid
#     CE ask
#     PE bid
#     PE ask
#     CE volume
#     PE volume
#     CE OI
#     PE OI
#     synthetic underlying
#     synthetic - actual NIFTY
#
# NO ORDERS ARE PLACED.
#
# ============================================================
#
# REQUIREMENT
# -----------
# pip install fyers-apiv3 pandas
#
# ============================================================
#
# AUTHENTICATION
# --------------
# Put your FYERS access token below.
#
# The access token must correspond to the APP_ID.
#
# Example:
#
# APP_ID = "XXXXXXXXXX-100"
# ACCESS_TOKEN = "your_access_token"
#
# ============================================================

import csv
import os
import sys
import time
import traceback
from datetime import datetime


# ============================================================
# FYERS IMPORT
# ============================================================

try:

    from fyers_apiv3 import fyersModel

except ImportError:

    print()
    print("ERROR: fyers-apiv3 is not installed.")
    print()
    print("Run:")
    print("    py -m pip install fyers-apiv3 pandas")
    print()

    sys.exit(1)


# ============================================================
# USER CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# FYERS APP DETAILS
# ------------------------------------------------------------

APP_ID = "J0ENYFX9ME-200"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcXNMSXVpSHpzdjRwTU1ZdGZjbXFYcm12NXphbmhCYklob3JoR2NIc1dnRllzV1FDbEgyaTdMZS13VDBqOUNuX25PcXhZWFdHY2hOd2ZWdW9vbUZOcTQxQ2QwZXhYeTNxVjJVYWR2N0ZvLTk3aWJEYz0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI0NDIxYTAyNDU1ZWQyOTg3NzZhYzdiMzRmYzAzODJmYjk2MTE4Y2M1NzE3Y2NlMWE0ZWJkYmU0YyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFBMzE0OTUiLCJhcHBUeXBlIjoyMDAsImV4cCI6MTc5MDAzNzAwMCwiaWF0IjoxNzg5OTY0ODQ2LCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3ODk5NjQ4NDYsInN1YiI6ImFjY2Vzc190b2tlbiJ9.HwMYuscLcXY-KfN2nnf6s16AxAggloUAs21KWeWpDxw"


# ------------------------------------------------------------
# MARKET
# ------------------------------------------------------------

UNDERLYING = "NSE:NIFTY50-INDEX"


# ------------------------------------------------------------
# STRIKE SETTINGS
# ------------------------------------------------------------

STRIKE_STEP = 50

ATM_RANGE = 2

# This means:
#
# ATM-2
# ATM-1
# ATM
# ATM+1
# ATM+2


# ------------------------------------------------------------
# SAMPLING
# ------------------------------------------------------------

SAMPLE_INTERVAL_SECONDS = 15


# ------------------------------------------------------------
# OPTION CHAIN API
# ------------------------------------------------------------

# We request a larger chain than we actually need.
#
# This gives us enough strikes around ATM.
#
OPTION_CHAIN_STRIKE_COUNT = 10


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

OUTPUT_FILE = "nifty_ce_pe_synthetic_log.csv"


# ============================================================
# FYERS CLIENT
# ============================================================

print()
print("=" * 100)
print("INITIALIZING FYERS")
print("=" * 100)

print(f"APP_ID       : {APP_ID}")
print(f"UNDERLYING   : {UNDERLYING}")
print()

if APP_ID == "YOUR_APP_ID":

    print("ERROR: Please enter your FYERS APP_ID.")
    sys.exit(1)


if ACCESS_TOKEN == "YOUR_ACCESS_TOKEN":

    print("ERROR: Please enter your FYERS ACCESS_TOKEN.")
    sys.exit(1)


try:

    fyers = fyersModel.FyersModel(
        client_id=APP_ID,
        token=ACCESS_TOKEN,
        is_async=False,
        log_path=""
    )

    print("FYERS CLIENT CREATED")

except Exception as e:

    print()
    print("ERROR CREATING FYERS CLIENT")
    print(repr(e))
    traceback.print_exc()

    sys.exit(1)


# ============================================================
# CSV COLUMNS
# ============================================================

CSV_COLUMNS = [

    "timestamp",
    "unix_timestamp",

    "nifty_spot",

    "expiry",

    "atm_strike",

    "strike_offset",

    "strike",

    "ce_symbol",
    "ce_ltp",
    "ce_bid",
    "ce_ask",
    "ce_volume",
    "ce_oi",

    "pe_symbol",
    "pe_ltp",
    "pe_bid",
    "pe_ask",
    "pe_volume",
    "pe_oi",

    "ce_minus_pe",

    "synthetic_underlying",

    "synthetic_minus_spot",

    "ce_mid",
    "pe_mid",

    "mid_ce_minus_pe",

    "synthetic_using_mid",

    "synthetic_mid_minus_spot",
]


# ============================================================
# CSV INITIALIZATION
# ============================================================

def initialize_csv():

    if os.path.exists(OUTPUT_FILE):

        print(
            f"[CSV] Existing file found: {OUTPUT_FILE}"
        )

        return


    try:

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS
            )

            writer.writeheader()


        print(
            f"[CSV] Created: {OUTPUT_FILE}"
        )


    except Exception as e:

        print()
        print("[CSV ERROR] Could not create CSV")
        print(repr(e))

        sys.exit(1)


# ============================================================
# GET NIFTY SPOT
# ============================================================

def get_nifty_spot():

    try:

        request = {

            "symbols": UNDERLYING

        }


        response = fyers.quotes(request)


        if response is None:

            print("[SPOT] Empty response")

            return None


        if response.get("s") != "ok":

            print("[SPOT ERROR]")
            print(response)

            return None


        data = response.get("d")


        if not data:

            print("[SPOT ERROR] d is empty")

            return None


        first = data[0]

        values = first.get("v", {})


        # FYERS quote response normally provides lp.
        ltp = values.get("lp")


        if ltp is None:

            print(
                "[SPOT ERROR] Could not find 'lp'"
            )

            print(response)

            return None


        return float(ltp)


    except Exception as e:

        print()
        print("[SPOT EXCEPTION]")
        print(repr(e))

        return None


# ============================================================
# GET OPTION CHAIN
# ============================================================

def get_option_chain():

    try:

        request = {

            "symbol": UNDERLYING,

            "strikecount":
                OPTION_CHAIN_STRIKE_COUNT,

            "timestamp": ""

        }


        response = fyers.optionchain(request)


        if response is None:

            print("[CHAIN] Empty response")

            return None


        if response.get("s") != "ok":

            print()
            print("[CHAIN ERROR]")
            print(response)

            return None


        return response


    except Exception as e:

        print()
        print("[CHAIN EXCEPTION]")
        print(repr(e))

        return None


# ============================================================
# DEBUG OPTION CHAIN STRUCTURE
# ============================================================

def inspect_option_chain_structure(response):

    print()
    print("=" * 100)
    print("OPTION CHAIN RESPONSE STRUCTURE")
    print("=" * 100)

    try:

        data = response.get("data", {})

        print()
        print("DATA KEYS:")
        print(list(data.keys()))

        options_chain = data.get(
            "optionsChain",
            []
        )

        print()
        print(
            "OPTIONS CHAIN COUNT:",
            len(options_chain)
        )

        if options_chain:

            print()
            print("FIRST OPTION RECORD:")
            print(options_chain[0])

        expiry_data = data.get(
            "expiryData",
            []
        )

        print()
        print(
            "EXPIRY DATA COUNT:",
            len(expiry_data)
        )

        if expiry_data:

            print()
            print("FIRST EXPIRY RECORD:")
            print(expiry_data[0])

    except Exception as e:

        print()
        print(
            "[DEBUG ERROR]",
            repr(e)
        )

    print("=" * 100)
    print()


# ============================================================
# PARSE OPTION CHAIN
# ============================================================

def parse_option_chain(response):

    try:

        data = response.get(
            "data",
            {}
        )


        options_chain = data.get(
            "optionsChain",
            []
        )


        if not options_chain:

            print(
                "[PARSE ERROR] optionsChain is empty"
            )

            return None


        # ----------------------------------------------------
        # EXPIRY
        # ----------------------------------------------------

        expiry = ""


        expiry_data = data.get(
            "expiryData",
            []
        )


        if expiry_data:

            first_expiry = expiry_data[0]


            if isinstance(
                first_expiry,
                dict
            ):

                expiry = (
                    first_expiry.get("date")
                    or
                    first_expiry.get("expiry")
                    or
                    first_expiry.get("expiryDate")
                    or
                    ""
                )


            elif isinstance(
                first_expiry,
                str
            ):

                expiry = first_expiry


        # ----------------------------------------------------
        # STRIKES
        # ----------------------------------------------------

        options_by_strike = {}


        for item in options_chain:

            if not isinstance(
                item,
                dict
            ):

                continue


            strike = (
                item.get("strike_price")
                or
                item.get("strikePrice")
            )


            if strike is None:

                continue


            try:

                strike = float(strike)

            except Exception:

                continue


            # ------------------------------------------------
            # Determine option type
            # ------------------------------------------------

            option_type = (
                item.get("option_type")
                or
                item.get("optionType")
            )


            # Some responses can expose symbol
            # instead of an explicit option_type.
            #
            # If option_type isn't available, infer
            # from symbol suffix where possible.

            symbol = item.get(
                "symbol",
                ""
            )


            if option_type is None:

                symbol_upper = symbol.upper()

                if "CE" in symbol_upper:

                    option_type = "CE"

                elif "PE" in symbol_upper:

                    option_type = "PE"


            if option_type not in (
                "CE",
                "PE"
            ):

                continue


            # ------------------------------------------------
            # Create strike entry
            # ------------------------------------------------

            if strike not in options_by_strike:

                options_by_strike[strike] = {

                    "CE": None,

                    "PE": None

                }


            # ------------------------------------------------
            # Store complete option record
            # ------------------------------------------------

            options_by_strike[
                strike
            ][option_type] = item


        if not options_by_strike:

            print(
                "[PARSE ERROR] No option strikes found"
            )

            return None


        return (
            expiry,
            options_by_strike
        )


    except Exception as e:

        print()
        print(
            "[PARSE EXCEPTION]",
            repr(e)
        )

        traceback.print_exc()

        return None


# ============================================================
# GET FIELD FROM OPTION RECORD
# ============================================================

def get_number(
    record,
    possible_keys
):

    if record is None:

        return None


    for key in possible_keys:

        value = record.get(key)


        if value is None:

            continue


        try:

            return float(value)

        except Exception:

            continue


    return None


# ============================================================
# GET SYMBOL
# ============================================================

def get_symbol(record):

    if record is None:

        return ""


    return str(
        record.get(
            "symbol",
            ""
        )
    )


# ============================================================
# FIND ATM STRIKE
# ============================================================

def find_atm_strike(
    nifty_spot,
    available_strikes
):

    if not available_strikes:

        return None


    return min(

        available_strikes,

        key=lambda strike:
            abs(
                strike -
                nifty_spot
            )

    )


# ============================================================
# FIND TARGET STRIKES
# ============================================================

def get_target_strikes(
    atm_strike,
    available_strikes
):

    target = []


    for offset in range(
        -ATM_RANGE,
        ATM_RANGE + 1
    ):

        expected_strike = (
            atm_strike
            +
            offset * STRIKE_STEP
        )


        # ----------------------------------------------------
        # First try exact expected strike.
        # ----------------------------------------------------

        matching = None


        for available in available_strikes:

            if abs(
                available -
                expected_strike
            ) < 0.01:

                matching = available

                break


        if matching is not None:

            target.append(
                (
                    offset,
                    matching
                )
            )

        else:

            print(
                f"[WARNING] Expected strike "
                f"{expected_strike} not found."
            )


    return target


# ============================================================
# PROCESS ONE SNAPSHOT
# ============================================================

def process_snapshot():

    timestamp = datetime.now()

    unix_timestamp = int(
        time.time()
    )


    print()
    print("=" * 110)

    print(
        "[SNAPSHOT]",
        timestamp.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    # ========================================================
    # NIFTY SPOT
    # ========================================================

    nifty_spot = get_nifty_spot()


    if nifty_spot is None:

        print(
            "[SKIP] NIFTY spot unavailable"
        )

        return


    print(
        f"[NIFTY] Spot = {nifty_spot:.2f}"
    )


    # ========================================================
    # OPTION CHAIN
    # ========================================================

    response = get_option_chain()


    if response is None:

        print(
            "[SKIP] Option chain unavailable"
        )

        return


    parsed = parse_option_chain(
        response
    )


    if parsed is None:

        print(
            "[SKIP] Could not parse option chain"
        )

        return


    expiry, options_by_strike = parsed


    available_strikes = sorted(
        options_by_strike.keys()
    )


    print(
        f"[CHAIN] Strikes received = "
        f"{len(available_strikes)}"
    )


    # ========================================================
    # ATM
    # ========================================================

    atm_strike = find_atm_strike(
        nifty_spot,
        available_strikes
    )


    if atm_strike is None:

        print(
            "[SKIP] Could not determine ATM"
        )

        return


    print(
        f"[ATM] {atm_strike:.0f}"
    )


    print(
        f"[EXPIRY] {expiry}"
    )


    # ========================================================
    # TARGET STRIKES
    # ========================================================

    target_strikes = get_target_strikes(
        atm_strike,
        available_strikes
    )


    if not target_strikes:

        print(
            "[SKIP] No target strikes"
        )

        return


    # ========================================================
    # DISPLAY HEADER
    # ========================================================

    print()

    print(
        f"{'Offset':>6} "
        f"{'Strike':>8} "
        f"{'CE':>9} "
        f"{'PE':>9} "
        f"{'CE-PE':>9} "
        f"{'Synthetic':>12} "
        f"{'Syn-Spot':>10}"
    )

    print("-" * 80)


    csv_rows = []


    # ========================================================
    # EACH STRIKE
    # ========================================================

    for offset, strike in target_strikes:

        strike_data = (
            options_by_strike
            .get(strike)
        )


        if strike_data is None:

            continue


        ce = strike_data.get(
            "CE"
        )


        pe = strike_data.get(
            "PE"
        )


        if ce is None:

            print(
                f"[MISSING CE] {strike:.0f}"
            )

            continue


        if pe is None:

            print(
                f"[MISSING PE] {strike:.0f}"
            )

            continue


        # ====================================================
        # CE DATA
        # ====================================================

        ce_ltp = get_number(
            ce,
            [
                "ltp",
                "lp"
            ]
        )


        ce_bid = get_number(
            ce,
            [
                "bid",
                "bid_price",
                "bp"
            ]
        )


        ce_ask = get_number(
            ce,
            [
                "ask",
                "ask_price",
                "ap"
            ]
        )


        ce_volume = get_number(
            ce,
            [
                "volume",
                "vol_traded_today",
                "v"
            ]
        )


        ce_oi = get_number(
            ce,
            [
                "oi",
                "open_interest"
            ]
        )


        # ====================================================
        # PE DATA
        # ====================================================

        pe_ltp = get_number(
            pe,
            [
                "ltp",
                "lp"
            ]
        )


        pe_bid = get_number(
            pe,
            [
                "bid",
                "bid_price",
                "bp"
            ]
        )


        pe_ask = get_number(
            pe,
            [
                "ask",
                "ask_price",
                "ap"
            ]
        )


        pe_volume = get_number(
            pe,
            [
                "volume",
                "vol_traded_today",
                "v"
            ]
        )


        pe_oi = get_number(
            pe,
            [
                "oi",
                "open_interest"
            ]
        )


        # ====================================================
        # REQUIRE LTP FOR SYNTHETIC
        # ====================================================

        if ce_ltp is None:

            print(
                f"[MISSING CE LTP] {strike:.0f}"
            )

            continue


        if pe_ltp is None:

            print(
                f"[MISSING PE LTP] {strike:.0f}"
            )

            continue


        # ====================================================
        # SYNTHETIC USING LTP
        # ====================================================

        ce_minus_pe = (
            ce_ltp -
            pe_ltp
        )


        synthetic_underlying = (
            strike +
            ce_minus_pe
        )


        synthetic_minus_spot = (
            synthetic_underlying -
            nifty_spot
        )


        # ====================================================
        # MID PRICES
        # ====================================================

        ce_mid = None

        pe_mid = None


        if (
            ce_bid is not None
            and
            ce_ask is not None
            and
            ce_bid >= 0
            and
            ce_ask >= 0
        ):

            ce_mid = (
                ce_bid +
                ce_ask
            ) / 2.0


        if (
            pe_bid is not None
            and
            pe_ask is not None
            and
            pe_bid >= 0
            and
            pe_ask >= 0
        ):

            pe_mid = (
                pe_bid +
                pe_ask
            ) / 2.0


        mid_ce_minus_pe = None

        synthetic_using_mid = None

        synthetic_mid_minus_spot = None


        if (
            ce_mid is not None
            and
            pe_mid is not None
        ):

            mid_ce_minus_pe = (
                ce_mid -
                pe_mid
            )


            synthetic_using_mid = (
                strike +
                mid_ce_minus_pe
            )


            synthetic_mid_minus_spot = (
                synthetic_using_mid -
                nifty_spot
            )


        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"{offset:>6} "
            f"{strike:>8.0f} "
            f"{ce_ltp:>9.2f} "
            f"{pe_ltp:>9.2f} "
            f"{ce_minus_pe:>9.2f} "
            f"{synthetic_underlying:>12.2f} "
            f"{synthetic_minus_spot:>10.2f}"
        )


        # ====================================================
        # CSV ROW
        # ====================================================

        csv_rows.append({

            "timestamp":
                timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "unix_timestamp":
                unix_timestamp,

            "nifty_spot":
                nifty_spot,

            "expiry":
                expiry,

            "atm_strike":
                atm_strike,

            "strike_offset":
                offset,

            "strike":
                strike,

            "ce_symbol":
                get_symbol(ce),

            "ce_ltp":
                ce_ltp,

            "ce_bid":
                ce_bid,

            "ce_ask":
                ce_ask,

            "ce_volume":
                ce_volume,

            "ce_oi":
                ce_oi,

            "pe_symbol":
                get_symbol(pe),

            "pe_ltp":
                pe_ltp,

            "pe_bid":
                pe_bid,

            "pe_ask":
                pe_ask,

            "pe_volume":
                pe_volume,

            "pe_oi":
                pe_oi,

            "ce_minus_pe":
                ce_minus_pe,

            "synthetic_underlying":
                synthetic_underlying,

            "synthetic_minus_spot":
                synthetic_minus_spot,

            "ce_mid":
                ce_mid,

            "pe_mid":
                pe_mid,

            "mid_ce_minus_pe":
                mid_ce_minus_pe,

            "synthetic_using_mid":
                synthetic_using_mid,

            "synthetic_mid_minus_spot":
                synthetic_mid_minus_spot,

        })


    # ========================================================
    # WRITE CSV
    # ========================================================

    if csv_rows:

        try:

            with open(
                OUTPUT_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=CSV_COLUMNS
                )

                writer.writerows(
                    csv_rows
                )


            print()
            print(
                f"[CSV] Wrote "
                f"{len(csv_rows)} rows"
            )


        except Exception as e:

            print()
            print(
                "[CSV WRITE ERROR]"
            )

            print(
                repr(e)
            )


    else:

        print()
        print(
            "[CSV] No rows written"
        )


# ============================================================
# WAIT UNTIL NEXT INTERVAL
# ============================================================

def wait_until_next_interval():

    now = time.time()


    next_time = (
        (
            int(now)
            //
            SAMPLE_INTERVAL_SECONDS
        )
        + 1
    ) * SAMPLE_INTERVAL_SECONDS


    sleep_seconds = (
        next_time -
        now
    )


    if sleep_seconds > 0:

        time.sleep(
            sleep_seconds
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print("NIFTY CE/PE SYNTHETIC UNDERLYING LOGGER")
    print("=" * 110)

    print()
    print("Underlying:")
    print(
        f"    {UNDERLYING}"
    )

    print()
    print("Strikes:")
    print(
        f"    ATM-{ATM_RANGE} "
        f"to ATM+{ATM_RANGE}"
    )

    print()
    print("Strike step:")
    print(
        f"    {STRIKE_STEP}"
    )

    print()
    print("Sampling:")
    print(
        f"    Every {SAMPLE_INTERVAL_SECONDS} seconds"
    )

    print()
    print("Output:")
    print(
        f"    {OUTPUT_FILE}"
    )

    print()
    print("Orders:")
    print("    NONE")

    print()
    print("=" * 110)


    initialize_csv()


    # ========================================================
    # INITIAL TEST REQUEST
    # ========================================================

    print()
    print(
        "[TEST] Testing NIFTY quote..."
    )


    test_spot = get_nifty_spot()


    if test_spot is None:

        print()
        print(
            "ERROR: NIFTY quote test failed."
        )

        print(
            "Check APP_ID and ACCESS_TOKEN."
        )

        sys.exit(1)


    print(
        f"[TEST] NIFTY = {test_spot:.2f}"
    )


    print()
    print(
        "[TEST] Testing option chain..."
    )


    test_chain = get_option_chain()


    if test_chain is None:

        print()
        print(
            "ERROR: Option-chain test failed."
        )

        sys.exit(1)


    print(
        "[TEST] Option chain request successful."
    )


    # --------------------------------------------------------
    # Print structure ONCE so we can diagnose API changes.
    # --------------------------------------------------------

    inspect_option_chain_structure(
        test_chain
    )


    # ========================================================
    # MAIN LOOP
    # ========================================================

    print()
    print(
        "STARTING LOGGER"
    )

    print(
        "Press Ctrl+C to stop."
    )

    print()


    while True:

        try:

            wait_until_next_interval()

            process_snapshot()


        except KeyboardInterrupt:

            print()
            print("=" * 110)
            print("LOGGER STOPPED")
            print("=" * 110)

            break


        except Exception as e:

            print()
            print("=" * 110)
            print("UNEXPECTED MAIN LOOP ERROR")
            print("=" * 110)

            print(
                repr(e)
            )

            traceback.print_exc()

            print(
                "Continuing..."
            )

            time.sleep(1)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()
