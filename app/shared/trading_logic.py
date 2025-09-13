import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from fyers_apiv3 import fyersModel
import logging
import uuid
from functools import lru_cache
import os
import subprocess
import sys
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

# Constants
class Config:
    REFRESH_INTERVAL = 30  # seconds
    API_RATE_LIMIT = 200
    API_WINDOW = 60  # seconds
    LOT_SIZE = 75
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    # Exit all positions when signal = SIDEWAYS
    SIDEWAYS_EXIT = True

# Initialize logging
logging.basicConfig(
    filename="trading_app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# Error handling decorator
def handle_errors(func):
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < Config.MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logging.error(f"Error in {func.__name__} (attempt {retries}): {str(e)}", exc_info=True)
                if retries < Config.MAX_RETRIES:
                    time.sleep(Config.RETRY_DELAY)
                else:
                    print(f"❌ Operation failed after {Config.MAX_RETRIES} attempts: {str(e)}")
                    return None

    return wrapper


# This will be replaced by a more robust state manager in the web app
class SessionState:
    def __init__(self):
        self.option_chain_api_count = 0
        self.quote_api_count = 0
        self.api_window_start = datetime.now()
        self.last_signal = None
        self.last_signal_order_ids = []
        self.live_last_signal = None
        self.live_last_signal_order_ids = []
        self.signal_history = []
        self.cid = ""
        self.token = ""
        self._hb_lock = False

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_credentials(session_state):
    """Load credentials from JSON file or environment variables"""
    try:
        with open('fyers_credentials.json', 'r') as f:
            creds = json.load(f)
            session_state.cid = creds.get('client_id', '')
            session_state.token = creds.get('access_token', '')
            logging.info("Credentials loaded from JSON file")
    except FileNotFoundError:
        logging.warning("JSON file not found, checking environment variables")
        session_state.cid = os.getenv("FYERS_CID", session_state.cid)
        session_state.token = os.getenv("FYERS_TOKEN", session_state.token)
    except Exception as e:
        logging.error(f"Error loading credentials: {str(e)}")
        session_state.cid = os.getenv("FYERS_CID", session_state.cid)
        session_state.token = os.getenv("FYERS_TOKEN", session_state.token)


def enforce_rate_limit(session_state):
    now = datetime.now()
    window = now - session_state.api_window_start
    total_calls = session_state.option_chain_api_count + session_state.quote_api_count

    if window > timedelta(seconds=Config.API_WINDOW):
        session_state.api_window_start = now
        session_state.option_chain_api_count = 0
        session_state.quote_api_count = 0
    elif total_calls >= Config.API_RATE_LIMIT:
        wait = Config.API_WINDOW - window.seconds
        logging.warning(f"Approaching rate limit, waiting {wait}s...")
        time.sleep(wait)
        session_state.api_window_start = datetime.now()
        session_state.option_chain_api_count = 0
        session_state.quote_api_count = 0


@handle_errors
@lru_cache(maxsize=32)
def get_option_chain_data(session_state, cid, token, sym, expiry_ts=""):
    enforce_rate_limit(session_state)
    session_state.option_chain_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.optionchain(data={"symbol": sym, "timestamp": expiry_ts})
    if not resp or resp.get("code") != 200:
        error_msg = f"Option chain error ({expiry_ts or 'nearest'}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        return None
    return resp["data"]["optionsChain"]


@handle_errors
def get_underlying_ltp(session_state, cid, token, sym):
    enforce_rate_limit(session_state)
    session_state.quote_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": sym})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error ({resp.get('code')}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        return None
    return resp["d"][0]["v"]["lp"]


signal_history = []
MAX_HISTORY = 3
MAJORITY_THRESHOLD = 3

def majority_signal(history, threshold=MAJORITY_THRESHOLD):
    if not history:
        return "SIDEWAYS"
    counter = Counter(history)
    most_common_signal, count = counter.most_common(1)[0]
    if count >= threshold:
        return most_common_signal
    return "SIDEWAYS"

def compute_signals(
    merged_df,
    atm_strike,
    ltp,
    min_range_pct: float = 1.0,
    tolerance_pct: float = 1.0,
    fyers_api=None,
):
    try:
        def safe_val(series, default=0.0):
            return float(series.iloc[0]) if not series.empty and not pd.isna(series.iloc[0]) else float(default)

        atm_row = merged_df[merged_df["Strike"] == atm_strike]
        atm_ce_price = safe_val(atm_row["CE_ltp"])
        atm_pe_price = safe_val(atm_row["PE_ltp"])

        price_range = atm_ce_price + atm_pe_price
        min_range_abs = float(ltp) * (min_range_pct / 100.0)

        if price_range >= min_range_abs:
            lower_bound = atm_strike - price_range
            upper_bound = atm_strike + price_range
        else:
            lower_bound = ltp * (1 - min_range_pct / 100.0)
            upper_bound = ltp * (1 + min_range_pct / 100.0)

        strikes = sorted(merged_df["Strike"].dropna().unique())

        def round_to_nearest_strike(value, strikes):
            if not strikes:
                return value
            return min(strikes, key=lambda x: abs(x - value))

        lower_bound = round_to_nearest_strike(lower_bound, strikes)
        upper_bound = round_to_nearest_strike(upper_bound, strikes)

        range_strikes = [s for s in strikes if lower_bound <= s <= upper_bound]
        if not range_strikes:
            return "SIDEWAYS"

        range_df = merged_df[merged_df["Strike"].isin(range_strikes)].copy()

        if fyers_api and {"CE_symbol", "PE_symbol"} <= set(range_df.columns):
            try:
                symbols = list(set(range_df["CE_symbol"].dropna().tolist() + range_df["PE_symbol"].dropna().tolist()))
                symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
                quotes = {}

                for i in range(0, len(symbols), 50):
                    batch = symbols[i:i + 50]
                    response = fyers_api.quotes({"symbols": ",".join(batch)})
                    if response.get("code") == 200:
                        quotes.update({item["n"]: item["v"] for item in response["d"] if item["s"] == "ok"})

                range_df["CE_vs_atp"] = "BELOW"
                range_df["PE_vs_atp"] = "BELOW"

                for idx, row in range_df.iterrows():
                    if row["CE_symbol"] in quotes:
                        ce_data = quotes[row["CE_symbol"]]
                        ce_ltp = float(ce_data.get("lp", 0) or 0)
                        atp = float(ce_data.get("atp", 0) or 0)
                        ce_high = float(ce_data.get("high_price", 0) or 0)
                        ce_low = float(ce_data.get("low_price", 0) or 0)
                        if atp > 0:
                            range_df.at[idx, "CE_vs_atp"] = "ABOVE" if ce_ltp > atp else "BELOW"
                        range_df.at[idx, "CE_ltp"] = ce_ltp
                        range_df.at[idx, "CE_high"] = ce_high
                        range_df.at[idx, "CE_low"] = ce_low

                    if row["PE_symbol"] in quotes:
                        pe_data = quotes[row["PE_symbol"]]
                        pe_ltp = float(pe_data.get("lp", 0) or 0)
                        atp = float(pe_data.get("atp", 0) or 0)
                        pe_high = float(pe_data.get("high_price", 0) or 0)
                        pe_low = float(pe_data.get("low_price", 0) or 0)
                        if atp > 0:
                            range_df.at[idx, "PE_vs_atp"] = "ABOVE" if pe_ltp > atp else "BELOW"
                        range_df.at[idx, "PE_ltp"] = pe_ltp
                        range_df.at[idx, "PE_high"] = pe_high
                        range_df.at[idx, "PE_low"] = pe_low

                range_df["CE_moneyflow"] = range_df["CE_ltp"].astype(float) * range_df["CE_Vol"].astype(float)
                range_df["PE_moneyflow"] = range_df["PE_ltp"].astype(float) * range_df["PE_Vol"].astype(float)

            except Exception as e:
                logging.warning(f"Fyers quotes error: {str(e)}")

        atm_idx = range_strikes.index(atm_strike) if atm_strike in range_strikes else len(range_strikes) // 2
        num_strikes = min(len(range_strikes[:atm_idx]), len(range_strikes[atm_idx + 1:]))
        itm_ce_strikes = range_strikes[atm_idx - num_strikes:atm_idx]
        itm_pe_strikes = range_strikes[atm_idx + 1:atm_idx + 1 + num_strikes]

        ce_net = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_moneyflow"].sum()
        pe_net = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_moneyflow"].sum()
        net_flow = (ce_net - pe_net) / (ce_net + pe_net) if (ce_net + pe_net) != 0 else 0.0

        ce_above_atp = range_df[(range_df["Strike"].isin(itm_ce_strikes)) & (range_df["CE_vs_atp"] == "ABOVE")].shape[0]
        pe_above_atp = range_df[(range_df["Strike"].isin(itm_pe_strikes)) & (range_df["PE_vs_atp"] == "ABOVE")].shape[0]
        direction_ratio = (ce_above_atp - pe_above_atp) / max(len(itm_ce_strikes) + len(itm_pe_strikes), 1)

        range_df["CE_range_placement"] = np.where(
            (range_df["CE_high"] - range_df["CE_low"]) > 0,
            (range_df["CE_ltp"] - range_df["CE_low"]) / (range_df["CE_high"] - range_df["CE_low"]),
            0,
        )
        range_df["PE_range_placement"] = np.where(
            (range_df["PE_high"] - range_df["PE_low"]) > 0,
            (range_df["PE_ltp"] - range_df["PE_low"]) / (range_df["PE_high"] - range_df["PE_low"]),
            0,
        )

        ce_placement_avg = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_range_placement"].mean()
        pe_placement_avg = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_range_placement"].mean()

        CE_HIGH_THRESHOLD = 0.6
        CE_LOW_THRESHOLD = 0.4
        PE_HIGH_THRESHOLD = 0.6
        PE_LOW_THRESHOLD = 0.4

        raw_signal = "SIDEWAYS"
        if direction_ratio >= 0.5 and ce_placement_avg > CE_HIGH_THRESHOLD and pe_placement_avg < PE_LOW_THRESHOLD:
            raw_signal = "BUY"
        elif direction_ratio <= -0.5 and pe_placement_avg > PE_HIGH_THRESHOLD and ce_placement_avg < CE_LOW_THRESHOLD:
            raw_signal = "SELL"

        global signal_history
        signal_history.append(raw_signal)

        if len(signal_history) > MAX_HISTORY:
            signal_history.pop(0)

        final_signal = majority_signal(signal_history)

        return final_signal

    except Exception as e:
        logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
        return "SIDEWAYS"


def select_strikes_atm_half_price(chain, atm_strike):
    try:
        df = pd.DataFrame(chain)
        if df.empty or atm_strike is None:
            return None, None, None, None

        ce = df[df["option_type"] == "CE"].copy()
        pe = df[df["option_type"] == "PE"].copy()

        atm_pe_row = pe[pe["strike_price"] == atm_strike]
        atm_ce_row = ce[ce["strike_price"] == atm_strike]

        buy_pe_row = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
        buy_ce_row = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None

        sell_pe_row, sell_ce_row = None, None

        if buy_pe_row and "ltp" in buy_pe_row and buy_pe_row["ltp"] > 0:
            target = buy_pe_row["ltp"] * 0.5
            pe_candidates = pe[pe["strike_price"] < atm_strike].copy()
            pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
            if not pe_candidates.empty:
                sell_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        if buy_ce_row and "ltp" in buy_ce_row and buy_ce_row["ltp"] > 0:
            target = buy_ce_row["ltp"] * 0.5
            ce_candidates = ce[ce["strike_price"] > atm_strike].copy()
            ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target).abs()
            if not ce_candidates.empty:
                sell_ce_row = ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        return buy_pe_row, sell_pe_row, buy_ce_row, sell_ce_row

    except Exception as e:
        logging.error(f"Error in select_strikes_atm_half_price: {str(e)}", exc_info=True)
        return None, None, None, None


@handle_errors
def place_order_and_check(fyers, order):
    try:
        resp = fyers.place_order(data=order)
        logging.info(f"Order placed for {order['symbol']}: {resp}")
        if resp.get("s") == "ok":
            oid = resp.get("id")
            time.sleep(2)
            status = get_order_status_by_id(fyers, oid)
            if status in [1, 5]:
                logging.error(f"Order {oid} was rejected or canceled. Status: {status}")
                return None, status
            logging.info(f"Order {oid} status: {status}")
            return oid, status
        else:
            logging.error(f"Order failed: {resp}")
            return None, None
    except Exception as e:
        logging.exception(f"Place order error: {e}")
        raise

def has_open_orders_for_last_signal(fyers, session_state):
    ids = session_state.live_last_signal_order_ids
    if not ids:
        return False
    try:
        ob = fyers.orderbook().get("orderBook", [])
        return any(o.get("id") in ids and o.get("status") == 2 for o in ob)
    except Exception as e:
        logging.warning(f"Could not check open orders: {e}")
        return False

def handle_basket_orders_atomic(session_state, app_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers):
    if session_state._hb_lock:
        return []
    session_state._hb_lock = True

    placed_ids = []
    try:
        lot = Config.LOT_SIZE
        current_signal = session_state.last_signal

        def _wait_until_flat(timeout=28, interval=2):
            end = time.time() + timeout
            while time.time() < end:
                pos_resp = fyers.positions()
                if pos_resp.get("s") == "ok":
                    if not [p for p in pos_resp.get("netPositions", []) if abs(p.get("netQty", 0)) != 0]:
                        return True
                time.sleep(interval)
            return False

        if signal == "SIDEWAYS":
            if getattr(Config, "SIDEWAYS_EXIT", True):
                try:
                    pos_resp = fyers.positions()
                    if pos_resp.get("s") == "ok" and [p for p in pos_resp.get("netPositions", []) if abs(p.get("netQty", 0)) != 0]:
                        fyers.exit_positions(data={"segment": [11], "side": [-1], "productType": ["ALL"]})
                        fyers.exit_positions(data={"segment": [11], "side": [1], "productType": ["ALL"]})
                        _wait_until_flat(timeout=60, interval=2)
                except Exception as e:
                    logging.error(f"Exception while exiting on SIDEWAYS: {str(e)}")
            return []

        signal_map = {
            "BUY": {"enter_hedge": buy_ce, "enter_short": sell_ce},
            "SELL": {"enter_hedge": buy_pe, "enter_short": sell_pe}
        }

        pos_resp = fyers.positions()
        net_positions = []
        if pos_resp.get("s") == "ok":
            net_positions = [p for p in pos_resp.get("netPositions", []) if abs(p.get("netQty", 0)) != 0]

        if current_signal in ["BUY", "SELL"] and current_signal != signal and net_positions:
            try:
                fyers.exit_positions(data={"segment": [11], "side": [-1], "productType": ["ALL"]})
                fyers.exit_positions(data={"segment": [11], "side": [1], "productType": ["ALL"]})
                if not _wait_until_flat(timeout=60, interval=2):
                    return placed_ids
            except Exception as e:
                logging.error(f"Exception during exit sequence: {str(e)}")
                return placed_ids

        pos_resp = fyers.positions()
        current_positions = []
        if pos_resp.get("s") == "ok":
            current_positions = [p for p in pos_resp.get("netPositions", []) if abs(p.get("netQty", 0)) != 0]

        if current_positions:
            return placed_ids

        enter_hedge = signal_map[signal]["enter_hedge"]
        enter_short = signal_map[signal]["enter_short"]

        entry_basket = [
            {"symbol": enter_hedge["symbol"], "qty": lot, "type": 2, "side": 1, "productType": "MARGIN", "validity": "DAY"},
            {"symbol": enter_short["symbol"], "qty": lot, "type": 2, "side": -1, "productType": "MARGIN", "validity": "DAY"}
        ]

        entry_resp = fyers.place_basket_orders(data=entry_basket)

        if entry_resp and entry_resp.get("s") == "ok":
            for item in entry_resp.get("data", []):
                body = item.get("body", {})
                order_details = {
                    "timestamp": datetime.now(pytz.timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S'),
                    "symbol": item.get("symbol"),
                    "status": body.get("message"),
                    "id": body.get("id", "N/A")
                }
                app_state.add_order(order_details)
                if body.get("s") == "ok" and body.get("id"):
                    placed_ids.append(body["id"])
            session_state.last_signal = signal
        else:
            logging.error(f"Basket order FAILED. Reason: {entry_resp.get('message', 'Unknown error')}")

        return placed_ids
    finally:
        session_state._hb_lock = False

@handle_errors
def update_all_data(session_state, cid, token, sym):
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    enforce_rate_limit(session_state)
    session_state.option_chain_api_count += 1

    meta = fy.optionchain(data={"symbol": sym, "timestamp": ""})
    if meta.get("code") != 200:
        logging.error("Failed to fetch expiry data")
        return None, None, None

    exp = sorted(
        [(e["date"], e.get("expiry")) for e in meta["data"]["expiryData"]],
        key=lambda x: datetime.strptime(x[0], "%d-%m-%Y")
    )
    today = datetime.today().date()
    dates = [datetime.strptime(d, "%d-%m-%Y").date() for d, _ in exp]
    idx = next((i for i, d in enumerate(dates) if d >= today), 0)
    curr_date, curr_ts = exp[idx]

    ltp = get_underlying_ltp(session_state, cid, token, sym)
    chain = get_option_chain_data(session_state, cid, token, sym, expiry_ts=curr_ts)

    return ltp, chain, curr_date

def format_and_show(chain, title, ltp, show_signals=False, fyers_api=None):
    df = pd.DataFrame(chain)
    if df.empty:
        return None, None, None

    ce = df[df["option_type"] == "CE"]
    pe = df[df["option_type"] == "PE"]

    cols = ["symbol", "strike_price", "oi", "volume", "ltp", "ask", "bid", "ltpch", "ltpchp", "oich", "oichp", "prev_oi"]
    ce_df = ce[cols].rename(columns={"strike_price": "Strike", "oi": "OI", "volume": "Vol", "ltpch": "LTPCh", "ltpchp": "LTPChP", "oich": "OICh", "oichp": "OIChP", "prev_oi": "PrevOI"})
    pe_df = pe[cols].rename(columns={"strike_price": "Strike", "oi": "OI", "volume": "Vol", "ltpch": "LTPCh", "ltpchp": "LTPChP", "oich": "OICh", "oichp": "OIChP", "prev_oi": "PrevOI"})

    merged = pd.merge(
        ce_df.add_prefix("CE_"),
        pe_df.add_prefix("PE_"),
        left_on="CE_Strike",
        right_on="PE_Strike",
        how="outer"
    ).rename(columns={"CE_Strike": "Strike"}).drop("PE_Strike", axis=1)

    atm = None
    if ltp is not None and not merged["Strike"].dropna().empty:
        atm = min(merged["Strike"].dropna(), key=lambda x: abs(x - ltp))

    signal = None
    if show_signals and atm is not None:
        signal = compute_signals(merged, atm, ltp, fyers_api=fyers_api)

    return signal, atm, merged

def trading_loop(session_state, app_state, symbol: str):
    while app_state.is_running():
        try:
            ltp, chain, curr_date = update_all_data(session_state, session_state.cid, session_state.token, symbol)
            ist = pytz.timezone("Asia/Kolkata")
            timestamp = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

            if ltp and chain:
                fy = fyersModel.FyersModel(client_id=session_state.cid, token=session_state.token, is_async=False, log_path="")
                signal, atm, merged_df = format_and_show(chain, f"Current Expiry: {curr_date}", ltp, show_signals=True, fyers_api=fy)

                app_state.add_log(f"[{timestamp}] LTP: {ltp}, ATM: {atm}, Signal: {signal}")
                app_state.update_live_data(ltp, atm, signal, merged_df.to_dict('records'), timestamp)

                if signal:
                    sell_pe, buy_pe, sell_ce, buy_ce = select_strikes_atm_half_price(chain, atm)

                    valid_buy = (sell_pe is not None and buy_pe is not None)
                    valid_sell = (sell_ce is not None and buy_ce is not None)

                    if signal == "BUY" and not valid_buy:
                        app_state.add_log("Missing option data for BUY signal. Unable to place trade.")
                    elif signal == "SELL" and not valid_sell:
                        app_state.add_log("Missing option data for SELL signal. Unable to place trade.")
                    elif signal in ("BUY", "SELL"):
                        if session_state.live_last_signal == signal and has_open_orders_for_last_signal(fy, session_state):
                            app_state.add_log(f"Basket for {signal} already open.")
                        else:
                            ids = handle_basket_orders_atomic(session_state, app_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
                            session_state.live_last_signal = signal
                            session_state.live_last_signal_order_ids = ids
                            app_state.add_log(f"Placed {signal} order. IDs: {ids}")
                    elif signal == "SIDEWAYS":
                        handle_basket_orders_atomic(session_state, app_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
                        app_state.add_log("Sideways signal. Exiting positions if configured.")

            time.sleep(Config.REFRESH_INTERVAL)
        except Exception as e:
            app_state.add_log(f"Error in trading loop: {e}")
            logging.error(f"Error in trading loop: {e}", exc_info=True)
            time.sleep(Config.REFRESH_INTERVAL)
