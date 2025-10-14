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
    REFRESH_INTERVAL = 60 # seconds
    API_RATE_LIMIT = 200
    API_WINDOW = 60  # seconds
    LOT_SIZE = 75  # nifty lot size
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    


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


# Initialize all session state in one place
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
        self.signal_buffer = []  # For majority signal calculation
        self.cid = ""
        self.token = ""
        self._hb_lock = False

    def get(self, key, default=None):
        """Dictionary-like get method for session state attributes"""
        return getattr(self, key, default)


# Load credentials from environment variables
def load_credentials(session_state):
    """Load credentials from JSON file or environment variables"""
    try:
        # Try to load from JSON file first
        with open('fyers_credentials.json', 'r') as f:
            creds = json.load(f)
            session_state.cid = creds.get('client_id', '')
            session_state.token = creds.get('access_token', '')
            print("✅ Credentials loaded from JSON file")
    except FileNotFoundError:
        # Fall back to environment variables if file doesn't exist
        print("⚠️ JSON file not found, checking environment variables")
        session_state.cid = os.getenv("FYERS_CID", session_state.cid)
        session_state.token = os.getenv("FYERS_TOKEN", session_state.token)
    except Exception as e:
        print(f"❌ Error loading credentials: {str(e)}")
        # Fall back to environment variables
        session_state.cid = os.getenv("FYERS CID", session_state.cid)
        session_state.token = os.getenv("FYERS_TOKEN", session_state.token)


# Rate limiting with improved logic
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
        print(f"⚠️ Approaching rate limit, waiting {wait}s...")
        time.sleep(wait)
        session_state.api_window_start = datetime.now()
        session_state.option_chain_api_count = 0
        session_state.quote_api_count = 0


# API calls with retry logic
@handle_errors
@lru_cache(maxsize=32)
def get_option_chain_data(session_state, cid, token, sym, expiry_ts=""):
    print(f"🔍 [DEBUG] Fetching option chain data for {sym}, expiry: {expiry_ts or 'nearest'}")
    enforce_rate_limit(session_state)
    session_state.option_chain_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.optionchain(data={"symbol": sym, "timestamp": expiry_ts})
    if not resp or resp.get("code") != 200:
        error_msg = f"Option chain error ({expiry_ts or 'nearest'}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        print(f"❌ {error_msg}")
        return None
    print(f"✅ [DEBUG] Option chain data fetched successfully")
    return resp["data"]["optionsChain"]


@handle_errors
def get_underlying_ltp(session_state, cid, token, sym):
    print(f"🔍 [DEBUG] Fetching underlying LTP for {sym}")
    enforce_rate_limit(session_state)
    session_state.quote_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": sym})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error ({resp.get('code')}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        print(f"❌ {error_msg}")
        return None
    ltp = resp["d"][0]["v"]["lp"]
    print(f"✅ [DEBUG] Underlying LTP: {ltp}")
    return ltp


@handle_errors
def get_symbol_ltp(session_state, cid, token, symbol):
    print(f"🔍 [DEBUG] Fetching LTP for {symbol}")
    enforce_rate_limit(session_state)
    session_state.quote_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": symbol})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error for {symbol}: {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        print(f"❌ [error_msg]")
        return None
    ltp = resp["d"][0]["v"]["lp"]
    print(f"✅ [DEBUG] Symbol LTP: {ltp}")
    return ltp



MAX_HISTORY = 3  # last N signals for majority check
MAJORITY_THRESHOLD = 3  # number of same signals to confirm majority

def majority_signal(history, threshold=MAJORITY_THRESHOLD):
    """
    Determine majority signal from the history.
    Returns the majority signal if all signals in buffer are the same, else HOLD.
    """
    if not history or len(history) < threshold:
        return "HOLD"

    if all(signal == history[0] for signal in history):
        return history[0]

    return "HOLD"


def compute_signals(
        merged_df,
        atm_strike,
        ltp,
        fyers_api=None,
        session_state=None
):
    """
    Compute intraday trading signals using ATM, ATM+1, and ATM-1 strikes.
    Returns: "BUY", "SELL", "SIDEWAYS", or "HOLD"
    """
    print(f"🔍 [DEBUG] Computing signals for ATM strike: {atm_strike}, LTP: {ltp}")

    try:
        # ----------------------------
        # Helper
        # ----------------------------
        def safe_val(series, default=0.0):
            return float(series.iloc[0]) if not series.empty and not pd.isna(series.iloc[0]) else float(default)

        # ----------------------------
        # 1. Select only ATM, ATM+1, ATM-1 strikes
        # ----------------------------
        strikes = sorted(merged_df["Strike"].dropna().unique())
        if atm_strike not in strikes:
            print(f"⚠️ [DEBUG] ATM strike {atm_strike} not found")
            return "SIDEWAYS"

        atm_idx = strikes.index(atm_strike)
        selected_strikes = []
        if atm_idx > 0:
            selected_strikes.append(strikes[atm_idx - 1])
        selected_strikes.append(strikes[atm_idx])
        if atm_idx < len(strikes) - 1:
            selected_strikes.append(strikes[atm_idx + 1])

        print(f"📊 [DEBUG] Selected strikes: {selected_strikes}")
        if not selected_strikes:
            return "SIDEWAYS"

        range_df = merged_df[merged_df["Strike"].isin(selected_strikes)].copy()

        # ----------------------------
        # 2. Update with Fyers quotes if available
        # ----------------------------
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
                        range_df.at[idx,"CE_atp"] = atp

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
                        range_df.at[idx, "PE_atp"] = atp

                range_df["CE_moneyflow"] = range_df["CE_atp"].astype(float) * range_df["CE_Vol"].astype(float)
                range_df["PE_moneyflow"] = range_df["PE_atp"].astype(float) * range_df["PE_Vol"].astype(float)


            except Exception as e:
                logging.warning(f"Fyers quotes error: {str(e)}")
                print(f"⚠️ [DEBUG] Fyers quotes error: {str(e)}")

        # ----------------------------
        # 3. Money flow & Direction
        # ----------------------------
        ce_net = range_df["CE_moneyflow"].sum()
        pe_net = range_df["PE_moneyflow"].sum()
        net_flow = (ce_net - pe_net) / (ce_net + pe_net) if (ce_net + pe_net) != 0 else 0.0

        ce_above_atp = range_df[range_df["CE_vs_atp"] == "ABOVE"].shape[0]
        pe_above_atp = range_df[range_df["PE_vs_atp"] == "ABOVE"].shape[0]
        direction_ratio = (ce_above_atp - pe_above_atp) / max(len(selected_strikes) * 2, 1)

        # ----------------------------
        # 4. Range placement logic + smoothing
        # ----------------------------
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

        # --- Step 4a: Per-bar average across strikes
        ce_per_bar = range_df["CE_range_placement"].mean()
        pe_per_bar = range_df["PE_range_placement"].mean()

        # --- Step 4b: Temporal smoothing (rolling avg across last N bars)
        MAX_SMOOTH_HISTORY = 3
        if session_state:
            if not hasattr(session_state, "ce_placement_buffer"):
                session_state.ce_placement_buffer = []
            if not hasattr(session_state, "pe_placement_buffer"):
                session_state.pe_placement_buffer = []

            session_state.ce_placement_buffer.append(ce_per_bar)
            session_state.pe_placement_buffer.append(pe_per_bar)

            session_state.ce_placement_buffer = session_state.ce_placement_buffer[-MAX_SMOOTH_HISTORY:]
            session_state.pe_placement_buffer = session_state.pe_placement_buffer[-MAX_SMOOTH_HISTORY:]

            ce_placement_avg = np.mean(session_state.ce_placement_buffer)
            pe_placement_avg = np.mean(session_state.pe_placement_buffer)

            # --- Debug printout for buffers
            print(f"📊 [DEBUG] CE buffer: {session_state.ce_placement_buffer} -> smoothed avg: {ce_placement_avg:.3f}")
            print(f"📊 [DEBUG] PE buffer: {session_state.pe_placement_buffer} -> smoothed avg: {pe_placement_avg:.3f}")
        else:
            ce_placement_avg = ce_per_bar
            pe_placement_avg = pe_per_bar


        # Relative placement_score
        # ----------------------------
        # 1. Compute placement_score
        # ----------------------------
        placement_score = ce_placement_avg - pe_placement_avg

        # ----------------------------
        # 2. Dynamic relative thresholds
        # ----------------------------
        RELATIVE_THRESHOLD = 2  # e.g., 2 times stronger
        dynamic_buy_threshold = pe_placement_avg * RELATIVE_THRESHOLD
        dynamic_sell_threshold = ce_placement_avg * RELATIVE_THRESHOLD

        # ----------------------------
        # 3. Raw signal generation
        # ----------------------------
        raw_signal = "SIDEWAYS"
        if  placement_score > dynamic_buy_threshold:
            raw_signal = "BUY"
        elif  placement_score < -dynamic_sell_threshold:
            raw_signal = "SELL"

        # ----------------------------
        # 7. Update rolling buffer and get majority
        # ----------------------------
        if session_state:
            if not hasattr(session_state, "signal_buffer"):
                session_state.signal_buffer = []

            session_state.signal_buffer.append(raw_signal)
            if len(session_state.signal_buffer) > MAX_HISTORY:
                session_state.signal_buffer.pop(0)

            print(f"📊 [DEBUG] Signal Buffer (smoothed): {session_state.signal_buffer}")
            final_signal = majority_signal(session_state.signal_buffer)
        else:
            print("Triggered a else in compute signal ")
            final_signal = raw_signal

        # ----------------------------
        # 8. Debug summary
        # ----------------------------
        print(f"\n📊 [DEBUG] Using ATM, ATM+1, ATM-1 strikes only")
        print(f"📊 [DEBUG] Selected strikes: {selected_strikes}")
        # print(f"📊 [DEBUG] ce_net: {ce_net}, pe_net: {pe_net}")
        # print(f"📊 [DEBUG] CE Above ATP: {ce_above_atp}, PE Above ATP: {pe_above_atp}")
        print(f"📊 [DEBUG] Net Flow: {net_flow:.2f}, Direction Ratio: {direction_ratio}")
        print(f"📊 [DEBUG] CE smoothed avg: {ce_placement_avg:.3f}, PE smoothed avg: {pe_placement_avg:.3f}")
        print(f"📊 [DEBUG] Placement Score: {placement_score} ,Dynamic Buy Threshold={dynamic_buy_threshold:.3f}, Dynamic Sell Threshold={dynamic_sell_threshold:.3f}")
        print(f"📊 [DEBUG] Raw Signal: {raw_signal}, Final Signal: {final_signal}")

        return final_signal

    except Exception as e:
        logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
        print(f"❌ [DEBUG] Error in compute_signals: {str(e)}")
        return "SIDEWAYS"


def select_strikes_atm_half_price(chain, atm_strike):
    """
    Strategy:
      - BUY ATM (both CE & PE)
      - SELL OTM where LTP ≈ 0.5 × ATM LTP
    """
    print(f"🔍 [DEBUG] Selecting strikes for ATM Buy / OTM Sell: {atm_strike}")
    try:
        df = pd.DataFrame(chain)
        if df.empty or atm_strike is None:
            print("⚠️ [DEBUG] No data or ATM strike for selection")
            return None, None, None, None

        ce = df[df["option_type"] == "CE"].copy()
        pe = df[df["option_type"] == "PE"].copy()

        # ---------------- ATM Buy legs ----------------
        atm_pe_row = pe[pe["strike_price"] == atm_strike]
        atm_ce_row = ce[ce["strike_price"] == atm_strike]

        buy_pe_row = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
        buy_ce_row = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None

        # ---------------- OTM Sell legs (half price) ----------------
        sell_pe_row, sell_ce_row = None, None

        if buy_pe_row and "ltp" in buy_pe_row and buy_pe_row["ltp"] > 0:
            target = buy_pe_row["ltp"] / 4
            # Only consider OTM PEs (strike < ATM strike)
            pe_candidates = pe[pe["strike_price"] < atm_strike].copy()
            pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
            if not pe_candidates.empty:
                sell_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        if buy_ce_row and "ltp" in buy_ce_row and buy_ce_row["ltp"] > 0:
            target = buy_ce_row["ltp"] / 4
            # Only consider OTM CEs (strike > ATM strike)
            ce_candidates = ce[ce["strike_price"] > atm_strike].copy()
            ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target).abs()
            if not ce_candidates.empty:
                sell_ce_row = ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        print(f"✅ [DEBUG] Strike selection complete:")
        print(f"   - BUY PE: {buy_pe_row['symbol'] if buy_pe_row else 'None'}")
        print(f"   - SELL PE (OTM): {sell_pe_row['symbol'] if sell_pe_row else 'None'}")
        print(f"   - BUY CE: {buy_ce_row['symbol'] if buy_ce_row else 'None'}")
        print(f"   - SELL CE (OTM): {sell_ce_row['symbol'] if sell_ce_row else 'None'}")

        return buy_pe_row, sell_pe_row, buy_ce_row, sell_ce_row

    except Exception as e:
        logging.error(f"Error in select_strikes_atm_half_price: {str(e)}", exc_info=True)
        print(f"❌ [DEBUG] Error in select_strikes_atm_half_price: {str(e)}")
        return None, None, None, None


# Order management with improved error handling
@handle_errors
def cancel_order(fyers, order_id):
    print(f"🔍 [DEBUG] Cancelling order: {order_id}")
    try:
        resp = fyers.cancel_order(data={"id": order_id})
        logging.info(f"Cancelled order {order_id}: {resp}")
        print(f"✅ [DEBUG] Order {order_id} cancelled: {resp}")
        return resp
    except Exception as e:
        logging.error(f"Exception while cancelling order {order_id}: {e}", exc_info=True)
        print(f"❌ [DEBUG] Error cancelling order {order_id}: {e}")
        raise


@handle_errors
def get_order_status_by_id(fyers, order_id):
    print(f"🔍 [DEBUG] Getting status for order: {order_id}")
    try:
        orderbook = fyers.orderbook()
        for o in orderbook.get("orderBook", []):
            if o.get("id") == order_id:
                status = o.get("status")
                print(f"✅ [DEBUG] Order {order_id} status: {status}")
                return status
        print(f"⚠️ [DEBUG] Order {order_id} not found in orderbook")
        return None
    except Exception as e:
        logging.warning(f"Could not fetch order status for {order_id}: {e}")
        print(f"❌ [DEBUG] Error getting order status for {order_id}: {e}")
        return None


@handle_errors
def place_order_and_check(fyers, order):
    print(f"🔍 [DEBUG] Placing order: {order}")
    try:
        resp = fyers.place_order(data=order)
        logging.info(f"Order placed for {order['symbol']}: {resp}")
        print(f"✅ [DEBUG] Order placed: {resp}")

        if resp.get("s") == "ok":
            oid = resp.get("id")
            time.sleep(2)  # Wait for order to process
            status = get_order_status_by_id(fyers, oid)
            if status in [1, 5]:  # Rejected or canceled
                logging.error(f"Order {oid} was rejected or canceled. Status: {status}")
                print(f"❌ [DEBUG] Order {oid} was rejected or canceled. Status: {status}")
                return None, status
            logging.info(f"Order {oid} status: {status}")
            print(f"✅ [DEBUG] Order {oid} status: {status}")
            return oid, status
        else:
            logging.error(f"Order failed: {resp}")
            print(f"❌ [DEBUG] Order failed: {resp}")
            return None, None
    except Exception as e:
        logging.exception(f"Place order error: {e}")
        print(f"❌ [DEBUG] Place order error: {e}")
        raise


# Trading logic with improved state management
# def has_open_orders_for_last_signal(fyers, session_state):
#     print(f"🔍 [DEBUG] Checking open orders for last signal")
#     ids = session_state.live_last_signal_order_ids
#     if not ids:
#         print("✅ [DEBUG] No order IDs for last signal")
#         return False
#     try:
#         ob = fyers.orderbook()
#         if ob and "orderBook" in ob:
#             has_open = any(o.get("id") in ids and o.get("status") == 2 for o in ob["orderBook"])
#             print(f"✅ [DEBUG] Open orders for last signal: {has_open}")
#             return has_open
#         print("⚠️ [DEBUG] No order book data")
#         return False
#     except Exception as e:
#         logging.warning(f"Could not check open orders: {e}")
#         print(f"❌ [DEBUG] Error checking open orders: {e}")
#         return False
def has_open_orders_for_last_signal(fyers, session_state):
    print(f"🔍 [DEBUG] Checking open orders for last signal")

    # First check if we have any actual positions
    try:
        pos_resp = fyers.positions()
        if pos_resp.get("s") == "ok":
            net_positions = [p for p in pos_resp.get("netPositions", [])
                             if abs(p.get("netQty", 0)) != 0]
            if not net_positions:
                print(f"✅ [DEBUG] No active positions found")
                session_state.last_signal_order_ids = []  # Clear stale order IDs
                return False
            else:
                print(f"📊 [DEBUG] Found {len(net_positions)} active positions")
                for p in net_positions:
                    print(f"  - {p.get('symbol')}: {p.get('netQty')} units")
                return True
    except Exception as e:
        print(f"❌ [DEBUG] Error checking positions: {e}")

    # Fallback to order book check
    ids = session_state.live_last_signal_order_ids
    if not ids:
        print("✅ [DEBUG] No order IDs for last signal")
        return False

    try:
        ob = fyers.orderbook()
        if ob and "orderBook" in ob:
            # Check for open orders (status 2 = open, 1 = completed, 5 = cancelled)
            open_orders = [o for o in ob["orderBook"]
                           if o.get("id") in ids and o.get("status") == 2]
            has_open = len(open_orders) > 0
            print(f"✅ [DEBUG] Open orders for last signal: {has_open} ({len(open_orders)} orders)")
            return has_open
        print("⚠️ [DEBUG] No order book data")
        return False
    except Exception as e:
        logging.warning(f"Could not check open orders: {e}")
        print(f"❌ [DEBUG] Error checking open orders: {e}")
        return False

def handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers):
    print(f"\n🔍 [DEBUG] Handling basket orders for signal: {signal}")

    # ------- simple lock to avoid overlapping runs -------
    if session_state._hb_lock:
        print("⚠️ [DEBUG] Another rebalance is already running. Skipping this cycle.")
        return []
    session_state._hb_lock = True

    placed_ids = []
    try:
        lot = Config.LOT_SIZE
        current_signal = session_state.last_signal
        print(f"🔄 [DEBUG] Current signal: {current_signal}, Incoming signal: {signal}")

        # ---------- helper: wait until positions are flat ----------
        def _wait_until_flat(timeout=28, interval=2):
            """Poll positions until no netQty remains, or timeout."""
            print(f"⏳ [DEBUG] Waiting up to {timeout}s for positions to flatten...")
            end = time.time() + timeout
            while time.time() < end:
                pos_resp = fyers.positions()
                print(f"📡 [DEBUG] Polled positions response: {pos_resp}")
                if pos_resp.get("s") == "ok":
                    still_open = [p for p in pos_resp.get("netPositions", [])
                                  if abs(p.get("netQty", 0)) != 0]
                    if not still_open:
                        print("✅ [DEBUG] All positions are flat now.")
                        return True
                    else:
                        print(f"⏳ [DEBUG] Still open positions: "
                              f"{[(p['symbol'], p['netQty']) for p in still_open]}")
                else:
                    print(f"❌ [DEBUG] Positions API error while waiting: {pos_resp}")
                time.sleep(interval)
            print("❌ [DEBUG] Timeout: positions did not flatten in time.")
            return False

        # ---------- current positions ----------
        pos_resp = fyers.positions()
        print(f"📡 [DEBUG] Initial positions API response: {pos_resp}")
        net_positions = []
        if pos_resp.get("s") == "ok":
            net_positions = [p for p in pos_resp.get("netPositions", [])
                             if abs(p.get("netQty", 0)) != 0]

        print(f"🔍 [DEBUG] Found {len(net_positions)} active positions at start:")
        for p in net_positions:
            print(f"  - {p.get('symbol')}: {p.get('netQty')} units")

        # ============================================================
        # 1) Handle HOLD signal (maintain existing positions)
        # ============================================================
        if signal == "HOLD":
            print("📊 [DEBUG] HOLD signal → Maintaining existing positions.")
            # Do nothing, just keep existing positions
            return []

        # ============================================================
        # 2) Handle SIDEWAYS signal (exit all positions)
        # ============================================================
        if signal == "SIDEWAYS":
            print("📊 [DEBUG] SIDEWAYS signal → Exiting all positions.")

            if not net_positions:
                print("✅ [DEBUG] No positions to exit for SIDEWAYS signal.")
                session_state.last_signal = signal
                return []

            try:
                # Exit all positions
                exit_shorts = {"segment": [11], "side": [-1], "productType": ["MARGIN"]}
                print(f"📤 [DEBUG] Sending SHORT exit request: {exit_shorts}")
                resp_short = fyers.exit_positions(data=exit_shorts)
                print(f"📥 [DEBUG] SHORT exit response: {resp_short}")

                exit_longs = {"segment": [11], "side": [1], "productType": ["MARGIN"]}
                print(f"📤 [DEBUG] Sending LONG exit request: {exit_longs}")
                resp_long = fyers.exit_positions(data=exit_longs)
                print(f"📥 [DEBUG] LONG exit response: {resp_long}")

                # Verify flattening
                if not _wait_until_flat(timeout=60, interval=2):
                    print("❌ [DEBUG] Flattening failed after exit. Manual check needed.")
                    return placed_ids

                print("✅ [DEBUG] All positions successfully exited for SIDEWAYS signal.")
                session_state.last_signal = signal
                return []

            except Exception as e:
                print(f"❌ [DEBUG] Exception during SIDEWAYS exit sequence: {str(e)}")
                return placed_ids

        # ============================================================
        # 3) FLATTEN on SIGNAL CHANGE (BUY/SELL to different BUY/SELL)
        # ============================================================
        if current_signal in ["BUY", "SELL"] and current_signal != signal and net_positions:
            print(f"⚠️ [DEBUG] Signal changed ({current_signal} → {signal}). "
                  f"Attempting to exit ALL open positions step by step...")

            try:
                # Step 1: Exit SHORT legs
                exit_shorts = {"segment": [11], "side": [-1], "productType": ["MARGIN"]}
                print(f"📤 [DEBUG] Sending SHORT exit request: {exit_shorts}")
                resp_short = fyers.exit_positions(data=exit_shorts)
                print(f"📥 [DEBUG] SHORT exit response: {resp_short}")

                # Step 2: Exit LONG legs
                exit_longs = {"segment": [11], "side": [1], "productType": ["MARGIN"]}
                print(f"📤 [DEBUG] Sending LONG exit request: {exit_longs}")
                resp_long = fyers.exit_positions(data=exit_longs)
                print(f"📥 [DEBUG] LONG exit response: {resp_long}")

                # Step 3: Verify flattening
                if not _wait_until_flat(timeout=60, interval=2):
                    print("❌ [DEBUG] Flattening failed after exit. Manual check needed.")
                    return placed_ids

                print("✅ [DEBUG] All positions successfully exited for signal change.")

            except Exception as e:
                print(f"❌ [DEBUG] Exception during exit sequence: {str(e)}")
                return placed_ids

        # ============================================================
        # 4) ENTER NEW POSITION for BUY/SELL signals (ATM only, no spread)
        # ============================================================
        pos_resp = fyers.positions()
        print(f"📡 [DEBUG] Positions before entry: {pos_resp}")
        current_positions = []
        if pos_resp.get("s") == "ok":
            current_positions = [p for p in pos_resp.get("netPositions", [])
                                 if abs(p.get("netQty", 0)) != 0]

        if current_positions:
            print("⚠️ [DEBUG] Some positions are STILL open → Skipping new entry.")
            for p in current_positions:
                print(f"  - OPEN: {p.get('symbol')} {p.get('netQty')}")
            return placed_ids

        entry_order = None

        if signal == "BUY":
            if buy_ce is not None:
                entry_order = {
                    "symbol": buy_ce["symbol"],
                    "qty": lot,
                    "type": 2,
                    "side": 1,  # BUY
                    "productType": "MARGIN",
                    "validity": "DAY",
                }
                print(f"📥 [DEBUG] Preparing new BUY ATM CE order: {entry_order}")
            else:
                print("⚠️ [DEBUG] Missing ATM CE for BUY signal.")
                return placed_ids

        elif signal == "SELL":
            if buy_pe is not None:
                entry_order = {
                    "symbol": buy_pe["symbol"],
                    "qty": lot,
                    "type": 2,
                    "side": 1,  # BUY
                    "productType": "MARGIN",
                    "validity": "DAY",
                }
                print(f"📥 [DEBUG] Preparing new SELL ATM PE order: {entry_order}")
            else:
                print("⚠️ [DEBUG] Missing ATM PE for SELL signal.")
                return placed_ids

        if entry_order:
            resp = fyers.place_order(data=entry_order)
            print(f"📥 [DEBUG] Order response: {resp}")
            if resp and resp.get("s") == "ok":
                placed_ids.append(resp.get("id"))
                session_state.last_signal = signal
                print(f"✅ [DEBUG] Successfully entered new {signal} order → ID: {resp.get('id')}")
            else:
                msg = resp.get("message", "Unknown error") if isinstance(resp, dict) else "Unknown error"
                print(f"❌ [DEBUG] Order FAILED. Reason: {msg}")

        return placed_ids

    except Exception as e:
        print(f"❌ [DEBUG] Fatal error in handle_basket_orders_atomic: {str(e)}")
        return []
    finally:
        session_state._hb_lock = False
        print("🔓 [DEBUG] Lock released.\n")

# Data display functions with improved formatting
def format_and_show(chain, title, ltp, show_signals=False, fyers_api=None, session_state=None):
    print(f"🔍 [DEBUG] Formatting and displaying data for: {title}")
    df = pd.DataFrame(chain)
    if df.empty:
        print(f"ℹ️ [DEBUG] No data for {title}")
        return None, None

    # Process CE and PE data
    ce = df[df["option_type"] == "CE"]
    pe = df[df["option_type"] == "PE"]

    # Define columns and rename them
    cols = ["symbol", "strike_price", "oi", "volume", "ltp", "ask", "bid",
            "ltpch", "ltpchp", "oich", "oichp", "prev_oi"]

    ce_df = ce[cols].rename(columns={
        "strike_price": "Strike",
        "oi": "OI",
        "volume": "Vol",
        "ltpch": "LTPCh",
        "ltpchp": "LTPChP",
        "oich": "OICh",
        "oichp": "OIChP",
        "prev_oi": "PrevOI"
    })

    pe_df = pe[cols].rename(columns={
        "strike_price": "Strike",
        "oi": "OI",
        "volume": "Vol",
        "ltpch": "LTPCh",
        "ltpchp": "LTPChP",
        "oich": "OICh",
        "oichp": "OIChP",
        "prev_oi": "PrevOI"
    })

    # Merge CE and PE data
    merged = pd.merge(
        ce_df.add_prefix("CE_"),
        pe_df.add_prefix("PE_"),
        left_on="CE_Strike",
        right_on="PE_Strike",
        how="outer"
    ).rename(columns={"CE_Strike": "Strike"}).drop("PE_Strike", axis=1)

    # Find ATM strike
    atm = None
    if ltp is not None and not merged["Strike"].dropna().empty:
        atm = min(merged["Strike"].dropna(), key=lambda x: abs(x - ltp))

    print(f"\n📈 {title}")

    if show_signals and atm is not None:
        # Calculate all required metrics
        strikes = sorted(merged["Strike"].dropna().unique().tolist())
        idx = strikes.index(atm)
        signal = compute_signals(merged, atm, ltp, fyers_api=fyers_api, session_state=session_state)
        # Display results with improved formatting
        print(f"✅ Signal: {signal}")

    # Display the dataframe
    print(f"📊 ATM Strike: {atm} | Underlying LTP: {ltp}")
    # print(merged.to_string(index=False))

    if show_signals and atm is not None:
        return signal, atm
    return None, atm


@handle_errors
def update_all_data(session_state, cid, token, sym):
    print(f"🔍 [DEBUG] Updating all data for {sym}")
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    enforce_rate_limit(session_state)
    session_state.option_chain_api_count += 1

    meta = fy.optionchain(data={"symbol": sym, "timestamp": ""})
    if meta.get("code") != 200:
        print(f"❌ [DEBUG] Option chain API error: {meta}")
        logging.error("Failed to fetch expiry data")
        print("❌ [DEBUG] Failed to fetch expiry data")
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

    print(f"✅ [DEBUG] Data update complete")
    return ltp, chain, curr_date


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_user_input(prompt, default=None, input_type=str):
    """Get user input with optional default value"""
    try:
        if default is not None:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if user_input == "":
                return default
            return input_type(user_input)
        else:
            return input_type(input(f"{prompt}: ").strip())
    except ValueError:
        print("❌ Invalid input. Please try again.")
        return get_user_input(prompt, default, input_type)


def settings_menu(session_state):
    """Display and handle settings menu"""
    while True:
        clear_screen()
        print("\n" + "=" * 50)
        print("⚙️ SETTINGS")
        print("=" * 50)
        print(f"1. Client ID: {session_state.cid}")
        print(f"2. Access Token: {'*' * len(session_state.token) if session_state.token else 'Not set'}")
        print(f"3. Lot Size: {Config.LOT_SIZE}")
        print("4. 🔙 Back to Main Menu")

        choice = get_user_input("Select option", input_type=int)

        if choice == 1:
            session_state.cid = get_user_input("Enter Client ID", session_state.cid)
        elif choice == 2:
            session_state.token = get_user_input("Enter Access Token", session_state.token)
        elif choice == 3:
            Config.LOT_SIZE = get_user_input("Enter Lot Size", Config.LOT_SIZE, int)
        elif choice == 4:
            break
        else:
            print("❌ Invalid option. Please try again.")


# Main application
def main():
    print("🚀 Starting Live Auto Trading Application")
    # Initialize session state
    session_state = SessionState()
    load_credentials(session_state)

    # Default symbol
    sym = 'NSE:NIFTY50-INDEX'

    while True:
        clear_screen()
        print("\n" + "=" * 100)
        print("📈 FYERS ALGO TRADING - LIVE AUTO TRADE")
        print("=" * 50)
        print("1. 📈 Start Live Trading")
        print("2. ⚙️ Settings")
        print("3. 🚪 Exit")
        print("=" * 50)

        choice = get_user_input("Select option", input_type=int)

        if choice == 1:  # Live Trading
            clear_screen()
            print("\n" + "=" * 100)
            print("📈 LIVE AUTO TRADING")
            print("=" * 100)

            # Check credentials
            if not (session_state.cid and session_state.token):
                print("❌ Please set Client ID and Access Token in Settings first.")
                input("Press Enter to continue...")
                continue

            # Get trading mode
            auto_trade = get_user_input("Enable Auto Trade? (y/n)", "y").lower() == 'y'

            if not auto_trade:
                print("ℹ️ Auto trade disabled. Returning to main menu.")
                input("Press Enter to continue...")
                continue

            print("🔄 Starting live trading loop...")

            try:
                while True:
                    print('=' * 50)
                    print(f"\n🔄 [DEBUG] Starting new iteration at {datetime.now().strftime('%H:%M:%S')}")

                    # Update data
                    ltp, chain, curr_date = update_all_data(session_state, session_state.cid, session_state.token, sym)

                    ist = pytz.timezone("Asia/Kolkata")
                    print(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")

                    if ltp:
                        print(f"✅ Underlying LTP: {ltp}")

                    if chain:
                        fy = fyersModel.FyersModel(client_id=session_state.cid, token=session_state.token,
                                                   is_async=False,
                                                   log_path="")
                        signal, atm = format_and_show(chain, f"Current Expiry: {curr_date}", ltp, show_signals=True,
                                                      fyers_api=fy, session_state=session_state)

                        # Record signal history
                        if signal and ltp:
                            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
                            session_state.signal_history.append({
                                "timestamp": timestamp,
                                "signal": signal,
                                "underlying_ltp": ltp,
                                "expiry_date": curr_date
                            })

                        # Select strikes
                        buy_pe, sell_pe, buy_ce, sell_ce = select_strikes_atm_half_price(chain, atm)

                        # Validate pairs
                        valid_buy = (sell_pe is not None and buy_pe is not None)
                        valid_sell = (sell_ce is not None and buy_ce is not None)

                        # Trading logic - handle all signal types including HOLD
                        if signal == "HOLD":
                            print("📊 HOLD signal → Maintaining existing positions.")
                            # No action needed, just continue holding

                        elif signal == "BUY" and not valid_buy:
                            print("⚠️ Missing option data for BUY signal. Unable to place trade.")
                        elif signal == "SELL" and not valid_sell:
                            print("⚠️ Missing option data for SELL signal. Unable to place trade.")
                        elif signal in ("BUY", "SELL"):
                            if session_state.live_last_signal == signal and has_open_orders_for_last_signal(fy,
                                                                                                            session_state):
                                print(f"ℹ️ Basket for {signal} already open.")
                            else:
                                ids = handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce,
                                                                  buy_ce, fy)
                                session_state.live_last_signal = signal
                                session_state.live_last_signal_order_ids = ids
                        elif signal == "SIDEWAYS":
                            handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
                        else:
                            print("ℹ️ No action.")

                    print(f"⏳ Waiting {Config.REFRESH_INTERVAL} seconds before next update...")
                    time.sleep(Config.REFRESH_INTERVAL)

            except KeyboardInterrupt:
                print("\n⏹️ Live trading stopped by user")
                input("Press Enter to continue...")

            except Exception as e:
                print(f"❌ Error in live trading: {str(e)}")
                input("Press Enter to continue...")

        elif choice == 2:  # Settings
            settings_menu(session_state)

        elif choice == 3:  # Exit
            print("👋 Exiting... Goodbye!")
            break

        else:
            print("❌ Invalid option. Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()