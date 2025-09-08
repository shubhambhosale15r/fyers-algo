import streamlit as st
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


# Constants
class Config:
    REFRESH_INTERVAL = 30  # seconds
    API_RATE_LIMIT = 200
    API_WINDOW = 60  # seconds
    LOT_SIZE = 75
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
            # st.success("✅ Credentials loaded from JSON file")
    except FileNotFoundError:
        # Fall back to environment variables if file doesn't exist
        # st.warning("⚠️ JSON file not found, checking environment variables")
        session_state.cid = os.getenv("FYERS_CID", session_state.cid)
        session_state.token = os.getenv("FYERS_TOKEN", session_state.token)
    except Exception as e:
        st.error(f"❌ Error loading credentials: {str(e)}")
        # Fall back to environment variables
        session_state.cid = os.getenv("FYERS_CID", session_state.cid)
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
        st.warning(f"⚠️ Approaching rate limit, waiting {wait}s...")
        time.sleep(wait)
        session_state.api_window_start = datetime.now()
        session_state.option_chain_api_count = 0
        session_state.quote_api_count = 0


# API calls with retry logic
@handle_errors
@lru_cache(maxsize=32)
def get_option_chain_data(session_state, cid, token, sym, expiry_ts=""):
    st.write(f"🔍 Fetching option chain data for {sym}, expiry: {expiry_ts or 'nearest'}")
    enforce_rate_limit(session_state)
    session_state.option_chain_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.optionchain(data={"symbol": sym, "timestamp": expiry_ts})
    if not resp or resp.get("code") != 200:
        error_msg = f"Option chain error ({expiry_ts or 'nearest'}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(f"❌ {error_msg}")
        return None
    st.success("✅ Option chain data fetched successfully")
    return resp["data"]["optionsChain"]


@handle_errors
def get_underlying_ltp(session_state, cid, token, sym):
    st.write(f"🔍 Fetching underlying LTP for {sym}")
    enforce_rate_limit(session_state)
    session_state.quote_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": sym})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error ({resp.get('code')}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(f"❌ {error_msg}")
        return None
    ltp = resp["d"][0]["v"]["lp"]
    st.write(f"✅ Underlying LTP: {ltp}")
    return ltp


@handle_errors
def get_symbol_ltp(session_state, cid, token, symbol):
    st.write(f"🔍 Fetching LTP for {symbol}")
    enforce_rate_limit(session_state)
    session_state.quote_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": symbol})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error for {symbol}: {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(f"❌ {error_msg}")
        return None
    ltp = resp["d"][0]["v"]["lp"]
    st.write(f"✅ Symbol LTP: {ltp}")
    return ltp


# Signal computation with improved error handling
def compute_signals(merged_df, atm_strike, ltp, min_range_pct=1, tolerance_pct=1, fyers_api=None):
    """
    Compute intraday trading signals using ITM options only.
    Now using ATP (Average Traded Price) instead of VWAP for comparison.
    Returns: (signal, net_flow, direction_ratio)
    """
    st.write(f"🔍 Computing signals for ATM strike: {atm_strike}, LTP: {ltp}")
    try:
        def safe_val(series, default=0):
            """Safe extraction from series."""
            return series.iloc[0] if not series.empty and not pd.isna(series.iloc[0]) else default

        # 1. Determine price range around ATM
        atm_row = merged_df[merged_df["Strike"] == atm_strike]
        atm_ce_price = safe_val(atm_row["CE_ltp"])
        atm_pe_price = safe_val(atm_row["PE_ltp"])
        atm_ce_volume = safe_val(atm_row["CE_Vol"])
        atm_pe_volume = safe_val(atm_row["PE_Vol"])
        atm_ce_moneyflow = atm_ce_price * atm_ce_volume
        atm_pe_moneyflow = atm_pe_price * atm_pe_volume
        price_range = atm_ce_price + atm_pe_price

        min_range_abs = ltp * (min_range_pct / 100)

        if price_range >= min_range_abs:
            lower_bound = atm_strike - price_range
            upper_bound = atm_strike + price_range
            range_type = f"ATM+CE Range (±{price_range:.2f} points)"
        else:
            lower_bound = ltp * (1 - min_range_pct / 100)
            upper_bound = ltp * (1 + min_range_pct / 100)
            range_type = f"{min_range_pct}% Underlying Range"

        def round_to_nearest_strike(value, strikes):
            """Round to the nearest available strike price"""
            if not strikes:
                return value
            return min(strikes, key=lambda x: abs(x - value))

        strikes = sorted(merged_df["Strike"].dropna().unique())
        lower_bound = round_to_nearest_strike(lower_bound, strikes)
        upper_bound = round_to_nearest_strike(upper_bound, strikes)
        range_strikes = [s for s in strikes if lower_bound <= s <= upper_bound]

        if not range_strikes:
            logging.warning(f"No strikes in range {lower_bound:.2f}-{upper_bound:.2f}")
            st.warning(f"⚠️ No strikes in range {lower_bound:.2f}-{upper_bound:.2f}")
            return "SIDEWAYS"

        # 2. Fill missing values
        merged_df = merged_df.fillna(0)


        range_df = merged_df[merged_df["Strike"].isin(range_strikes)].copy()

        # 3. Fyers API quotes for ITM strikes - comparing LTP vs ATP
        if fyers_api and 'CE_symbol' in range_df.columns and 'PE_symbol' in range_df.columns:
            try:
                symbols = list(set(range_df['CE_symbol'].tolist() + range_df['PE_symbol'].tolist()))
                symbols = [s for s in symbols if s != 0 and isinstance(s, str)]
                quotes = {}
                for i in range(0, len(symbols), 50):
                    batch = symbols[i:i + 50]
                    response = fyers_api.quotes({"symbols": ",".join(batch)})
                    if response.get('code') == 200:
                        quotes.update({item['n']: item['v'] for item in response['d'] if item['s'] == 'ok'})

                # Initialize default values
                range_df['CE_vs_atp'] = 'BELOW'  # Default to BELOW if no data
                range_df['PE_vs_atp'] = 'BELOW'
                range_df['ce_strength'] = "LOW"
                range_df['pe_strength'] = "LOW"

                for idx, row in range_df.iterrows():
                    if row['CE_symbol'] in quotes:
                        ce_data = quotes[row['CE_symbol']]
                        ltp = ce_data.get('lp', 0)
                        atp = ce_data.get('atp', 0)
                        high_price = ce_data.get('high_price', 0)
                        if atp > 0:
                            range_df.at[idx, 'CE_vs_atp'] = 'ABOVE' if ltp > atp else 'BELOW'
                        if high_price > 0 and ltp > 0:
                            range_df.at[idx, 'CE_high'] = high_price
                            range_df.at[idx, 'CE_high_diff'] = high_price - ltp
                        range_df.at[idx, 'CE_ltp'] = ltp

                    if row['PE_symbol'] in quotes:
                        pe_data = quotes[row['PE_symbol']]
                        ltp = pe_data.get('lp', 0)
                        atp = pe_data.get('atp', 0)
                        high_price = pe_data.get('high_price', 0)
                        if atp > 0:
                            range_df.at[idx, 'PE_vs_atp'] = 'ABOVE' if ltp > atp else 'BELOW'
                        if high_price > 0 and ltp > 0:
                            range_df.at[idx, 'PE_high'] = high_price
                            range_df.at[idx, 'PE_high_diff'] = high_price - ltp
                        range_df.at[idx, 'PE_ltp'] = ltp
                merged_df["CE_moneyflow"] = merged_df["CE_ltp"] * merged_df["CE_Vol"]
                merged_df["PE_moneyflow"] = merged_df["PE_ltp"] * merged_df["PE_Vol"]

            except Exception as e:
                logging.warning(f"Fyers quotes error: {str(e)}")
                st.warning(f"⚠️ Fyers quotes error: {str(e)}")

        # 4. Select ITM strikes
        atm_idx = range_strikes.index(atm_strike) if atm_strike in range_strikes else len(range_strikes) // 2
        num_strikes = min(len(range_strikes[:atm_idx]), len(range_strikes[atm_idx + 1:]))
        itm_ce_strikes = range_strikes[atm_idx - num_strikes:atm_idx]
        itm_pe_strikes = range_strikes[atm_idx + 1:atm_idx + 1 + num_strikes]

        # 5. Calculate money flow and direction ratio (using ATP comparison)
        ce_net = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_moneyflow"].sum()
        pe_net = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_moneyflow"].sum()


        net_flow = (ce_net - pe_net) / (ce_net + pe_net) if (ce_net + pe_net) != 0 else 0

        ce_above_atp = range_df[range_df["Strike"].isin(itm_ce_strikes) & (range_df["CE_vs_atp"] == 'ABOVE')].shape[0]
        pe_above_atp = range_df[range_df["Strike"].isin(itm_pe_strikes) & (range_df["PE_vs_atp"] == 'ABOVE')].shape[0]

        direction_ratio = (ce_above_atp - pe_above_atp) / max(len(itm_ce_strikes) + len(itm_pe_strikes), 1)

        # 6. Convert absolute high diff → % diff from high
        range_df["CE_high_pct"] = np.where(
            range_df["CE_high"] > 0,
            range_df["CE_high_diff"] / range_df["CE_high"],
            0
        )
        range_df["PE_high_pct"] = np.where(
            range_df["PE_high"] > 0,
            range_df["PE_high_diff"] / range_df["PE_high"],
            0
        )

        ce_high_pct_avg = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_high_pct"].mean()
        pe_high_pct_avg = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_high_pct"].mean()

        # Strength metric: -1 (PE stronger) → +1 (CE stronger)
        strength = (ce_high_pct_avg - pe_high_pct_avg) / max(ce_high_pct_avg + pe_high_pct_avg, 1e-6)

        # 7. Generate signal
        signal = "SIDEWAYS"
        if direction_ratio >= 0.5 and strength < -0.2:      #0.2 mean 50% more    0.3 mean 100% more
            st.write("📈 Market is in BUY entry found entering the trade")
            signal = "BUY"
        elif direction_ratio <= -0.5 and strength > 0.2:
            st.write("📉 Market is in SELL entry found entering the trade")
            signal = "SELL"

        # Print signal details
        st.write(f"\n📊 {range_type}")
        st.write(f"📊 Range: {lower_bound:.2f} to {upper_bound:.2f}")
        st.write(f"📊 ITM CE Strikes: {list(itm_ce_strikes)}")
        st.write(f"📊 ITM PE Strikes: {list(itm_pe_strikes)}")
        st.write(f"📊 Net Money Flow: {net_flow:.2f}")
        st.write(
            f"📊 Direction Ratio: {direction_ratio:.2f} | CE Above ATP: {ce_above_atp} | PE Above ATP: {pe_above_atp} | Total Strikes: {len(itm_ce_strikes) + len(itm_pe_strikes)}")
        st.write(
            f"📊 Avg CE % below High: {ce_high_pct_avg:.2%} | Avg PE % below High: {pe_high_pct_avg:.2%} | Strength: {strength:.2f}")

        return signal

    except Exception as e:
        logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
        st.error(f"❌ Error in compute_signals: {str(e)}")
        return "SIDEWAYS"


# Optimized strike selection
def select_strikes_atm_half_price(chain, atm_strike):
    st.write(f"🔍 Selecting strikes for ATM: {atm_strike}")
    try:
        df = pd.DataFrame(chain)
        if df.empty or atm_strike is None:
            st.warning("⚠️ No data or ATM strike for selection")
            return None, None, None, None

        ce = df[df["option_type"] == "CE"].copy()
        pe = df[df["option_type"] == "PE"].copy()

        # Find ATM PE and CE rows
        atm_pe_row = pe[pe["strike_price"] == atm_strike]
        atm_ce_row = ce[ce["strike_price"] == atm_strike]

        sell_pe_row = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
        sell_ce_row = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None

        # Find half-price strikes with vectorized operations
        buy_pe_row = None
        if sell_pe_row and "ltp" in sell_pe_row and sell_pe_row["ltp"] > 0:
            target = sell_pe_row["ltp"] / 2
            pe_candidates = pe[pe["strike_price"] != atm_strike].copy()
            pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
            if not pe_candidates.empty:
                buy_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        buy_ce_row = None
        if sell_ce_row and "ltp" in sell_ce_row and sell_ce_row["ltp"] > 0:
            target = sell_ce_row["ltp"] / 2
            ce_candidates = ce[ce["strike_price"] != atm_strike].copy()
            ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target).abs()
            if not ce_candidates.empty:
                buy_ce_row = ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        st.write(f"✅ Strike selection complete:")
        st.write(f"   - SELL PE: {sell_pe_row['symbol'] if sell_pe_row else 'None'}")
        st.write(f"   - BUY PE: {buy_pe_row['symbol'] if buy_pe_row else 'None'}")
        st.write(f"   - SELL CE: {sell_ce_row['symbol'] if sell_ce_row else 'None'}")
        st.write(f"   - BUY CE: {buy_ce_row['symbol'] if buy_ce_row else 'None'}")

        return sell_pe_row, buy_pe_row, sell_ce_row, buy_ce_row

    except Exception as e:
        logging.error(f"Error in select_strikes_atm_half_price: {str(e)}", exc_info=True)
        st.error(f"❌ Error in select_strikes_atm_half_price: {str(e)}")
        return None, None, None, None


# Order management with improved error handling
@handle_errors
def cancel_order(fyers, order_id):
    st.write(f"🔍 Cancelling order: {order_id}")
    try:
        resp = fyers.cancel_order(data={"id": order_id})
        logging.info(f"Cancelled order {order_id}: {resp}")
        st.success(f"✅ Order {order_id} cancelled: {resp}")
        return resp
    except Exception as e:
        logging.error(f"Exception while cancelling order {order_id}: {e}", exc_info=True)
        st.error(f"❌ Error cancelling order {order_id}: {e}")
        raise


@handle_errors
def get_order_status_by_id(fyers, order_id):
    st.write(f"🔍 Getting status for order: {order_id}")
    try:
        orderbook = fyers.orderbook()
        for o in orderbook.get("orderBook", []):
            if o.get("id") == order_id:
                status = o.get("status")
                st.write(f"✅ Order {order_id} status: {status}")
                return status
        st.warning(f"⚠️ Order {order_id} not found in orderbook")
        return None
    except Exception as e:
        logging.warning(f"Could not fetch order status for {order_id}: {e}")
        st.error(f"❌ Error getting order status for {order_id}: {e}")
        return None


@handle_errors
def place_order_and_check(fyers, order):
    st.write(f"🔍 Placing order: {order}")
    try:
        resp = fyers.place_order(data=order)
        logging.info(f"Order placed for {order['symbol']}: {resp}")
        st.success(f"✅ Order placed: {resp}")

        if resp.get("s") == "ok":
            oid = resp.get("id")
            time.sleep(2)  # Wait for order to process
            status = get_order_status_by_id(fyers, oid)
            if status in [1, 5]:  # Rejected or canceled
                logging.error(f"Order {oid} was rejected or canceled. Status: {status}")
                st.error(f"❌ Order {oid} was rejected or canceled. Status: {status}")
                return None, status
            logging.info(f"Order {oid} status: {status}")
            st.write(f"✅ Order {oid} status: {status}")
            return oid, status
        else:
            logging.error(f"Order failed: {resp}")
            st.error(f"❌ Order failed: {resp}")
            return None, None
    except Exception as e:
        logging.exception(f"Place order error: {e}")
        st.error(f"❌ Place order error: {e}")
        raise


# Trading logic with improved state management
def has_open_orders_for_last_signal(fyers, session_state):
    st.write(f"🔍 Checking open orders for last signal")
    ids = session_state.live_last_signal_order_ids
    if not ids:
        st.write("✅ No order IDs for last signal")
        return False
    try:
        ob = fyers.orderbook().get("orderBook", [])
        has_open = any(o.get("id") in ids and o.get("status") == 2 for o in ob)
        st.write(f"✅ Open orders for last signal: {has_open}")
        return has_open
    except Exception as e:
        logging.warning(f"Could not check open orders: {e}")
        st.error(f"❌ Error checking open orders: {e}")
        return False


# Live trading with improved order handling
def handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers):
    """
    Atomic-ish rebalancing using positions as the ground truth:
    - If SIDEWAYS: do nothing.
    - If signal changed vs last_signal:
        1) Exit SHORTS first, then LONGS (via exit_positions with segment/side/productType)
        2) Wait until positions API shows FLAT (or timeout)
        3) Enter new basket for `signal`
    - If same signal: avoid duplicate entries (re-check positions)
    Notes:
      - No orderbook polling; positions are the source of truth.
      - Includes a simple re-entrancy lock in session_state to avoid overlap.
      - Debug logs added for every step to aid troubleshooting.
    """
    st.write(f"\n🔍 Handling basket orders for signal: {signal}")

    # ------- simple lock to avoid overlapping runs -------
    if session_state._hb_lock:
        st.warning("⚠️ Another rebalance is already running. Skipping this cycle.")
        return []
    session_state._hb_lock = True

    placed_ids = []
    try:
        lot = Config.LOT_SIZE
        current_signal = session_state.last_signal
        st.write(f"🔄 Current signal: {current_signal}, Incoming signal: {signal}")

        if signal not in ["BUY", "SELL"]:
            st.write("📊 Signal = SIDEWAYS → No trading action.")
            return []

        # Map entry legs for the new signal
        signal_map = {
            "BUY": {"enter_hedge": buy_pe, "enter_short": sell_pe},
            "SELL": {"enter_hedge": buy_ce, "enter_short": sell_ce}
        }

        # ---------- helper: wait until positions are flat ----------
        def _wait_until_flat(timeout=28, interval=2):
            """Poll positions until no netQty remains, or timeout."""
            st.write(f"⏳ Waiting up to {timeout}s for positions to flatten...")
            end = time.time() + timeout
            while time.time() < end:
                pos_resp = fyers.positions()
                st.write(f"📡 Polled positions response: {pos_resp}")
                if pos_resp.get("s") == "ok":
                    still_open = [p for p in pos_resp.get("netPositions", [])
                                  if abs(p.get("netQty", 0)) != 0]
                    if not still_open:
                        st.success("✅ All positions are flat now.")
                        return True
                    else:
                        st.write(f"⏳ Still open positions: "
                              f"{[(p['symbol'], p['netQty']) for p in still_open]}")
                else:
                    st.error(f"❌ Positions API error while waiting: {pos_resp}")
                time.sleep(interval)
            st.error("❌ Timeout: positions did not flatten in time.")
            return False

        # ---------- current positions ----------
        pos_resp = fyers.positions()
        st.write(f"📡 Initial positions API response: {pos_resp}")
        net_positions = []
        if pos_resp.get("s") == "ok":
            net_positions = [p for p in pos_resp.get("netPositions", [])
                             if abs(p.get("netQty", 0)) != 0]

        st.write(f"🔍 Found {len(net_positions)} active positions at start:")
        for p in net_positions:
            st.write(f"  - {p.get('symbol')}: {p.get('netQty')} units")

        # ============================================================
        # 1) FLATTEN EVERYTHING on SIGNAL CHANGE
        # ============================================================
        # ============================================================
        # EXIT LOGIC: SHORTS FIRST, THEN LONGS/HEGES
        # ============================================================
        if current_signal in ["BUY", "SELL"] and current_signal != signal and net_positions:
            st.warning(f"⚠️ Signal changed ({current_signal} → {signal}). "
                f"Attempting to exit ALL open positions step by step...")

            try:
                # ---------- Step 1: Exit SHORT legs ----------
                exit_shorts = {
                    "segment": [11],          # Equity Derivatives
                    "side": [-1],             # Shorts only
                    "productType": ["ALL"]    # Try ALL first
                }
                st.write(f"📤 Sending SHORT exit request: {exit_shorts}")
                resp_short = fyers.exit_positions(data=exit_shorts)
                st.write(f"📥 SHORT exit response: {resp_short}")

                if resp_short.get("code") != 200:
                    # Fallback using specific productTypes
                    product_types = list(set([p.get('productType', 'MARGIN') for p in net_positions if p.get("netQty",0) < 0]))
                    fallback_short = {"segment": [11], "side": [-1], "productType": product_types}
                    st.write(f"📤 SHORT exit fallback request: {fallback_short}")
                    resp_short = fyers.exit_positions(data=fallback_short)
                    st.write(f"📥 SHORT fallback response: {resp_short}")

                # ---------- Step 2: Exit LONG/HEDGE legs ----------
                exit_longs = {
                    "segment": [11],
                    "side": [1],              # Longs only
                    "productType": ["ALL"]
                }
                st.write(f"📤 Sending LONG exit request: {exit_longs}")
                resp_long = fyers.exit_positions(data=exit_longs)
                st.write(f"📥 LONG exit response: {resp_long}")

                if resp_long.get("code") != 200:
                    product_types = list(set([p.get('productType', 'MARGIN') for p in net_positions if p.get("netQty",0) > 0]))
                    fallback_long = {"segment": [11], "side": [1], "productType": product_types}
                    st.write(f"📤 LONG exit fallback request: {fallback_long}")
                    resp_long = fyers.exit_positions(data=fallback_long)
                    st.write(f"📥 LONG fallback response: {resp_long}")

                # ---------- Step 3: Verify flattening ----------
                if not _wait_until_flat(timeout=60, interval=2):
                    st.error("❌ Flattening failed after stepwise exit. Triggering EMERGENCY exit.")
                    # emergency_exit_positions(fyers, net_positions)
                    return placed_ids

                st.success("✅ All positions successfully exited step by step.")

            except Exception as e:
                st.error(f"❌ Exception during exit sequence: {str(e)}")
                # emergency_exit_positions(fyers, net_positions)
                return placed_ids
        # ============================================================
        # 2) ENTER NEW BASKET (only if nothing is open)
        # ============================================================
        pos_resp = fyers.positions()
        st.write(f"📡 Positions before entry: {pos_resp}")
        current_positions = []
        if pos_resp.get("s") == "ok":
            current_positions = [p for p in pos_resp.get("netPositions", [])
                                 if abs(p.get("netQty", 0)) != 0]

        if current_positions:
            st.warning("⚠️ Some positions are STILL open → Skipping new entry.")
            for p in current_positions:
                st.write(f"  - OPEN: {p.get('symbol')} {p.get('netQty')}")
            return placed_ids

        # Basket entry
        enter_hedge = signal_map[signal]["enter_hedge"]
        enter_short = signal_map[signal]["enter_short"]

        st.write(f"📥 Preparing new {signal} basket entry:")
        st.write(f"  - Hedge leg (BUY): {enter_hedge}")
        st.write(f"  - Short leg (SELL): {enter_short}")

        entry_basket = [
            {
                "symbol": enter_hedge["symbol"],
                "qty": lot,
                "type": 2,
                "side": 1,
                "productType": "MARGIN",
                "validity": "DAY",
            },
            {
                "symbol": enter_short["symbol"],
                "qty": lot,
                "type": 2,
                "side": -1,
                "productType": "MARGIN",
                "validity": "DAY",
            }
        ]

        st.write(f"📤 Sending basket order request: {entry_basket}")
        entry_resp = fyers.place_basket_orders(data=entry_basket)
        st.write(f"📥 Basket order response: {entry_resp}")

        if entry_resp and entry_resp.get("s") == "ok":
            for item in entry_resp.get("data", []):
                body = item.get("body", {})
                if body.get("s") == "ok" and body.get("id"):
                    placed_ids.append(body["id"])
            session_state.last_signal = signal
            st.success(f"✅ Successfully entered new {signal} basket orders → IDs: {placed_ids}")
        else:
            msg = entry_resp.get("message", "Unknown error") if isinstance(entry_resp, dict) else "Unknown error"
            st.error(f"❌ Basket order FAILED. Reason: {msg}")

        return placed_ids

    except Exception as e:
        st.error(f"❌ Fatal error in handle_basket_orders_atomic: {str(e)}")
        return []
    finally:
        session_state._hb_lock = False
        st.write("🔓 Lock released.\n")


# Data display functions with improved formatting
def format_and_show(chain, title, ltp, show_signals=False, fyers_api=None):
    st.write(f"🔍 Formatting and displaying data for: {title}")
    df = pd.DataFrame(chain)
    if df.empty:
        st.write(f"ℹ️ No data for {title}")
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

    st.subheader(f"📈 {title}")

    if show_signals and atm is not None:
        # Calculate all required metrics
        strikes = sorted(merged["Strike"].dropna().unique().tolist())
        idx = strikes.index(atm)
        signal = compute_signals(merged, atm, ltp, fyers_api=fyers_api)
        # Display results with improved formatting
        st.success(f"✅ Signal: {signal}")

    # Display the dataframe
    st.write(f"📊 ATM Strike: {atm} | Underlying LTP: {ltp}")
    st.dataframe(merged)

    if show_signals and atm is not None:
        return signal, atm
    return None, atm


@handle_errors
def update_all_data(session_state, cid, token, sym):
    st.write(f"🔍 Updating all data for {sym}")
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    enforce_rate_limit(session_state)
    session_state.option_chain_api_count += 1

    meta = fy.optionchain(data={"symbol": sym, "timestamp": ""})
    if meta.get("code") != 200:
        logging.error("Failed to fetch expiry data")
        st.error("❌ Failed to fetch expiry data")
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

    st.success("✅ Data update complete")
    return ltp, chain, curr_date


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Fyers Algo Trading",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'session_state' not in st.session_state:
        st.session_state.session_state = SessionState()
    
    session_state = st.session_state.session_state
    load_credentials(session_state)
    
    # Default symbol
    # sym = 'NSE:NIFTY50-INDEX'
    
    st.title("📈 FYERS ALGO TRADING - LIVE AUTO TRADE")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Credentials input
        session_state.cid = st.text_input("Client ID", value=session_state.cid, type="default")
        session_state.token = st.text_input("Access Token", value=session_state.token, type="password")
        sym = st.selectbox(
            "Select Symbol & Change Lot Size as per selected symbol",
            ("NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX"),
        )

        st.write("You selected:", sym)
        
        # Trading parameters
        Config.LOT_SIZE = st.number_input("Lot Size", min_value=1, value=Config.LOT_SIZE)
        Config.REFRESH_INTERVAL = st.number_input("Refresh Interval (seconds)", min_value=5, value=Config.REFRESH_INTERVAL)
        
        # Save credentials button
        if st.button("Save Credentials"):
            try:
                with open('fyers_credentials.json', 'w') as f:
                    json.dump({
                        'client_id': session_state.cid,
                        'access_token': session_state.token
                    }, f)
                st.success("✅ Credentials saved to file")
            except Exception as e:
                st.error(f"❌ Error saving credentials: {str(e)}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📈 Live Trading", "📊 Signal History", "ℹ️ About"])
    
    with tab1:
        st.header("Live Auto Trading")
        
        # Check credentials
        if not (session_state.cid and session_state.token):
            st.error("❌ Please set Client ID and Access Token in Settings first.")
            st.stop()
        
        # Trading controls
        col1, col2, col3 = st.columns(3)
        with col1:
            # auto_trade = st.checkbox("Enable Auto Trade", value=True)
            pass
        with col2:
            manual_refresh = st.button("🔄 Manual Refresh")
        with col3:
            stop_trading = st.button("⏹️ Stop Trading")
        
        if stop_trading:
            st.session_state.trading_active = False
            st.warning("Trading stopped. Click 'Start Trading' to resume.")
        
        # Initialize trading state
        if 'trading_active' not in st.session_state:
            st.session_state.trading_active = False
        
        if st.button("▶️ Start Trading") and not st.session_state.trading_active:
            st.session_state.trading_active = True
            st.success("Trading started!")
        
        # Trading status
        status_placeholder = st.empty()
        data_placeholder = st.empty()
        signal_placeholder = st.empty()
        
        # Trading loop
        if st.session_state.trading_active:
            status_placeholder.info("🔄 Trading in progress...")
            
            try:
                # Update data
                ltp, chain, curr_date = update_all_data(session_state, session_state.cid, session_state.token, sym)

                ist = pytz.timezone("Asia/Kolkata")
                current_time = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
                
                with data_placeholder.container():
                    st.write(f"🕒 Updated: {current_time}")
                    
                    if ltp:
                        st.metric("Underlying LTP", f"{ltp:.2f}")

                    if chain:
                        fy = fyersModel.FyersModel(client_id=session_state.cid, token=session_state.token, is_async=False,
                                                log_path="")
                        signal, atm = format_and_show(chain, f"Current Expiry: {curr_date}", ltp, show_signals=True,
                                                    fyers_api=fy)

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
                        sell_pe, buy_pe, sell_ce, buy_ce = select_strikes_atm_half_price(chain, atm)

                        # Validate pairs
                        valid_buy = (sell_pe is not None and buy_pe is not None)
                        valid_sell = (sell_ce is not None and buy_ce is not None)

                        # Trading logic
                        if signal == "BUY" and not valid_buy:
                            st.warning("⚠️ Missing option data for BUY signal. Unable to place trade.")
                        elif signal == "SELL" and not valid_sell:
                            st.warning("⚠️ Missing option data for SELL signal. Unable to place trade.")
                        elif signal in ("BUY", "SELL"):
                            if session_state.live_last_signal == signal and has_open_orders_for_last_signal(fy, session_state):
                                st.info(f"ℹ️ Basket for {signal} already open.")
                            else:
                                ids = handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
                                session_state.live_last_signal = signal
                                session_state.live_last_signal_order_ids = ids
                        elif signal == "SIDEWAYS":
                            handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
                        else:
                            st.info("ℹ️ No action.")
                
                # Wait for next update
                time.sleep(Config.REFRESH_INTERVAL)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error in live trading: {str(e)}")
                st.session_state.trading_active = False
                st.rerun()
        
        else:
            status_placeholder.warning("⏸️ Trading paused. Click 'Start Trading' to begin.")
    
    with tab2:
        st.header("Signal History")
        if session_state.signal_history:
            history_df = pd.DataFrame(session_state.signal_history)
            st.dataframe(history_df)
            
            # Signal statistics
            if not history_df.empty:
                st.subheader("Signal Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Signals", len(history_df))
                with col2:
                    buy_signals = len(history_df[history_df['signal'] == 'BUY'])
                    st.metric("BUY Signals", buy_signals)
                with col3:
                    sell_signals = len(history_df[history_df['signal'] == 'SELL'])
                    st.metric("SELL Signals", sell_signals)
        else:
            st.info("No signal history available yet.")
    
    with tab3:
        st.header("About")
        st.write("""
        This is a live auto trading application for Fyers API.
        
        **Features:**
        - Real-time option chain data analysis
        - Automated trading signals based on money flow analysis
        - Basket order execution
        - Risk management with position flattening
        
        **How to use:**
        1. Enter your Fyers Client ID and Access Token in the Settings sidebar
        2. Configure your trading parameters (lot size, refresh interval)
        3. Click "Start Trading" to begin automated trading
        4. Monitor signals and trades in real-time
        
        **Note:** Trading involves risk. Use this application at your own risk.
        """)


if __name__ == "__main__":
    main()
