import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from fyers_apiv3 import fyersModel
import logging
import uuid
from streamlit_autorefresh import st_autorefresh
from functools import lru_cache
import os
import plotly.express as px
import subprocess

# Setting wide mode as default
st.set_page_config(
    page_title="Fyers Algo",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
class Config:
    REFRESH_INTERVAL = 10  # seconds
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
                    st.error(f"Operation failed after {Config.MAX_RETRIES} attempts: {str(e)}")
                    return None
    return wrapper

# Initialize all session state in one place
def init_session_state():
    defaults = {
        "option_chain_api_count": 0,
        "quote_api_count": 0,
        "api_window_start": datetime.now(),
        "last_signal": None,
        "last_signal_order_ids": [],
        "paper_trades": [],
        "paper_positions": {},
        "paper_pnl": {"realized": 0.0, "unrealized": 0.0},
        "trade_history": [],
        "paper_trade_active": False,
        "pnl_update_time": datetime.now(),
        "paper_last_signal": None,
        "paper_last_signal_order_ids": [],
        "live_last_signal": None,
        "live_last_signal_order_ids": [],
        "signal_history": [],
        "cid": "",
        "token": "",
        "page": "trading",
        "total_api_calls": 0,  # New: Total API calls since start
        "api_call_history": []  # New: History of API calls with timestamps
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Load credentials from environment variables
def load_credentials():
    st.session_state.cid = os.getenv("FYERS_CID", st.session_state.get("cid", ""))
    st.session_state.token = os.getenv("FYERS_TOKEN", st.session_state.get("token", ""))

# API call counter decorator
def count_api_call(api_type):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Count the API call
            st.session_state[f"{api_type}_api_count"] += 1
            st.session_state["total_api_calls"] += 1
            
            # Record the API call with timestamp
            st.session_state["api_call_history"].append({
                "timestamp": datetime.now(),
                "type": api_type,
                "function": func.__name__
            })
            
            # Enforce rate limiting
            enforce_rate_limit()
            
            # Call the original function
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Rate limiting with improved logic
def enforce_rate_limit():
    now = datetime.now()
    window = now - st.session_state["api_window_start"]
    total_calls = st.session_state["option_chain_api_count"] + st.session_state["quote_api_count"]
    
    if window > timedelta(seconds=Config.API_WINDOW):
        st.session_state["api_window_start"] = now
        st.session_state["option_chain_api_count"] = 0
        st.session_state["quote_api_count"] = 0
    elif total_calls >= Config.API_RATE_LIMIT:
        wait = Config.API_WINDOW - window.seconds
        logging.warning(f"Approaching rate limit, waiting {wait}s...")
        st.warning(f"Approaching rate limit, waiting {wait}s...")
        time.sleep(wait)
        st.session_state["api_window_start"] = datetime.now()
        st.session_state["option_chain_api_count"] = 0
        st.session_state["quote_api_count"] = 0

# API calls with retry logic and counting
@handle_errors
@count_api_call("option_chain")
@lru_cache(maxsize=32)
def get_option_chain_data(cid, token, sym, expiry_ts=""):
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.optionchain(data={"symbol": sym, "timestamp": expiry_ts})
    if not resp or resp.get("code") != 200:
        error_msg = f"Option chain error ({expiry_ts or 'nearest'}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(error_msg)
        return None
    return resp["data"]["optionsChain"]

@handle_errors
@count_api_call("quote")
def get_underlying_ltp(cid, token, sym):
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": sym})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error ({resp.get('code')}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(error_msg)
        return None
    return resp["d"][0]["v"]["lp"]

@handle_errors
@count_api_call("quote")
def get_symbol_ltp(cid, token, symbol):
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": symbol})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error for {symbol}: {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(error_msg)
        return None
    return resp["d"][0]["v"]["lp"]

# Signal computation with improved error handling
def compute_signals(merged_df, atm_strike, ltp, min_range_pct=1, tolerance_pct=1, fyers_api=None):
    """
    Compute intraday trading signals using ITM options only.
    Now using ATP (Average Traded Price) instead of VWAP for comparison.
    Returns: (signal, net_flow, direction_ratio)
    """
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
            lower_bound = ltp * (1 - min_range_pct/100)
            upper_bound = ltp * (1 + min_range_pct/100)
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
            return "SIDEWAYS", 0, 0

        # 2. Fill missing values
        merged_df = merged_df.fillna(0)
        merged_df["CE_moneyflow"] = merged_df["CE_ltp"] * merged_df["CE_Vol"]
        merged_df["PE_moneyflow"] = merged_df["PE_ltp"] * merged_df["PE_Vol"]

        range_df = merged_df[merged_df["Strike"].isin(range_strikes)].copy()

        # 3. Fyers API quotes for ITM strikes - comparing LTP vs ATP
        if fyers_api and 'CE_symbol' in range_df.columns and 'PE_symbol' in range_df.columns:
            try:
                symbols = list(set(range_df['CE_symbol'].tolist() + range_df['PE_symbol'].tolist()))
                symbols = [s for s in symbols if s != 0 and isinstance(s, str)]
                quotes = {}
                for i in range(0, len(symbols), 50):
                    batch = symbols[i:i+50]
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

            except Exception as e:
                logging.warning(f"Fyers quotes error: {str(e)}")
        # st.write(range_df)
        # 4. Select ITM strikes
        atm_idx = range_strikes.index(atm_strike) if atm_strike in range_strikes else len(range_strikes)//2
        itm_ce_strikes = range_strikes[:atm_idx]  # exclude ATM
        itm_pe_strikes = range_strikes[atm_idx+1:]    # exclude ATM

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
        if direction_ratio >= 0.5 and strength < -0.33:
            st.info("Market is in BUY entry found entering the trade")
            signal = "BUY"

        # elif  direction_ratio >= 0.5 and strength < -0.33:
        #     st.info("Market is in BUY looking for entries")

        elif  direction_ratio <= -0.5 and strength > 0.33:
            st.info("Market is in SELL entry found entering the trade")
            signal = "SELL"

        # elif direction_ratio <= -0.3 and strength > 0.33:
        #     st.info("Market is in SELL looking for entries")

        # if 0.33 < net_flow <= 0.66 and direction_ratio > 0 and strength < -0.33:
        #     st.info("Market is in BUY entry found entering the trade")
        #     signal = "BUY"

        # elif net_flow > 0.66 and direction_ratio > 0 and strength < -0.33:
        #     st.info("Market is in BUY looking for entries")

        # elif -0.66 <= net_flow < -0.33 and direction_ratio < 0 and strength > 0.33:
        #     st.info("Market is in SELL entry found entering the trade")
        #     signal = "SELL"

        # elif net_flow < -0.66 and direction_ratio < 0 and strength > 0.33:
        #     st.info("Market is in SELL looking for entries")

        st.markdown(f"""
        **{range_type}**  <br>
        **Range:** {lower_bound:.2f} to {upper_bound:.2f}  <br>
        **ITM CE Strikes:** {list(itm_ce_strikes)}  <br>
        **ITM PE Strikes:** {list(itm_pe_strikes)} <br>
        **CE Money Flow:** {ce_net:.2f} <br> 
        **PE Money Flow:** {pe_net:.2f} <br>
        **Net Money Flow:** {net_flow:.2f}  <br>
        **Direction Ratio:** {direction_ratio:.2f}  | CE Above ATP: {ce_above_atp} | PE Above ATP: {pe_above_atp} | Total Strikes: {len(itm_ce_strikes) + len(itm_pe_strikes)} <br>  
        **Avg CE % below High:** {ce_high_pct_avg:.2%} | **Avg PE % below High:** {pe_high_pct_avg:.2%} | Strength: {strength:.2f} <br>
        """, unsafe_allow_html=True)
        return signal

    except Exception as e:
        logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
        return "SIDEWAYS"

# Optimized strike selection
def select_strikes_atm_half_price(chain, atm_strike):
    try:
        df = pd.DataFrame(chain)
        if df.empty or atm_strike is None:
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

        return sell_pe_row, buy_pe_row, sell_ce_row, buy_ce_row

    except Exception as e:
        logging.error(f"Error in select_strikes_atm_half_price: {str(e)}", exc_info=True)
        return None, None, None, None

# Order management with improved error handling
@handle_errors
def cancel_order(fyers, order_id):
    try:
        resp = fyers.cancel_order(data={"id": order_id})
        logging.info(f"Cancelled order {order_id}: {resp}")
        return resp
    except Exception as e:
        logging.error(f"Exception while cancelling order {order_id}: {e}", exc_info=True)
        raise

@handle_errors
def get_order_status_by_id(fyers, order_id):
    try:
        orderbook = fyers.orderbook()
        for o in orderbook.get("orderBook", []):
            if o.get("id") == order_id:
                return o.get("status")
        return None
    except Exception as e:
        logging.warning(f"Could not fetch order status for {order_id}: {e}")
        return None


##########################################################################
# Trading logic with improved state management
#functions added for live auto trade
def has_open_orders_for_last_signal(fyers):
    """Check if there are still open orders from the last signal"""
    ids = st.session_state.get("live_last_signal_order_ids", [])
    if not ids:
        return False
    
    try:
        orderbook_response = fyers.orderbook()
        order_book = orderbook_response.get("orderBook", []) if orderbook_response else []
        
        # Check for any open orders (status 4: Transit, 6: Pending, 7: Expired)
        open_orders = [o for o in order_book 
                      if o.get("id") in ids and o.get("status") in [4, 6, 7]]
        
        if open_orders:
            logging.info(f"Found {len(open_orders)} open orders from last signal")
            return True
        
        # Also check if orders were rejected or canceled
        failed_orders = [o for o in order_book 
                        if o.get("id") in ids and o.get("status") in [1, 5]]
        
        if failed_orders:
            logging.warning(f"Found {len(failed_orders)} failed orders from last signal")
            # Clear the failed order IDs from session state
            remaining_ids = [
                oid for oid in ids 
                if not any(o.get("id") == oid for o in failed_orders)
            ]
            st.session_state["live_last_signal_order_ids"] = remaining_ids
        
        return False
        
    except Exception as e:
        logging.warning(f"Could not check open orders: {e}")
        return False
    
def verify_positions_closed(fyers, original_positions):
    """Verify that all original positions have been closed"""
    try:
        time.sleep(1)
        positions_response = fyers.positions()
        current_positions = positions_response.get("netPositions", []) if positions_response else []
        
        # Create a map of original positions
        original_map = {}
        for p in original_positions:
            if (p.get("segment") in ["NFO", 11] and 
                p.get("qty", 0) != 0 and 
                abs(int(p.get("qty", 0))) > 0):
                original_map[p["symbol"]] = abs(int(p.get("qty", 0)))
        
        # Check if any original positions still exist
        for pos in current_positions:
            if (pos.get("segment") in ["NFO", 11] and 
                pos.get("symbol") in original_map and 
                abs(int(pos.get("qty", 0))) > 0):
                logging.warning(f"Position still open: {pos['symbol']} - Qty: {pos['qty']}")
                return False
        
        return True
    except Exception as e:
        logging.error(f"Error verifying positions: {e}")
        return False
    
def cancel_order(fyers, order_id):
    """Cancel an order with proper error handling"""
    try:
        response = fyers.cancel_order(order_id)
        if response and response.get("s") == "ok":
            logging.info(f"Successfully cancelled order: {order_id}")
            return True
        else:
            logging.error(f"Failed to cancel order {order_id}: {response}")
            return False
    except Exception as e:
        logging.error(f"Exception cancelling order {order_id}: {e}")
        return False
def place_order_and_check(fyers, order):
    """
    Place an order and check its status
    Returns: (order_id, status) or (None, None) if failed
    """
    try:
        response = fyers.place_order(order)
        if response and response.get("s") == "ok":
            order_id = response.get("id")
            logging.info(f"Order placed successfully: {order_id} for {order['symbol']}")
            
            # Check order status after a short delay
            time.sleep(0.5)
            orderbook_response = fyers.orderbook()
            if orderbook_response:
                orders = orderbook_response.get("orderBook", [])
                for o in orders:
                    if o.get("id") == order_id:
                        return order_id, o.get("status")
            
            return order_id, None
        else:
            logging.error(f"Order placement failed: {response} for {order['symbol']}")
            return None, None
    except Exception as e:
        logging.error(f"Exception placing order for {order['symbol']}: {e}")
        return None, None
##################################################

# Paper trading with improved position tracking
def handle_paper_trade(signal, sell_pe, buy_pe, sell_ce, buy_ce):
    try:
        orders = []
        lot = Config.LOT_SIZE
        timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
        current_signal = st.session_state.get("paper_last_signal")

        # Skip if signal is the same as last time
        if current_signal == signal and st.session_state.paper_positions:
            logging.info(f"[PAPER] Signal {signal} is already active. No new trades.")
            return []

        # Case 1: BUY → SELL (Exit BUY, Enter SELL)
        if current_signal == "BUY" and signal == "SELL":
            for symbol, position in list(st.session_state.paper_positions.items()):
                close_action = "SELL" if position["action"] == "BUY" else "BUY"
                orders.append({
                    "symbol": symbol,
                    "qty": position["qty"],
                    "action": close_action,
                    "price": position["current_price"],
                    "type": "CLOSE"
                })
            logging.info("[PAPER] Exiting BUY positions for SELL signal")

            if buy_ce and sell_ce:
                orders.append({
                    "symbol": buy_ce["symbol"], 
                    "qty": lot, 
                    "side": 1, 
                    "action": "BUY", 
                    "price": buy_ce["ltp"]
                })
                orders.append({
                    "symbol": sell_ce["symbol"], 
                    "qty": lot, 
                    "side": -1, 
                    "action": "SELL", 
                    "price": sell_ce["ltp"]
                })

        # Case 2: SELL → BUY (Exit SELL, Enter BUY)
        elif current_signal == "SELL" and signal == "BUY":
            for symbol, position in list(st.session_state.paper_positions.items()):
                close_action = "SELL" if position["action"] == "BUY" else "BUY"
                orders.append({
                    "symbol": symbol,
                    "qty": position["qty"],
                    "action": close_action,
                    "price": position["current_price"],
                    "type": "CLOSE"
                })
            logging.info("[PAPER] Exiting SELL positions for BUY signal")

            if buy_pe and sell_pe:
                orders.append({
                    "symbol": buy_pe["symbol"], 
                    "qty": lot, 
                    "side": 1, 
                    "action": "BUY", 
                    "price": buy_pe["ltp"]
                })
                orders.append({
                    "symbol": sell_pe["symbol"], 
                    "qty": lot, 
                    "side": -1, 
                    "action": "SELL", 
                    "price": sell_pe["ltp"]
                })

        # Case 3: New signal (no existing position)
        elif current_signal is None and signal in ["BUY", "SELL"]:
            if signal == "BUY" and buy_pe and sell_pe:
                orders.append({
                    "symbol": buy_pe["symbol"], 
                    "qty": lot, 
                    "side": 1, 
                    "action": "BUY", 
                    "price": buy_pe["ltp"]
                })
                orders.append({
                    "symbol": sell_pe["symbol"], 
                    "qty": lot, 
                    "side": -1, 
                    "action": "SELL", 
                    "price": sell_pe["ltp"]
                })
            elif signal == "SELL" and buy_ce and sell_ce:
                orders.append({
                    "symbol": buy_ce["symbol"], 
                    "qty": lot, 
                    "side": 1, 
                    "action": "BUY", 
                    "price": buy_ce["ltp"]
                })
                orders.append({
                    "symbol": sell_ce["symbol"], 
                    "qty": lot, 
                    "side": -1, 
                    "action": "SELL", 
                    "price": sell_ce["ltp"]
                })

        # Case 4: SIDEWAYS (do nothing)
        elif signal == "SIDEWAYS":
            logging.info("[PAPER] SIDEWAYS: Holding existing positions")
            return []

        # Execute orders
        placed_ids = []
        for order in orders:
            trade_id = str(uuid.uuid4())[:8]
            trade = {
                "id": trade_id,
                "timestamp": timestamp,
                "symbol": order["symbol"],
                "qty": order["qty"],
                "action": order["action"],
                "price": order["price"],
                "signal": signal,
                "type": order.get("type", "OPEN")
            }
            update_paper_positions(trade)
            st.session_state.trade_history.append(trade)
            placed_ids.append(trade_id)
            logging.info(
                f"[PAPER] {order['action']} {order['qty']} of {order['symbol']} "
                f"at {order['price']}"
            )

        if placed_ids and signal in ["BUY", "SELL"]:
            st.session_state["paper_last_signal"] = signal

        return placed_ids

    except Exception as e:
        logging.error(f"Error in handle_paper_trade: {str(e)}", exc_info=True)
        return []

def update_paper_positions(trade):
    try:
        symbol = trade["symbol"]
        if trade["type"] == "OPEN":
            st.session_state.paper_positions[symbol] = {
                "symbol": trade["symbol"],
                "id": trade["id"],
                "entry_time": trade["timestamp"],
                "qty": trade["qty"],
                "action": trade["action"],
                "side": 1 if trade["action"] == "BUY" else -1,
                "entry_price": trade["price"],
                "current_price": trade["price"],
                "signal": trade["signal"]
            }
        else:
            if symbol in st.session_state.paper_positions:
                position = st.session_state.paper_positions[symbol]
                
                # Calculate PnL
                if position["action"] == "BUY":
                    pnl = (trade["price"] - position["entry_price"]) * position["qty"]
                else:
                    pnl = (position["entry_price"] - trade["price"]) * position["qty"]
                    
                st.session_state.paper_pnl["realized"] += pnl
                trade["pnl"] = pnl
                
                del st.session_state.paper_positions[symbol]
                
                if not st.session_state.paper_positions:
                    st.session_state["last_signal"] = None
    except Exception as e:
        logging.error(f"Error in update_paper_positions: {str(e)}", exc_info=True)

@handle_errors
def update_unrealized_pnl(cid, token):
    if not st.session_state.paper_positions:
        st.session_state.paper_pnl["unrealized"] = 0.0
        return
        
    total_unrealized = 0.0
    for symbol, position in st.session_state.paper_positions.items():
        ltp = get_symbol_ltp(cid, token, symbol)
        if ltp is None:
            ltp = position.get("current_price", position["entry_price"])
        
        position["current_price"] = ltp
        
        if position["action"] == "BUY":
            pnl = (ltp - position["entry_price"]) * position["qty"]
        else:
            pnl = (position["entry_price"] - ltp) * position["qty"]
            
        total_unrealized += pnl
    
    st.session_state.paper_pnl["unrealized"] = total_unrealized
    st.session_state.pnl_update_time = datetime.now()

@handle_errors
def update_position_prices(cid, token):
    if not st.session_state.paper_positions:
        return
        
    for symbol, position in st.session_state.paper_positions.items():
        ltp = get_symbol_ltp(cid, token, symbol)
        if ltp is not None:
            position["current_price"] = ltp

# Live trading with improved order handling

def handle_basket_orders_atomic(signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers):
    """
    Handle basket orders atomically - always close existing positions before opening new ones
    """
    try:
        current_signal = st.session_state.get("last_signal")
        lot = Config.LOT_SIZE
        placed_ids = []

        # Skip if signal is the same as last time and orders are still open
        if current_signal == signal and has_open_orders_for_last_signal(fyers):
            logging.info(f"Signal {signal} is already active with open orders. No new trades.")
            return []

        # Get current positions and open orders
        positions_response = fyers.positions()
        positions = positions_response.get("netPositions", []) if positions_response else []
        
        orderbook_response = fyers.orderbook()
        open_orders = orderbook_response.get("orderBook", []) if orderbook_response else []
        
        # Filter only active orders (status 4: Transit, 6: Pending, 7: Expired)
        active_orders = [o for o in open_orders if o.get("status") in [4, 6, 7]]
        
        # Case 1: BUY → SELL (Exit BUY first, then Enter SELL)
        if current_signal == "BUY" and signal == "SELL":
            try:
                # First, cancel any pending orders
                for order in active_orders:
                    if order.get("segment") in [11, "NFO"]:  # F&O segment
                        cancel_response = fyers.cancel_order(order["id"])
                        if cancel_response and cancel_response.get("s") == "ok":
                            logging.info(f"Cancelled pending order: {order['id']}")
                        time.sleep(1)  # Small delay between cancellations
                
                # Exit all existing positions
                # Separate shorts and longs
                short_positions = [p for p in positions if int(p.get("qty", 0)) < 0]
                long_positions = [p for p in positions if int(p.get("qty", 0)) > 0]

                # Exit shorts first, then longs
                ordered_positions = short_positions + long_positions
                exit_orders = []
                for p in ordered_positions:
                    if (p.get("segment") in ["NFO", 11] and
                        p.get("qty") != 0 and
                        abs(int(p.get("qty", 0))) > 0):

                        side = -1 if float(p.get("qty", 0)) > 0 else 1
                        exit_orders.append({
                            "symbol": p["symbol"],
                            "qty": abs(int(p["qty"])),
                            "type": 2,  # Market order
                            "side": side,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        })
                
                # Execute exit orders first
                if exit_orders:
                    logging.info(f"[LIVE] Exiting {len(exit_orders)} BUY positions for SELL signal")
                    for order in exit_orders:
                        oid, status = place_order_and_check(fyers, order)
                        if oid is None or status in [1, 5]:  # Canceled or Rejected
                            st.error(f"Exit order failed: {order['symbol']}")
                            # Cancel any successful exit orders if one fails
                            for pid in placed_ids:
                                cancel_order(fyers, pid)
                            return []
                        placed_ids.append(oid)
                
                # Wait for exits to process and verify
                time.sleep(2)
                if not verify_positions_closed(fyers, positions):
                    st.error("Failed to verify position closure")
                    for pid in placed_ids:
                        cancel_order(fyers, pid)
                    return []
                
                # Then enter new SELL positions
                if buy_ce and sell_ce:
                    entry_orders = [
                        {
                            "symbol": buy_ce["symbol"], 
                            "qty": lot, 
                            "type": 2, 
                            "side": 1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        },
                        {
                            "symbol": sell_ce["symbol"], 
                            "qty": lot, 
                            "type": 2, 
                            "side": -1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        }
                    ]
                    
                    for order in entry_orders:
                        oid, status = place_order_and_check(fyers, order)
                        if oid is None or status in [1, 5]:
                            st.error(f"Entry order failed: {order['symbol']}")
                            # Cancel any successful entry orders
                            for pid in placed_ids:
                                cancel_order(fyers, pid)
                            return []
                        placed_ids.append(oid)

            except Exception as e:
                st.error(f"Failed in BUY→SELL transition: {e}", exc_info=True)
                return []

        # Case 2: SELL → BUY (Exit SELL first, then Enter BUY)
        elif current_signal == "SELL" and signal == "BUY":
            try:
                # Cancel pending orders first
                for order in active_orders:
                    if order.get("segment") in [11, "NFO"]:
                        cancel_response = fyers.cancel_order(order["id"])
                        if cancel_response and cancel_response.get("s") == "ok":
                            st.info(f"Cancelled pending order: {order['id']}")
                        time.sleep(0.1)
                
                # Exit positions
                # Separate shorts and longs
                short_positions = [p for p in positions if int(p.get("qty", 0)) < 0]
                long_positions = [p for p in positions if int(p.get("qty", 0)) > 0]
                 # Exit shorts first, then longs
                ordered_positions = short_positions + long_positions
                exit_orders = []
                for p in ordered_positions:
                    if (p.get("segment") in ["NFO", 11] and 
                        p.get("qty") != 0 and 
                        abs(int(p.get("qty", 0))) > 0):
                        
                        side = -1 if float(p.get("qty", 0)) > 0 else 1
                        exit_orders.append({
                            "symbol": p["symbol"],
                            "qty": abs(int(p["qty"])),
                            "type": 2,
                            "side": side,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        })
                
                if exit_orders:
                    st.info(f"[LIVE] Exiting {len(exit_orders)} SELL positions for BUY signal")
                    for order in exit_orders:
                        oid, status = place_order_and_check(fyers, order)
                        if oid is None or status in [1, 5]:
                            st.error(f"Exit order failed: {order['symbol']}")
                            for pid in placed_ids:
                                cancel_order(fyers, pid)
                            return []
                        placed_ids.append(oid)
                
                time.sleep(2)
                if not verify_positions_closed(fyers, positions):
                    st.error("Failed to verify position closure")
                    for pid in placed_ids:
                        cancel_order(fyers, pid)
                    return []
                
                # Enter new positions
                if buy_pe and sell_pe:
                    entry_orders = [
                        {
                            "symbol": buy_pe["symbol"], 
                            "qty": lot, 
                            "type": 2, 
                            "side": 1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        },
                        {
                            "symbol": sell_pe["symbol"], 
                            "qty": lot, 
                            "type": 2, 
                            "side": -1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        }
                    ]
                    
                    for order in entry_orders:
                        oid, status = place_order_and_check(fyers, order)
                        if oid is None or status in [1, 5]:
                            st.error(f"Entry order failed: {order['symbol']}")
                            for pid in placed_ids:
                                cancel_order(fyers, pid)
                            return []
                        placed_ids.append(oid)

            except Exception as e:
                st.error(f"Failed in SELL→BUY transition: {e}", exc_info=True)
                return []

        # Case 3: New signal (no existing position)
        elif current_signal is None and signal in ["BUY", "SELL"]:
            try:
                # First cancel any pending orders (clean slate)
                for order in active_orders:
                    if order.get("segment") in [11, "NFO"]:
                        cancel_response = fyers.cancel_order(order["id"])
                        if cancel_response and cancel_response.get("s") == "ok":
                            logging.info(f"Cancelled pending order: {order['id']}")
                        time.sleep(0.1)
                
                # Check if there are any existing positions (shouldn't be, but safety check)
                if positions:
                    logging.warning(f"Found {len(positions)} existing positions for new signal. Closing first.")
                    # Separate shorts and longs
                    short_positions = [p for p in positions if int(p.get("qty", 0)) < 0]
                    long_positions = [p for p in positions if int(p.get("qty", 0)) > 0]
                    # Exit shorts first, then longs
                    ordered_positions = short_positions + long_positions
                    exit_orders = []
                    for p in ordered_positions:
                        if (p.get("segment") in ["NFO", 11] and 
                            p.get("qty") != 0 and 
                            abs(int(p.get("qty", 0))) > 0):
                            
                            side = -1 if float(p.get("qty", 0)) > 0 else 1
                            exit_orders.append({
                                "symbol": p["symbol"],
                                "qty": abs(int(p["qty"])),
                                "type": 2,
                                "side": side,
                                "productType": "MARGIN",
                                "validity": "DAY",
                                "offlineOrder": "False"
                            })
                    
                    if exit_orders:
                        logging.info(f"[LIVE] Closing {len(exit_orders)} unexpected positions for new signal")
                        for order in exit_orders:
                            oid, status = place_order_and_check(fyers, order)
                            if oid is None or status in [1, 5]:
                                st.error(f"Cleanup order failed: {order['symbol']}")
                                return []
                            placed_ids.append(oid)
                        
                        time.sleep(2)
                        if not verify_positions_closed(fyers, positions):
                            st.error("Failed to verify cleanup position closure")
                            return []
                
                # Now place new orders
                if signal == "BUY" and buy_pe and sell_pe:
                    entry_orders = [
                        {
                            "symbol": buy_pe["symbol"], 
                            "qty": lot, 
                            "type": 2, 
                            "side": 1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        },
                        {
                            "symbol": sell_pe["symbol"], 
                            "qty": lot, 
                            "type": 2, 
                            "side": -1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        }
                    ]
                elif signal == "SELL" and buy_ce and sell_ce:
                    entry_orders = [
                        {
                            "symbol": buy_ce["symbol"], 
                            "qty": lot, 
                            "type": 2, 
                            "side": 1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        },
                        {
                            "symbol": sell_ce["symbol"], 
                            "qty": lot, 
                            "type": 2, 
                            "side": -1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                            "offlineOrder": "False"
                        }
                    ]
                else:
                    st.error(f"Invalid options provided for {signal} signal")
                    return []
                
                for order in entry_orders:
                    oid, status = place_order_and_check(fyers, order)
                    if oid is None or status in [1, 5]:
                        logging.error(f"New position order failed: {order['symbol']}")
                        for pid in placed_ids:
                            cancel_order(fyers, pid)
                        return []
                    placed_ids.append(oid)
                    
            except Exception as e:
                st.warning(f"Failed in new signal placement: {e}", exc_info=True)
                return []

        # Case 4: SIDEWAYS (do nothing)
        elif signal == "SIDEWAYS":
            st.info("[LIVE] SIDEWAYS: Holding positions")
            return []

        # Case 5: Same signal but no open orders (re-entry)
        elif current_signal == signal and not has_open_orders_for_last_signal(fyers):
            logging.info(f"Signal {signal} is same but no open orders. Re-entering.")
            # This would follow similar logic to Case 3
            # For simplicity, we'll just log and return
            st.info("Re-entry logic not implemented. Holding current positions.")
            return []

        # Update session state with order IDs for tracking
        if placed_ids:
            st.session_state["last_signal"] = signal
            st.session_state["live_last_signal_order_ids"] = placed_ids
            st.info(f"[LIVE] {signal} basket executed successfully! Order IDs: {placed_ids}")
        
        return placed_ids

    except Exception as e:
        st.error(f"Error in handle_basket_orders_atomic: {str(e)}", exc_info=True)
        return []
    
# Data display functions with improved formatting
def format_and_show(chain, title, ltp, show_signals=False, fyers_api=None):
    df = pd.DataFrame(chain)
    if df.empty:
        st.info(f"No data for {title}")
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
    
    st.subheader(title)
    
    if show_signals and atm is not None:
        # Calculate all required metrics
        strikes = sorted(merged["Strike"].dropna().unique().tolist())
        idx = strikes.index(atm)
        signal = compute_signals(merged, atm, ltp, fyers_api=fyers_api)
        # Display results with improved formatting
        st.success(f"Signal: {signal}")

    # Highlight ATM row in the dataframe
    styled = merged.style.apply(
        lambda row: ["background: yellow" if row["Strike"] == atm else "" for _ in row],
        axis=1
    )
    
    st.dataframe(styled)
    st.caption(f"ATM Strike: {atm} | Underlying LTP: {ltp}")
    
    if show_signals and atm is not None:
        return signal, atm
    return None, atm

@handle_errors
def update_all_data(cid, token, sym, paper_trade_enabled):
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    
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

    ltp = get_underlying_ltp(cid, token, sym)
    chain = get_option_chain_data(cid, token, sym, expiry_ts=curr_ts)

    if paper_trade_enabled and st.session_state.get("paper_positions"):
        update_position_prices(cid, token)
        update_unrealized_pnl(cid, token)

    return ltp, chain, curr_date

def calculate_daily_metrics(trade_history):
    if not trade_history:
        return None
    
    df = pd.DataFrame(trade_history)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Calculate PnL if not present
    if 'pnl' not in df.columns:
        df['pnl'] = 0  # Default to 0 if PnL not calculated
    
    # Group by date
    daily = df.groupby('date').agg({
        'symbol': 'count',
        'pnl': ['sum', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()],
        'price': lambda x: abs(x).sum()  # For volume proxy
    })
    
    daily.columns = ['total_trades', 'net_pnl', 'winning_trades', 'losing_trades', 'volume']
    
    # Calculate additional metrics
    daily['win_rate'] = (daily['winning_trades'] / daily['total_trades']) * 100
    daily['loss_rate'] = (daily['losing_trades'] / daily['total_trades']) * 100
    
    # Calculate profit factor
    gross_profit = df[df['pnl'] > 0].groupby('date')['pnl'].sum()
    gross_loss = abs(df[df['pnl'] < 0].groupby('date')['pnl'].sum())
    daily['profit_factor'] = gross_profit / gross_loss.replace(0, 1)  # Avoid division by zero
    
    # Calculate expectancy
    avg_win = df[df['pnl'] > 0].groupby('date')['pnl'].mean()
    avg_loss = df[df['pnl'] < 0].groupby('date')['pnl'].mean()
    daily['expectancy'] = (daily['win_rate']/100 * avg_win) + (daily['loss_rate']/100 * avg_loss)
    
    # Calculate max win/loss per trade
    daily['max_win'] = df.groupby('date')['pnl'].max()
    daily['max_loss'] = df.groupby('date')['pnl'].min()
    
    # Calculate risk-reward ratio
    daily['risk_reward'] = abs(avg_win / avg_loss.replace(0, 1))  # Avoid division by zero
    
    return daily.reset_index()

def show_daily_analysis_page():
    st.subheader("📊 Daily Trade Analysis")
    
    if not st.session_state.trade_history:
        st.info("No trade history available yet")
        return
    
    # Calculate metrics
    daily_metrics = calculate_daily_metrics(st.session_state.trade_history)
    
    if daily_metrics is None:
        st.warning("Could not calculate daily metrics")
        return
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trading Days", len(daily_metrics))
    col2.metric("Average Daily PnL", f"₹{daily_metrics['net_pnl'].mean():.2f}")
    col3.metric("Average Win Rate", f"{daily_metrics['win_rate'].mean():.1f}%")
    
    # Show daily performance table
    st.subheader("Daily Performance Summary")
    
    # Format the dataframe for display
    display_df = daily_metrics.copy()
    display_df['net_pnl'] = display_df['net_pnl'].apply(lambda x: f"₹{x:.2f}")
    display_df['volume'] = display_df['volume'].apply(lambda x: f"₹{x:,.2f}")
    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%")
    display_df['loss_rate'] = display_df['loss_rate'].apply(lambda x: f"{x:.1f}%")
    display_df['profit_factor'] = display_df['profit_factor'].apply(lambda x: f"{x:.2f}")
    display_df['expectancy'] = display_df['expectancy'].apply(lambda x: f"₹{x:.2f}")
    display_df['max_win'] = display_df['max_win'].apply(lambda x: f"₹{x:.2f}")
    display_df['max_loss'] = display_df['max_loss'].apply(lambda x: f"₹{x:.2f}")
    display_df['risk_reward'] = display_df['risk_reward'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df)
    
    # Visualizations
    st.subheader("Performance Charts")
    
    # PnL over time
    fig1 = px.line(daily_metrics, x='date', y='net_pnl', 
                  title='Daily Net PnL Trend',
                  labels={'net_pnl': 'Net PnL (₹)', 'date': 'Date'})
    st.plotly_chart(fig1, use_container_width=True)
    
    # Win rate over time
    fig2 = px.line(daily_metrics, x='date', y='win_rate', 
                  title='Daily Win Rate Trend',
                  labels={'win_rate': 'Win Rate (%)', 'date': 'Date'})
    st.plotly_chart(fig2, use_container_width=True)
    
    # Profit factor over time
    fig3 = px.bar(daily_metrics, x='date', y='profit_factor', 
                 title='Daily Profit Factor',
                 labels={'profit_factor': 'Profit Factor', 'date': 'Date'})
    st.plotly_chart(fig3, use_container_width=True)
    
    # Download button
    csv = daily_metrics.to_csv(index=False)
    st.download_button(
        label="📥 Download Daily Metrics",
        data=csv,
        file_name="daily_trade_metrics.csv",
        mime="text/csv"
    )

# UI Pages
def show_signal_history():
    st.subheader("📜 Signal History")
    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not st.session_state.get("signal_history", []):
        st.info("No signal history available yet")
        return
    
    history_df = pd.DataFrame(st.session_state.signal_history)
    history_df = history_df.sort_values("timestamp", ascending=False)
    formatted_df = history_df.copy()
    formatted_df["underlying_ltp"] = formatted_df["underlying_ltp"].apply(
        lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x
    )

    st.dataframe(formatted_df)
    # Assuming formatted_df is your DataFrame
    df = formatted_df.copy()

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Remove currency symbol and convert underlying_ltp to float
    df['underlying_ltp'] = df['underlying_ltp'].replace('[₹,]', '', regex=True).astype(float)

    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['timestamp',  'underlying_ltp'])

    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download Signal History",
        data=csv,
        file_name="signal_history.csv",
        mime="text/csv"
    )

def show_paper_trading_page(cid, token):
    st.subheader("📝 Paper Trading Dashboard")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Realized PnL", f"₹{st.session_state.paper_pnl['realized']:.2f}")
    col2.metric("📊 Unrealized PnL", f"₹{st.session_state.paper_pnl['unrealized']:.2f}")
    total_pnl = st.session_state.paper_pnl['realized'] + st.session_state.paper_pnl['unrealized']
    col3.metric("💵 Total PnL", f"₹{total_pnl:.2f}", delta=f"{total_pnl:.2f}")
    
    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.subheader("📊 Current Positions")
    if st.session_state.paper_positions:
        positions_list = []
        for symbol, position in st.session_state.paper_positions.items():
            if position["action"] == "BUY":
                pnl = (position["current_price"] - position["entry_price"]) * position["qty"]
            else:
                pnl = (position["entry_price"] - position["current_price"]) * position["qty"]
                
            positions_list.append({
                "Symbol": position["symbol"],
                "Action": position["action"],
                "Qty": position["qty"],
                "Entry Price": f"₹{position['entry_price']:.2f}",
                "Current Price": f"₹{position['current_price']:.2f}",
                "PnL": f"₹{pnl:.2f}",
                "Signal": position["signal"]
            })
        
        positions_df = pd.DataFrame(positions_list)
        st.dataframe(positions_df)
    else:
        st.info("No open positions")
    
    st.subheader("📜 Trade History")
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        
        if not history_df.empty:
            desired_columns = ["timestamp", "symbol", "action", "qty", "price", "signal", "type", "pnl"]
            available_columns = [col for col in desired_columns if col in history_df.columns]
            history_df = history_df[available_columns]
            
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
            history_df["timestamp"] = history_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            history_df = history_df.sort_values("timestamp", ascending=False)
            
            if "price" in history_df:
                history_df["price"] = history_df["price"].apply(
                    lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x
                )
            if "pnl" in history_df:
                history_df["pnl"] = history_df["pnl"].apply(
                    lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x
                )
        
        st.dataframe(history_df)
        
        if st.button("💾 Download Trade History"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="paper_trade_history.csv",
                mime="text/csv"
            )
    else:
        st.info("No trade history yet")
    
    if st.button("🔄 Reset Paper Trading"):
        st.session_state.paper_trades = []
        st.session_state.paper_positions = {}
        st.session_state.paper_pnl = {"realized": 0.0, "unrealized": 0.0}
        st.session_state.trade_history = []
        st.session_state["last_signal"] = None
        st.success("Paper trading data reset!")

# Main application
def main():
    # Initialize everything
    init_session_state()
    load_credentials()
    
    # Global refresh
    st_autorefresh(interval=Config.REFRESH_INTERVAL * 1000, key="global_refresh")
    st.title("Fyers Algo Trading")
    
    # Settings sidebar
    with st.sidebar:
        st.subheader("⚙️ Trading Settings")
        auto_trade = st.checkbox("Enable Auto Trade", value=False)
        paper_trade = st.checkbox("📝 Enable Paper Trading", value=False)

        # Navigation
        st.subheader("📊 Navigation")
        if st.button("📈 Signal History"):
            st.session_state.page = "signal_history"
        if st.button("🔙 Back to Trading"):
            st.session_state.page = "trading"
        if st.button("📊 Daily Trade Analysis"):
            st.session_state.page = "daily_analysis"
        if paper_trade:
            if st.button("📊 Paper Trading Dashboard"):
                st.session_state.page = "paper_trading"

        # Credentials
        st.subheader("🔐 API Credentials")
        cid = st.text_input("Client ID", value=st.session_state.cid)
        token = st.text_input("Access Token", type="password", value=st.session_state.token)
        sym = st.selectbox('Choose Index: ', ['NSE:NIFTY50-INDEX', 'BSE:SENSEX-INDEX'])
        st.write(f'You selected: {sym}')
        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=None,
            value=Config.LOT_SIZE,  # Uses the default 75 as initial value
            step=1
        )
        # Update the config with user input
        Config.LOT_SIZE = quantity  # Now becomes whatever user entered

        # Store credentials
        st.session_state.cid = cid
        st.session_state.token = token

        # API usage
        st.subheader("📊 API Usage")
        oc = st.session_state.get("option_chain_api_count", 0)
        qc = st.session_state.get("quote_api_count", 0)
        total = st.session_state.get("total_api_calls", 0)
        
        st.write(f"Current Window (last 60s):")
        st.write(f"- Option Chain API: {oc}")
        st.write(f"- Quote API: {qc}")
        st.write(f"- Total: {oc + qc}/{Config.API_RATE_LIMIT}")
        
        st.write(f"Total API Calls Since Start: {total}")
        
        # Show API call history if requested
        if st.checkbox("Show API Call History"):
            api_history = st.session_state.get("api_call_history", [])
            if api_history:
                api_df = pd.DataFrame(api_history)
                api_df["timestamp"] = pd.to_datetime(api_df["timestamp"])
                api_df["timestamp"] = api_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(api_df.sort_values("timestamp", ascending=False))
            else:
                st.info("No API call history yet")

    # Early return if no credentials
    if not (cid and token and sym):
        st.info("Enter credentials in the sidebar.")
        return

    # Page routing
    if st.session_state.page == "signal_history":
        show_signal_history()
        return

    if st.session_state.page == "daily_analysis":
        show_daily_analysis_page()
        return

    if st.session_state.page == "paper_trading" and paper_trade:
        show_paper_trading_page(cid, token)
        return

    # Main trading page
    st.subheader("📈 Live Trading")

    # Update data
    ltp, chain, curr_date = update_all_data(cid, token, sym, paper_trade)

    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")

    if ltp:
        st.success(f"📊 Underlying LTP: {ltp}")

    if chain:
        fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
        signal, atm = format_and_show(chain, f"Current Expiry: {curr_date}", ltp, show_signals=True, fyers_api=fy)

        # Record signal history
        if signal and ltp:
            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.signal_history.append({
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

        # Initialize Fyers for auto trading
        fy = None
        if auto_trade:
            fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")

        # Trading logic
        if signal == "BUY" and not valid_buy:
            st.warning("Missing option data for BUY signal. Unable to place trade.")
        elif signal == "SELL" and not valid_sell:
            st.warning("Missing option data for SELL signal. Unable to place trade.")
        elif signal in ("BUY", "SELL"):
            if paper_trade:
                logging.info(f"📢 Signal detected: {signal} - Attempting paper trade")
                ids = handle_paper_trade(signal, sell_pe, buy_pe, sell_ce, buy_ce)
                st.session_state["paper_last_signal_order_ids"] = ids
            elif auto_trade:
                if st.session_state.get("live_last_signal") == signal and has_open_orders_for_last_signal(fy):
                    st.info(f"Basket for {signal} already open.")
                else:
                    ids = handle_basket_orders_atomic(signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
                    st.session_state["live_last_signal"] = signal
                    st.session_state["live_last_signal_order_ids"] = ids
            else:
                st.info(f"Auto trade disabled; signal: {signal}.")
        elif signal == "SIDEWAYS":
            if paper_trade:
                handle_paper_trade(signal, sell_pe, buy_pe, sell_ce, buy_ce)
            elif auto_trade:
                handle_basket_orders_atomic(signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
            else:
                st.info("Auto trade disabled; sideways.")
        else:
            st.info("No action.")

if __name__ == "__main__":
    main()
