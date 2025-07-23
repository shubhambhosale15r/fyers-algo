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
# from dotenv import load_dotenv


#setting wide mode as default
st.set_page_config(
    page_title="Fyers Algo OI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"  # This is the key line
)
# Constants
class Config:
    REFRESH_INTERVAL = 60  # seconds
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
        "page": "trading"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Load credentials from environment variables
def load_credentials():
    # load_dotenv()
    st.session_state.cid = os.getenv("FYERS_CID", st.session_state.get("cid", ""))
    st.session_state.token = os.getenv("FYERS_TOKEN", st.session_state.get("token", ""))

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

# API calls with retry logic
@handle_errors
@lru_cache(maxsize=32)
def get_option_chain_data(cid, token, sym, expiry_ts=""):
    enforce_rate_limit()
    st.session_state["option_chain_api_count"] += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.optionchain(data={"symbol": sym, "timestamp": expiry_ts})
    if not resp or resp.get("code") != 200:
        error_msg = f"Option chain error ({expiry_ts or 'nearest'}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(error_msg)
        return None
    return resp["data"]["optionsChain"]

@handle_errors
def get_underlying_ltp(cid, token, sym):
    enforce_rate_limit()
    st.session_state["quote_api_count"] += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": sym})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error ({resp.get('code')}): {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(error_msg)
        return None
    return resp["d"][0]["v"]["lp"]

@handle_errors
def get_symbol_ltp(cid, token, symbol):
    enforce_rate_limit()
    st.session_state["quote_api_count"] += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": symbol})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        error_msg = f"Quotes API error for {symbol}: {resp.get('message', 'Unknown')}"
        logging.error(error_msg)
        st.error(error_msg)
        return None
    return resp["d"][0]["v"]["lp"]

# Signal computation with improved logging and safeguards
def compute_signals(merged_df, atm_strike, ltp):
    try:
        # Ensure required columns exist
        required_columns = [
            "Strike", "CE_ltp", "PE_ltp", "CE_OICh", "PE_OICh"
        ]
        for col in required_columns:
            if col not in merged_df.columns:
                logging.error(f"Missing required column: {col}")
                return "SIDEWAYS", "Error", "Error"

        # Prepare strikes list
        strikes = sorted(merged_df["Strike"].dropna().unique().tolist())
        if atm_strike not in strikes:
            logging.error(f"ATM Strike {atm_strike} not found in strikes: {strikes}")
            return "SIDEWAYS", "Error", "Error"

        idx = strikes.index(atm_strike)

        # Filter ATM row
        atm_row = merged_df[merged_df["Strike"] == atm_strike]
        atm_ce_price = atm_row["CE_ltp"].values[0] if not atm_row.empty else 0.0
        atm_pe_price = atm_row["PE_ltp"].values[0] if not atm_row.empty else 0.0

        # Identify ITM/OTM strikes
        ce_itm_strikes = strikes[max(0, idx - 3):idx] if idx >= 1 else []
        pe_itm_strikes = strikes[idx + 1:min(len(strikes), idx + 4)] if idx < len(strikes) - 1 else []

        ce_otm_strikes = strikes[idx + 1:min(len(strikes), idx + 4)] if idx < len(strikes) - 1 else []
        pe_otm_strikes = strikes[max(0, idx - 3):idx] if idx >= 1 else []

        # Helper to sum column values with .fillna(0)
        def sum_column(strikes_list, col_name):
            return merged_df[merged_df["Strike"].isin(strikes_list)][col_name].fillna(0).sum()

        # Prices
        itm_pe_price = sum_column(pe_itm_strikes, "PE_ltp")
        itm_ce_price = sum_column(ce_itm_strikes, "CE_ltp")
        otm_pe_price = sum_column(pe_otm_strikes, "PE_ltp")
        otm_ce_price = sum_column(ce_otm_strikes, "CE_ltp")

        # OI changes
        itm_pe_oi = sum_column(pe_itm_strikes, "PE_OICh")
        itm_ce_oi = sum_column(ce_itm_strikes, "CE_OICh")
        otm_pe_oi = sum_column(pe_otm_strikes, "PE_OICh")
        otm_ce_oi = sum_column(ce_otm_strikes, "CE_OICh")

        # Initial money flow
        ce_money_flow = (itm_ce_price * itm_ce_oi) + (otm_ce_price * otm_ce_oi)
        pe_money_flow = (itm_pe_price * itm_pe_oi) + (otm_pe_price * otm_pe_oi)

        # Add ATM contribution
        if not atm_row.empty:
            atm_ce_ltp = atm_row["CE_ltp"].values[0]
            atm_pe_ltp = atm_row["PE_ltp"].values[0]
            atm_ce_oi = atm_row["CE_OICh"].values[0]
            atm_pe_oi = atm_row["PE_OICh"].values[0]

            # Handle NaNs
            atm_ce_ltp = 0.0 if pd.isna(atm_ce_ltp) else atm_ce_ltp
            atm_pe_ltp = 0.0 if pd.isna(atm_pe_ltp) else atm_pe_ltp
            atm_ce_oi = 0.0 if pd.isna(atm_ce_oi) else atm_ce_oi
            atm_pe_oi = 0.0 if pd.isna(atm_pe_oi) else atm_pe_oi

            ce_money_flow += atm_ce_ltp * atm_ce_oi
            pe_money_flow += atm_pe_ltp * atm_pe_oi
        else:
            atm_ce_ltp = atm_pe_ltp = 0.0

        # Compute total prices
        total_ce_price = itm_ce_price + otm_ce_price + atm_ce_price
        total_pe_price = itm_pe_price + otm_pe_price + atm_pe_price

        # Default signal
        signal = "SIDEWAYS"

        # Signal logic
        # if total_ce_price > 1.10 * total_pe_price:
        if pe_money_flow > 1.10 * ce_money_flow or (ce_money_flow < 0 and pe_money_flow > 0):
                signal = "BUY"
        # elif total_pe_price > 1.10 * total_ce_price:
        elif ce_money_flow > 1.10 * pe_money_flow or (pe_money_flow < 0 and ce_money_flow > 0):
                signal = "SELL"

        # Logging for debug
        logging.info(f"Computed Signal: {signal} | CE Flow: {ce_money_flow:.2f} | PE Flow: {pe_money_flow:.2f} ")

        return signal, ce_money_flow, pe_money_flow

    except Exception as e:
        logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
        return "SIDEWAYS", "Error", "Error"
    
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
            target = sell_pe_row["ltp"] / 4
            pe_candidates = pe[pe["strike_price"] != atm_strike].copy()
            pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
            if not pe_candidates.empty:
                buy_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        buy_ce_row = None
        if sell_ce_row and "ltp" in sell_ce_row and sell_ce_row["ltp"] > 0:
            target = sell_ce_row["ltp"] / 4
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

@handle_errors
def place_order_and_check(fyers, order):
    try:
        resp = fyers.place_order(data=order)
        logging.info(f"Order placed for {order['symbol']}: {resp}")
        
        if resp.get("s") == "ok":
            oid = resp.get("id")
            time.sleep(2)  # Wait for order to process
            status = get_order_status_by_id(fyers, oid)
            if status in [1, 5]:  # Rejected or canceled
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

# Trading logic with improved state management
def has_open_orders_for_last_signal(fyers):
    ids = st.session_state.get("live_last_signal_order_ids", [])
    if not ids:
        return False
    try:
        ob = fyers.orderbook().get("orderBook", [])
        return any(o.get("id") in ids and o.get("status") == 2 for o in ob)
    except Exception as e:
        logging.warning(f"Could not check open orders: {e}")
        return False

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
    try:
        orders = []
        lot = Config.LOT_SIZE
        current_signal = st.session_state.get("last_signal")

        # Skip if signal is the same as last time
        if current_signal == signal:
            logging.info(f"Signal {signal} is already active. No new trades.")
            return []

        # Case 1: BUY → SELL (Exit BUY, Enter SELL)
        if current_signal == "BUY" and signal == "SELL":
            try:
                positions = fyers.positions().get("netPositions", [])
                for p in positions:
                    if p.get("segment") == "NFO" and p.get("qty") != 0:
                        side = -1 if p["qty"] > 0 else 1
                        orders.append({
                            "symbol": p["symbol"],
                            "qty": abs(int(p["qty"])),
                            "type": 2,
                            "side": side,
                            "productType": "MARGIN",
                            "validity": "DAY"
                        })
                logging.info("[LIVE] Exiting BUY positions for SELL signal")
            except Exception as e:
                logging.error(f"Failed to exit positions: {e}", exc_info=True)
                return []

            if buy_ce and sell_ce:
                orders.append({
                    "symbol": buy_ce["symbol"], 
                    "qty": lot, 
                    "type": 2, 
                    "side": 1,
                    "productType": "MARGIN",
                    "validity": "DAY"
                })
                orders.append({
                    "symbol": sell_ce["symbol"], 
                    "qty": lot, 
                    "type": 2, 
                    "side": -1,
                    "productType": "MARGIN",
                    "validity": "DAY"
                })

        # Case 2: SELL → BUY (Exit SELL, Enter BUY)
        elif current_signal == "SELL" and signal == "BUY":
            try:
                positions = fyers.positions().get("netPositions", [])
                for p in positions:
                    if p.get("segment") == "NFO" and p.get("qty") != 0:
                        side = -1 if p["qty"] > 0 else 1
                        orders.append({
                            "symbol": p["symbol"],
                            "qty": abs(int(p["qty"])),
                            "type": 2,
                            "side": side,
                            "productType": "MARGIN",
                            "validity": "DAY"
                        })
                logging.info("[LIVE] Exiting SELL positions for BUY signal")
            except Exception as e:
                logging.error(f"Failed to exit positions: {e}", exc_info=True)
                return []

            if buy_pe and sell_pe:
                orders.append({
                    "symbol": buy_pe["symbol"], 
                    "qty": lot, 
                    "type": 2, 
                    "side": 1,
                    "productType": "MARGIN",
                    "validity": "DAY"
                })
                orders.append({
                    "symbol": sell_pe["symbol"], 
                    "qty": lot, 
                    "type": 2, 
                    "side": -1,
                    "productType": "MARGIN",
                    "validity": "DAY"
                })

        # Case 3: New signal (no existing position)
        elif current_signal is None and signal in ["BUY", "SELL"]:
            if signal == "BUY" and buy_pe and sell_pe:
                orders.append({
                    "symbol": buy_pe["symbol"], 
                    "qty": lot, 
                    "type": 2, 
                    "side": 1,
                    "productType": "MARGIN",
                    "validity": "DAY"
                })
                orders.append({
                    "symbol": sell_pe["symbol"], 
                    "qty": lot, 
                    "type": 2, 
                    "side": -1,
                    "productType": "MARGIN",
                    "validity": "DAY"
                })
            elif signal == "SELL" and buy_ce and sell_ce:
                orders.append({
                    "symbol": buy_ce["symbol"], 
                    "qty": lot, 
                    "type": 2, 
                    "side": 1,
                    "productType": "MARGIN",
                    "validity": "DAY"
                })
                orders.append({
                    "symbol": sell_ce["symbol"], 
                    "qty": lot, 
                    "type": 2, 
                    "side": -1,
                    "productType": "MARGIN",
                    "validity": "DAY"
                })

        # Case 4: SIDEWAYS (do nothing)
        elif signal == "SIDEWAYS":
            logging.info("[LIVE] SIDEWAYS: Holding positions")
            return []

        # Execute orders
        placed_ids = []
        for order in orders:
            oid, status = place_order_and_check(fyers, order)
            if oid is None or status in [1, 5]:
                logging.error(f"Order failed: {order['symbol']}")
                for pid in placed_ids:
                    cancel_order(fyers, pid)
                return []
            placed_ids.append(oid)

        if placed_ids:
            st.session_state["last_signal"] = signal
            logging.info(f"[LIVE] {signal} basket executed!")
        
        return placed_ids

    except Exception as e:
        logging.error(f"Error in handle_basket_orders_atomic: {str(e)}", exc_info=True)
        return []

# Data display functions with improved formatting
def format_and_show(chain, title, ltp, show_signals=False):
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
    
    signal,ce_money_flow,pe_money_flow = compute_signals(merged, atm, ltp)
    
    st.write("Signal: ", signal)
    st.write("CE Money Flow: ", ce_money_flow)
    st.write("PE Money Flow: ", pe_money_flow)


    st.caption(f"ATM Strike: {atm} | Underlying LTP: {ltp}")
    
    if show_signals and atm is not None:
        return signal, atm
    return None, atm


@handle_errors
def update_all_data(cid, token, sym, paper_trade_enabled):
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    enforce_rate_limit()
    st.session_state["option_chain_api_count"] += 1

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
    # st.set_page_config(layout="wide")
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
        if paper_trade:
            if st.button("📊 Paper Trading Dashboard"):
                st.session_state.page = "paper_trading"

        # Credentials
        st.subheader("🔐 API Credentials")
        cid = st.text_input("Client ID", value=st.session_state.cid)
        token = st.text_input("Access Token", type="password", value=st.session_state.token)
        sym = st.selectbox('Choose Index: ', ['NSE:NIFTY50-INDEX', 'BSE:SENSEX-INDEX'])
        st.write(f'You selected: {sym}')

        # Store credentials
        st.session_state.cid = cid
        st.session_state.token = token

        # API usage
        st.subheader("📊 API Usage")
        oc = st.session_state.get("option_chain_api_count", 0)
        qc = st.session_state.get("quote_api_count", 0)
        st.write(f"Option Chain API: {oc}")
        st.write(f"Quote API: {qc}")
        st.write(f"Total: {oc + qc}/{Config.API_RATE_LIMIT} per minute")

    # Early return if no credentials
    if not (cid and token and sym):
        st.info("Enter credentials in the sidebar.")
        return

    # Update data
    ltp, chain, curr_date = update_all_data(cid, token, sym, paper_trade)

    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")

    if ltp:
        st.success(f"📊 Underlying LTP: {ltp}")

    # Page routing
    if st.session_state.page == "signal_history":
        show_signal_history()
        return

    if st.session_state.page == "paper_trading" and paper_trade:
        show_paper_trading_page(cid, token)
        return

    # Main trading page
    st.subheader("📈 Live Trading")

    if chain:
        signal, atm = format_and_show(chain, f"Current Expiry: {curr_date}", ltp, show_signals=True)

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
