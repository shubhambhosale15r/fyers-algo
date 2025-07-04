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

# Setting wide mode as default and handling Streamlit page config securely
if not st.get_option("server.headless"):
    st.set_page_config(
        page_title="Fyers Algo OI",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# --- Constants ---
class Config:
    REFRESH_INTERVAL = 60  # seconds
    API_RATE_LIMIT = 200
    API_WINDOW = 60  # seconds
    LOT_SIZE = 75
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

# --- Logging ---
LOG_FILE = "trading_app.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Error Handling Decorator ---
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

# --- Session State Initialization ---
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

# --- Credentials Loader ---
def load_credentials():
    st.session_state.cid = os.getenv("FYERS_CID", st.session_state.get("cid", ""))
    st.session_state.token = os.getenv("FYERS_TOKEN", st.session_state.get("token", ""))

# --- Rate Limiting ---
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

# --- API Calls With Retry ---
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

# --- Signal Computation ---
def compute_signals(merged_df, atm_strike, ltp):
    try:
        # Compute PCR for each strike
        merged_df["PCR"] = merged_df["PE_OI"] / merged_df["CE_OI"]
        # Sort the dataframe by Strike ascending
        merged_df_sorted = merged_df.sort_values("Strike")

        # Find the first strike where PCR < 0.5
        first_low = merged_df_sorted[merged_df_sorted["PCR"] < 0.5].head(1)
        # Find the first strike where PCR >= 1.5
        first_high = merged_df_sorted[merged_df_sorted["PCR"] >= 1.5].tail(1)

        # Get selected strike details for each
        low_strike = int(first_low["Strike"].values[0]) if not first_low.empty else None
        low_pcr = float(first_low["PCR"].values[0]) if not first_low.empty else None
        dist_low = abs(low_strike - atm_strike) if low_strike is not None else float('inf')

        high_strike = int(first_high["Strike"].values[0]) if not first_high.empty else None
        high_pcr = float(first_high["PCR"].values[0]) if not first_high.empty else None
        dist_high = abs(high_strike - atm_strike) if high_strike is not None else float('inf')

        # Determine signal
        if dist_low < dist_high:
            signal = "BEARISH"
        elif dist_high < dist_low:
            signal = "BULLISH"
        else:
            signal = "SIDEWAYS"

        # Return all the details
        return {
            "signal": signal,
            "atm_strike": atm_strike,
            "low_strike": low_strike,
            "low_pcr": low_pcr,
            "low_dist": dist_low,
            "high_strike": high_strike,
            "high_pcr": high_pcr,
            "high_dist": dist_high
        }
    except Exception as e:
        logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
        return {
            "signal": "SIDEWAYS",
            "atm_strike": atm_strike,
            "low_strike": None,
            "low_pcr": None,
            "low_dist": None,
            "high_strike": None,
            "high_pcr": None,
            "high_dist": None
        }

# --- Strike Selection ---
def select_strikes_atm_half_price(chain, atm_strike):
    try:
        df = pd.DataFrame(chain)
        if df.empty or atm_strike is None:
            return None, None, None, None
        ce = df[df["option_type"] == "CE"].copy()
        pe = df[df["option_type"] == "PE"].copy()
        atm_pe_row = pe[pe["strike_price"] == atm_strike]
        atm_ce_row = ce[ce["strike_price"] == atm_strike]
        sell_pe_row = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
        sell_ce_row = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None
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

# --- Order Management ---
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

# --- Trading Logic State ---
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

# --- Paper Trading ---
def handle_paper_trade(signal, sell_pe, buy_pe, sell_ce, buy_ce):
    try:
        orders = []
        lot = Config.LOT_SIZE
        timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
        current_signal = st.session_state.get("paper_last_signal")
        if current_signal == signal and st.session_state.paper_positions:
            logging.info(f"[PAPER] Signal {signal} is already active. No new trades.")
            return []
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
                    "symbol": buy_ce["symbol"], "qty": lot, "side": 1, "action": "BUY", "price": buy_ce["ltp"]
                })
                orders.append({
                    "symbol": sell_ce["symbol"], "qty": lot, "side": -1, "action": "SELL", "price": sell_ce["ltp"]
                })
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
                    "symbol": buy_pe["symbol"], "qty": lot, "side": 1, "action": "BUY", "price": buy_pe["ltp"]
                })
                orders.append({
                    "symbol": sell_pe["symbol"], "qty": lot, "side": -1, "action": "SELL", "price": sell_pe["ltp"]
                })
        elif current_signal is None and signal in ["BUY", "SELL"]:
            if signal == "BUY" and buy_pe and sell_pe:
                orders.append({
                    "symbol": buy_pe["symbol"], "qty": lot, "side": 1, "action": "BUY", "price": buy_pe["ltp"]
                })
                orders.append({
                    "symbol": sell_pe["symbol"], "qty": lot, "side": -1, "action": "SELL", "price": sell_pe["ltp"]
                })
            elif signal == "SELL" and buy_ce and sell_ce:
                orders.append({
                    "symbol": buy_ce["symbol"], "qty": lot, "side": 1, "action": "BUY", "price": buy_ce["ltp"]
                })
                orders.append({
                    "symbol": sell_ce["symbol"], "qty": lot, "side": -1, "action": "SELL", "price": sell_ce["ltp"]
                })
        elif signal == "SIDEWAYS":
            logging.info("[PAPER] SIDEWAYS: Holding existing positions")
            return []
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
                f"[PAPER] {order['action']} {order['qty']} of {order['symbol']} at {order['price']}"
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

# --- Live Trading ---
def handle_basket_orders_atomic(signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers):
    try:
        orders = []
        lot = Config.LOT_SIZE
        current_signal = st.session_state.get("last_signal")
        if current_signal == signal:
            logging.info(f"Signal {signal} is already active. No new trades.")
            return []
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
                    "symbol": buy_ce["symbol"], "qty": lot, "type": 2, "side": 1, "productType": "MARGIN", "validity": "DAY"
                })
                orders.append({
                    "symbol": sell_ce["symbol"], "qty": lot, "type": 2, "side": -1, "productType": "MARGIN", "validity": "DAY"
                })
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
                    "symbol": buy_pe["symbol"], "qty": lot, "type": 2, "side": 1, "productType": "MARGIN", "validity": "DAY"
                })
                orders.append({
                    "symbol": sell_pe["symbol"], "qty": lot, "type": 2, "side": -1, "productType": "MARGIN", "validity": "DAY"
                })
        elif current_signal is None and signal in ["BUY", "SELL"]:
            if signal == "BUY" and buy_pe and sell_pe:
                orders.append({
                    "symbol": buy_pe["symbol"], "qty": lot, "type": 2, "side": 1, "productType": "MARGIN", "validity": "DAY"
                })
                orders.append({
                    "symbol": sell_pe["symbol"], "qty": lot, "type": 2, "side": -1, "productType": "MARGIN", "validity": "DAY"
                })
            elif signal == "SELL" and buy_ce and sell_ce:
                orders.append({
                    "symbol": buy_ce["symbol"], "qty": lot, "type": 2, "side": 1, "productType": "MARGIN", "validity": "DAY"
                })
                orders.append({
                    "symbol": sell_ce["symbol"], "qty": lot, "type": 2, "side": -1, "productType": "MARGIN", "validity": "DAY"
                })
        elif signal == "SIDEWAYS":
            logging.info("[LIVE] SIDEWAYS: Holding positions")
            return []
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

# --- Data Display Functions ---
def format_and_show(chain, title, ltp, show_signals=False):
    df = pd.DataFrame(chain)
    if df.empty:
        st.info(f"No data for {title}")
        return None, None
    ce = df[df["option_type"] == "CE"]
    pe = df[df["option_type"] == "PE"]
    cols = ["symbol", "strike_price", "oi", "volume", "ltp", "ask", "bid", "ltpch", "ltpchp", "oich", "oichp", "prev_oi"]
    ce_df = ce[cols].rename(columns={
        "strike_price": "Strike", "oi": "OI", "volume": "Vol", "ltpch": "LTPCh", "ltpchp": "LTPChP", "oich": "OICh", "oichp": "OIChP", "prev_oi": "PrevOI"
    })
    pe_df = pe[cols].rename(columns={
        "strike_price": "Strike", "oi": "OI", "volume": "Vol", "ltpch": "LTPCh", "ltpchp": "LTPChP", "oich": "OICh", "oichp": "OIChP", "prev_oi": "PrevOI"
    })
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
    st.subheader(title)
    if show_signals and atm is not None:
        result = compute_signals(merged, atm, ltp)
        st.metric("Signal", result["signal"])
        st.write(f"ATM Strike: {result['atm_strike']}")
        st.write(f"Low PCR (<0.5): Strike={result['low_strike']}, PCR={result['low_pcr']}, Distance={result['low_dist']}")
        st.write(f"High PCR (>=1.5): Strike={result['high_strike']}, PCR={result['high_pcr']}, Distance={result['high_dist']}")

    styled = merged.style.apply(
        lambda row: ["background: yellow" if row["Strike"] == atm else "" for _ in row],
        axis=1
    )
    st.dataframe(styled)
    st.caption(f"ATM Strike: {atm} | Underlying LTP: {ltp}")
    if show_signals and atm is not None:
        return result, atm
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

# --- UI Pages ---
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

# --- Main Application ---
def main():
    init_session_state()
    st_autorefresh(interval=Config.REFRESH_INTERVAL * 1000, key="global_refresh")
    st.title("Fyers Algo Trading")
    # --- Sidebar ---
    with st.sidebar:
        st.subheader("⚙️ Trading Settings")
        auto_trade = st.checkbox("Enable Auto Trade", value=False)
        paper_trade = st.checkbox("📝 Enable Paper Trading", value=False)
        st.subheader("📊 Navigation")
        if st.button("📈 Signal History"):
            st.session_state.page = "signal_history"
        if st.button("🔙 Back to Trading"):
            st.session_state.page = "trading"
        if paper_trade:
            if st.button("📊 Paper Trading Dashboard"):
                st.session_state.page = "paper_trading"
        st.subheader("🔐 API Credentials")
        cid = st.text_input("Client ID", value=st.session_state.cid)
        token = st.text_input("Access Token", type="password", value=st.session_state.token)
        sym = st.selectbox('Choose Index: ', ['NSE:NIFTY50-INDEX', 'BSE:SENSEX-INDEX'])
        st.write(f'You selected: {sym}')
        st.session_state.cid = cid
        st.session_state.token = token
        st.subheader("📊 API Usage")
        oc = st.session_state.get("option_chain_api_count", 0)
        qc = st.session_state.get("quote_api_count", 0)
        st.write(f"Option Chain API: {oc}")
        st.write(f"Quote API: {qc}")
        st.write(f"Total: {oc + qc}/{Config.API_RATE_LIMIT} per minute")

    if not (cid and token and sym):
        st.info("Enter credentials in the sidebar.")
        return

    ltp, chain, curr_date = update_all_data(cid, token, sym, paper_trade)
    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    if ltp:
        st.success(f"📊 Underlying LTP: {ltp}")

    if st.session_state.page == "signal_history":
        show_signal_history()
        return
    if st.session_state.page == "paper_trading" and paper_trade:
        show_paper_trading_page(cid, token)
        return

    st.subheader("📈 Live Trading")
    if chain:
        signal, atm = format_and_show(chain, f"Current Expiry: {curr_date}", ltp, show_signals=True)
        if signal and ltp:
            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.signal_history.append({
                "timestamp": timestamp,
                "signal": signal,
                "underlying_ltp": ltp,
                "expiry_date": curr_date
            })
        sell_pe, buy_pe, sell_ce, buy_ce = select_strikes_atm_half_price(chain, atm)
        valid_buy = (sell_pe is not None and buy_pe is not None)
        valid_sell = (sell_ce is not None and buy_ce is not None)
        fy = None
        if auto_trade:
            fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
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
