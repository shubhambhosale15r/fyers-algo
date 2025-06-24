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

#setting wide mode as default
st.set_page_config(
    page_title="Fyers Algo",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"  # This is the key line
)

# — SESSION STATE FOR API COUNTERS & RATE LIMIT —
if "option_chain_api_count" not in st.session_state:
    st.session_state["option_chain_api_count"] = 0
if "quote_api_count" not in st.session_state:
    st.session_state["quote_api_count"] = 0
if "api_window_start" not in st.session_state:
    st.session_state["api_window_start"] = datetime.now()


# For duplicate order prevention
if "last_signal" not in st.session_state:
    st.session_state["last_signal"] = None
if "last_signal_order_ids" not in st.session_state:
    st.session_state["last_signal_order_ids"] = []

# Paper trading state
if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []
if "paper_positions" not in st.session_state:
    st.session_state.paper_positions = {}
if "paper_pnl" not in st.session_state:
    st.session_state.paper_pnl = {"realized": 0.0, "unrealized": 0.0}
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "paper_trade_active" not in st.session_state:
    st.session_state.paper_trade_active = False
if "pnl_update_time" not in st.session_state:
    st.session_state.pnl_update_time = datetime.now()

# for signal tracking separation!
if "paper_last_signal" not in st.session_state:
    st.session_state["paper_last_signal"] = None
if "paper_last_signal_order_ids" not in st.session_state:
    st.session_state["paper_last_signal_order_ids"] = []
if "live_last_signal" not in st.session_state:
    st.session_state["live_last_signal"] = None
if "live_last_signal_order_ids" not in st.session_state:
    st.session_state["live_last_signal_order_ids"] = []

def enforce_rate_limit():
    now = datetime.now()
    window = now - st.session_state["api_window_start"]
    total_calls = st.session_state["option_chain_api_count"] + st.session_state["quote_api_count"]
    if window > timedelta(seconds=60):
        st.session_state["api_window_start"] = now
        st.session_state["option_chain_api_count"] = 0
        st.session_state["quote_api_count"] = 0
    elif total_calls >= 200:
        wait = 60 - window.seconds
        st.warning(f"Approaching rate limit, waiting {wait}s...")
        time.sleep(wait)
        st.session_state["api_window_start"] = datetime.now()
        st.session_state["option_chain_api_count"] = 0
        st.session_state["quote_api_count"] = 0


def get_option_chain_data(cid, token, sym, expiry_ts=""):
    enforce_rate_limit()
    st.session_state["option_chain_api_count"] += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.optionchain(data={"symbol": sym, "timestamp": expiry_ts})
    if not resp or resp.get("code") != 200:
        st.error(f"Option chain error ({expiry_ts or 'nearest'}): {resp.get('message', 'Unknown')}")
        return None
    return resp["data"]["optionsChain"]


def get_underlying_ltp(cid, token, sym):
    enforce_rate_limit()
    st.session_state["quote_api_count"] += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": sym})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        st.error(f"Quotes API error ({resp.get('code')}): {resp.get('message', 'Unknown')}")
        return None
    return resp["d"][0]["v"]["lp"]


def get_symbol_ltp(cid, token, symbol):
    enforce_rate_limit()
    st.session_state["quote_api_count"] += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.quotes(data={"symbols": symbol})
    if not resp or resp.get("code") != 200 or not resp.get("d"):
        st.error(f"Quotes API error for {symbol}: {resp.get('message', 'Unknown')}")
        return None
    return resp["d"][0]["v"]["lp"]

def compute_signals(merged_df, atm_strike, ltp):
    # Computing Δbias and Synthetic Forward Deviation
    # bias = C - P - (S - K)
    #volumeavgbias = ( (C* CallVol - P * PutVol) / (CallVol + PutVol) ) -(S - K)
    # syn_forward_price = (C - P + K) - S
    # bias: measures deviation from put-call parity
    # syn_forward_price: measures market’s implied forward direction
    # C: Call price    # P: Put price    # K: Strike    # S: Spot price

    # Recommended Update Intervals 
    # Timeframe you trade        Ideal update freq
    # Scalping (<1 min)          15–30 seconds
    # Intraday 2–15 min          ✅ 3–5 mins
    # Trend following                   10–15 mins

    strikes = sorted(merged_df["Strike"].dropna().unique().tolist())
    try:
        idx = strikes.index(atm_strike)
    except ValueError:
        st.error(f"ATM strike {atm_strike} not in available strikes")
        return "SIDEWAYS"

    # Get K1 = ATM-1, K2 = ATM, K3 = ATM+1
    target_idxs = [idx - 1, idx, idx + 1]
    selected_strikes = []
    for i in target_idxs:
        if i < 0 or i >= len(strikes):
            continue
        selected_strikes.append(strikes[i])

    bias_list = []
    for K in selected_strikes:
        row = merged_df[merged_df["Strike"] == K]
        if row.empty:
            continue
        # Use column names based on your merged_df construction
        C = row["CE_ltp"].values[0] if "CE_ltp" in row else None
        CV = row["CE_Vol"].values[0] if "CE_Vol" in row else None
        P = row["PE_ltp"].values[0] if "PE_ltp" in row else None
        PV = row["PE_Vol"].values[0] if "PE_Vol" in row else None
        st.write(f"K={K}, C={C}, CV={CV}, P={P},PV={PV}, S={ltp}, bias={((C*CV)-(P*PV)) / ((C*CV)+(P*PV))}")

        if C is None or P is None:
            continue
        S = ltp  # Spot price
        avg_price = ((C * CV) - (P * PV)) / ((C * CV) + (P * PV))
        bias_list.append(round(avg_price,2))


    if len(bias_list) < 2:
        st.warning("Not enough strikes to compute Δbias_avg")
        return "SIDEWAYS"

    delta_bias_avg = sum(bias_list) / len(bias_list)
    st.write(f"selected_strikes :{selected_strikes}")
    st.write(f"Δbias values: {bias_list}")
    st.write(f"Δbias_avg: {round(delta_bias_avg, 2)}")



#----------------------------logic2-----------------
    try:
        strikes = sorted(merged_df["Strike"].dropna().unique().tolist())
        idx = strikes.index(atm_strike)

        # ATM Strike and atm ce and atm pe prices
        atm_row = merged_df[merged_df["Strike"] == atm_strike]
        atm_ce_price = atm_row["CE_ltp"].values[0] if "CE_ltp" in atm_row else None
        atm_pe_price = atm_row["PE_ltp"].values[0] if "PE_ltp" in atm_row else None

        # Identify ITM strikes
        ce_itm_strikes = strikes[max(0, idx-3):idx] if idx >= 1 else []
        pe_itm_strikes = strikes[idx+1:min(len(strikes), idx+4)] if idx < len(strikes) - 1 else []

        # Identify OTM strikes
        ce_otm_strikes = strikes[idx+1:min(len(strikes), idx+4)] if idx < len(strikes) - 1 else []
        pe_otm_strikes = strikes[max(0, idx-3):idx] if idx >= 1 else []

        # Calculate prices
        itm_pe_price = merged_df[merged_df["Strike"].isin(pe_itm_strikes)]["PE_ltp"].sum()
        itm_ce_price = merged_df[merged_df["Strike"].isin(ce_itm_strikes)]["CE_ltp"].sum()
        otm_pe_price = merged_df[merged_df["Strike"].isin(pe_otm_strikes)]["PE_ltp"].sum()
        otm_ce_price = merged_df[merged_df["Strike"].isin(ce_otm_strikes)]["CE_ltp"].sum()

        # Calculate volume
        itm_pe_vol = merged_df[merged_df["Strike"].isin(pe_itm_strikes)]["PE_Vol"].sum()
        itm_ce_vol = merged_df[merged_df["Strike"].isin(ce_itm_strikes)]["CE_Vol"].sum()
        otm_pe_vol = merged_df[merged_df["Strike"].isin(pe_otm_strikes)]["PE_Vol"].sum()
        otm_ce_vol = merged_df[merged_df["Strike"].isin(ce_otm_strikes)]["CE_Vol"].sum()

        # ((C itm * CV) - (c otm * PV)) / ((p itm * CV) - (P otm * PV))
        # C = itm_ce_price P = itm_pe_price   CV=itm_ce_vol sum PV=itm_pe_vol sum

        numerator = ((itm_ce_price * itm_ce_vol) - (otm_ce_price * otm_ce_vol)) - ((itm_pe_price * itm_pe_vol) - (otm_pe_price * otm_pe_vol))
        denominator = abs(((itm_ce_price * itm_ce_vol) - (otm_ce_price * otm_ce_vol))) + abs(((itm_pe_price * itm_pe_vol) - (otm_pe_price * otm_pe_vol)))

        ratio = numerator / denominator if denominator != 0 else 0


    
    if delta_bias_avg > 0.2 and ratio > 0.2:
        signal = "BUY"
    elif delta_bias_avg < -0.2 and ratio < 0.2:
        signal = "SELL"
    else:
        signal = "SIDEWAYS"

    return signal




def select_strikes_atm_half_price(chain, atm_strike):
    """
    For BUY: Sell ATM PE, Buy PE with LTP ≈ half ATM PE LTP
    For SELL: Sell ATM CE, Buy CE with LTP ≈ half ATM CE LTP
    """
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

    # For BUY: Buy PE with LTP ≈ half ATM PE LTP
    buy_pe_row = None
    if sell_pe_row and "ltp" in sell_pe_row and sell_pe_row["ltp"] > 0:
        target = sell_pe_row["ltp"] / 4
        pe_candidates = pe.copy()
        pe_candidates = pe_candidates[pe_candidates["strike_price"] != atm_strike]
        pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
        if not pe_candidates.empty:
            buy_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

    # For SELL: Buy CE with LTP ≈ half ATM CE LTP
    buy_ce_row = None
    if sell_ce_row and "ltp" in sell_ce_row and sell_ce_row["ltp"] > 0:
        target = sell_ce_row["ltp"] / 4
        ce_candidates = ce.copy()
        ce_candidates = ce_candidates[ce_candidates["strike_price"] != atm_strike]
        ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target).abs()
        if not ce_candidates.empty:
            buy_ce_row = ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

    return sell_pe_row, buy_pe_row, sell_ce_row, buy_ce_row

def cancel_order(fyers, order_id):
    try:
        resp = fyers.cancel_order(data={"id": order_id})
        st.write(f"Cancelled order {order_id}: {resp}")
        logging.info(f"Cancelled order {order_id}: {resp}")
    except Exception as e:
        st.error(f"Exception while cancelling order {order_id}: {e}")
        logging.exception(f"Error cancelling order {order_id}: {e}")

def get_order_status_by_id(fyers, order_id):
    try:
        orderbook = fyers.orderbook()
        for o in orderbook.get("orderBook", []):
            if o.get("id") == order_id:
                return o.get("status")
        return None
    except Exception as e:
        st.warning(f"Could not fetch order status for {order_id}: {e}")
        return None

def place_order_and_check(fyers, order):
    try:
        resp = fyers.place_order(data=order)
        st.write(f"Order placed for {order['symbol']} ({'BUY' if order['side']==1 else 'SELL'}): {resp}")
        logging.info(f"Order placed for {order['symbol']}: {resp}")
        if resp.get("s") == "ok":
            oid = resp.get("id")
            time.sleep(2)
            status = get_order_status_by_id(fyers, oid)
            if status in [1,5]:
                st.error(f"Order {oid} was rejected or canceled.")
                return None, oid
            st.info(f"Order {oid} status: {status}")
            return oid, status
        else:
            st.error(f"Order failed: {resp}")
            return None, None
    except Exception as e:
        st.error(f"Exception placing order: {e}")
        logging.exception(f"Place order error: {e}")
        return None, None

def has_open_orders_for_last_signal(fyers):
    ids = st.session_state.get("live_last_signal_order_ids", [])
    if not ids:
        return False
    try:
        ob = fyers.orderbook().get("orderBook", [])
        # Status 2 means "open/pending"
        return any(o.get("id") in ids and o.get("status") == 2 for o in ob)
    except Exception as e:
        st.warning(f"Could not check open orders: {e}")
        return False

def handle_paper_trade(signal, sell_pe, buy_pe, sell_ce, buy_ce):
    orders = []
    lot = 75
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
    current_signal = st.session_state.get("paper_last_signal")

    # Skip if signal is the same as last time (no duplicate orders)
    if current_signal == signal and st.session_state.paper_positions:
        st.write(f"[PAPER] Signal {signal} is already active. No new trades.")
        return []

    # Case 1: BUY → SELL (Exit BUY, Enter SELL)
    if current_signal == "BUY" and signal == "SELL":
        # Exit all BUY positions first
        for symbol, position in list(st.session_state.paper_positions.items()):
            close_action = "SELL" if position["action"] == "BUY" else "BUY"
            orders.append({
                "symbol": symbol,
                "qty": position["qty"],
                "action": close_action,
                "price": position["current_price"],
                "type": "CLOSE"
            })
        st.write("[PAPER] Exiting BUY positions for SELL signal")

        # Enter SELL basket (Buy CE half-price & Sell CE ATM)
        if buy_ce and sell_ce:
            buy_symbol = buy_ce["symbol"]
            sell_symbol = sell_ce["symbol"]
            orders.append({"symbol": buy_symbol, "qty": lot, "side": 1, "action": "BUY", "price": buy_ce["ltp"]})
            orders.append({"symbol": sell_symbol, "qty": lot, "side": -1, "action": "SELL", "price": sell_ce["ltp"]})
            st.write(f"[PAPER] BUY {buy_symbol} @ {buy_ce['ltp']} & SELL {sell_symbol} @ {sell_ce['ltp']}")

    # Case 2: SELL → BUY (Exit SELL, Enter BUY)
    elif current_signal == "SELL" and signal == "BUY":
        # Exit all SELL positions first
        for symbol, position in list(st.session_state.paper_positions.items()):
            close_action = "SELL" if position["action"] == "BUY" else "BUY"
            orders.append({
                "symbol": symbol,
                "qty": position["qty"],
                "action": close_action,
                "price": position["current_price"],
                "type": "CLOSE"
            })
        st.write("[PAPER] Exiting SELL positions for BUY signal")

        # Enter BUY basket (Buy PE half-price & Sell PE ATM)
        if buy_pe and sell_pe:
            buy_symbol = buy_pe["symbol"]
            sell_symbol = sell_pe["symbol"]
            orders.append({"symbol": buy_symbol, "qty": lot, "side": 1, "action": "BUY", "price": buy_pe["ltp"]})
            orders.append({"symbol": sell_symbol, "qty": lot, "side": -1, "action": "SELL", "price": sell_pe["ltp"]})
            st.write(f"[PAPER] BUY {buy_symbol} @ {buy_pe['ltp']} & SELL {sell_symbol} @ {sell_pe['ltp']}")

    # Case 3: New signal (no existing position)
    elif current_signal is None and signal in ["BUY", "SELL"]:
        if signal == "BUY" and buy_pe and sell_pe:
            buy_symbol = buy_pe["symbol"]
            sell_symbol = sell_pe["symbol"]
            orders.append({"symbol": buy_symbol, "qty": lot, "side": 1, "action": "BUY", "price": buy_pe["ltp"]})
            orders.append({"symbol": sell_symbol, "qty": lot, "side": -1, "action": "SELL", "price": sell_pe["ltp"]})
            st.write(f"[PAPER] BUY {buy_symbol} @ {buy_pe['ltp']} & SELL {sell_symbol} @ {sell_pe['ltp']}")
        elif signal == "SELL" and buy_ce and sell_ce:
            buy_symbol = buy_ce["symbol"]
            sell_symbol = sell_ce["symbol"]
            orders.append({"symbol": buy_symbol, "qty": lot, "side": 1, "action": "BUY", "price": buy_ce["ltp"]})
            orders.append({"symbol": sell_symbol, "qty": lot, "side": -1, "action": "SELL", "price": sell_ce["ltp"]})
            st.write(f"[PAPER] BUY {buy_symbol} @ {buy_ce['ltp']} & SELL {sell_symbol} @ {sell_ce['ltp']}")

    # Case 4: SIDEWAYS (do nothing)
    elif signal == "SIDEWAYS":
        st.write("[PAPER] SIDEWAYS: Holding existing positions")
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
            "type": "OPEN" if "type" not in order else order["type"]
        }
        update_paper_positions(trade)
        st.session_state.trade_history.append(trade)
        placed_ids.append(trade_id)
        st.write(f"[PAPER] {order['action']} {order['qty']} of {order['symbol']} at {order['price']}")

    # Update last_signal if new positions opened
    if placed_ids and signal in ["BUY", "SELL"]:
        st.session_state["paper_last_signal"] = signal

    return placed_ids

def update_paper_positions(trade):
    symbol = trade["symbol"]
    timestamp = trade["timestamp"]
    if trade["type"] == "OPEN":
        # Open new position
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
        # Close position
        if symbol in st.session_state.paper_positions:
            position = st.session_state.paper_positions[symbol]

            # Calculate PnL
            if position["action"] == "BUY":  # Long position
                pnl = (trade["price"] - position["entry_price"]) * position["qty"]
            else:  # Short position
                pnl = (position["entry_price"] - trade["price"]) * position["qty"]

            # Update realized PnL
            st.session_state.paper_pnl["realized"] += pnl

            # Record closing trade with PnL
            trade["pnl"] = pnl

            # Remove position
            del st.session_state.paper_positions[symbol]
            st.write(f"[PAPER] Closed {position['action']} position for {symbol}. PnL: {pnl:.2f}")

            # Update last signal if position is closed
            if not st.session_state.paper_positions:
                st.session_state["last_signal"] = None
        else:
            st.warning(f"[PAPER] No position found to close for {symbol}")

def update_unrealized_pnl(cid, token):
    """Update unrealized PnL using current market prices"""
    if not st.session_state.paper_positions:
        st.session_state.paper_pnl["unrealized"] = 0.0
        return

    total_unrealized = 0.0
    for symbol, position in st.session_state.paper_positions.items():
        # Get current LTP for the symbol
        ltp = get_symbol_ltp(cid, token, symbol)
        if ltp is None:
            # If we can't get current price, use last known price
            ltp = position.get("current_price", position["entry_price"])

        # Update current price
        position["current_price"] = ltp

        # Calculate position PnL
        if position["action"] == "BUY":  # Long position
            pnl = (ltp - position["entry_price"]) * position["qty"]
        else:  # Short position
            pnl = (position["entry_price"] - ltp) * position["qty"]

        total_unrealized += pnl

    st.session_state.paper_pnl["unrealized"] = total_unrealized
    st.session_state.pnl_update_time = datetime.now()

def update_position_prices(cid, token):
    if not st.session_state.paper_positions:
        return

    for symbol, position in st.session_state.paper_positions.items():
        ltp = get_symbol_ltp(cid, token, symbol)
        if ltp is not None:
            position["current_price"] = ltp

def handle_basket_orders_atomic(signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers):
    orders = []
    lot = 75
    current_signal = st.session_state.get("last_signal")

    # Skip if signal is the same as last time (no duplicates)
    if current_signal == signal:
        st.warning(f"Signal {signal} is already active. No new trades.")
        return []

    # Case 1: BUY → SELL (Exit BUY, Enter SELL)
    if current_signal == "BUY" and signal == "SELL":
        # Exit all BUY positions first
        try:
            positions = fyers.positions().get("netPositions", [])
            for p in positions:
                if p.get("segment") == "NFO" and p.get("qty") != 0:
                    side = -1 if p["qty"] > 0 else 1  # Reverse position
                    orders.append({
                        "symbol": p["symbol"],
                        "qty": abs(int(p["qty"])),
                        "type": 2,  # Market order
                        "side": side,
                        "productType": "MARGIN",
                        "validity": "DAY"
                    })
            st.write("[LIVE] Exiting BUY positions for SELL signal")
        except Exception as e:
            st.error(f"Failed to exit positions: {e}")
            return []

        # Enter SELL basket (Buy CE half-price & Sell CE ATM)
        if buy_ce and sell_ce:
            orders.append({
                "symbol": buy_ce["symbol"], 
                "qty": lot, 
                "type": 2, 
                "side": 1,  # Buy CE
                "productType": "MARGIN",
                "validity": "DAY"
            })
            orders.append({
                "symbol": sell_ce["symbol"], 
