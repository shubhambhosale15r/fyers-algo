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

def enforce_rate_limit():
    now = datetime.now()
    window = now - st.session_state["api_window_start"]
    total_calls = st.session_state["option_chain_api_count"] + st.session_state["quote_api_count"]
    if window > timedelta(seconds=60):
        st.session_state["api_window_start"] = now
        st.session_state["option_chain_api_count"] = 0
        st.session_state["quote_api_count"] = 0
    elif total_calls >= 500:
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


def compute_signals(merged_df, atm_strike):
    strikes = sorted(merged_df["Strike"].dropna().unique().tolist())
    try:
        idx = strikes.index(atm_strike)
    except ValueError:
        st.error(f"ATM strike {atm_strike} not in available strikes")
        return "SIDEWAYS"

    # Identify ITM strikes (3 closest below ATM for CE, above ATM for PE)
    ce_itm_strikes = []
    pe_itm_strikes = []
    if idx >= 1:  # Ensure there are strikes below ATM
        ce_itm_strikes = strikes[max(0, idx-3):idx]  # Up to 3 strikes below ATM
    if idx < len(strikes) - 1:  # Ensure there are strikes above ATM
        pe_itm_strikes = strikes[idx+1:min(len(strikes), idx+4)]  # Up to 3 strikes above ATM

    # Identify OTM strikes (3 closest above ATM for CE, below ATM for PE)
    ce_otm_strikes = []
    pe_otm_strikes = []
    if idx < len(strikes) - 1:  # Ensure there are strikes above ATM
        ce_otm_strikes = strikes[idx+1:min(len(strikes), idx+4)]  # Up to 3 strikes above ATM
    if idx >= 1:  # Ensure there are strikes below ATM
        pe_otm_strikes = strikes[max(0, idx-3):idx]  # Up to 3 strikes below ATM

    # Compute PCR OI for ITM and OTM
    itm_pe_oi = merged_df[merged_df["Strike"].isin(pe_itm_strikes)]["PE_OICh"].sum()
    itm_ce_oi = merged_df[merged_df["Strike"].isin(ce_itm_strikes)]["CE_OICh"].sum()
    otm_pe_oi = merged_df[merged_df["Strike"].isin(pe_otm_strikes)]["PE_OICh"].sum()
    otm_ce_oi = merged_df[merged_df["Strike"].isin(ce_otm_strikes)]["CE_OICh"].sum()

    # Avoid division by zero
    if itm_pe_oi > 0 and itm_ce_oi > 0 and otm_pe_oi > 0 and otm_ce_oi > 0:
        itm_pcr = itm_pe_oi / itm_ce_oi 
        otm_pcr = otm_pe_oi / otm_ce_oi 

        # Generate signals based on PCR values
        if itm_pcr < 0.8 and otm_pcr > 1.2:
            signal = "BUY"
        elif itm_pcr > 1.2 and otm_pcr < 0.8:
            signal = "SELL"
        else:
            signal = "SIDEWAYS"
        st.write(f"ITM PCR: {itm_pcr:.4f} (PE OI: {itm_pe_oi}, CE OI: {itm_ce_oi})")
        st.write(f"OTM PCR: {otm_pcr:.4f} (PE OI: {otm_pe_oi}, CE OI: {otm_ce_oi})")
    else:
        if itm_pe_oi < 0 and itm_ce_oi > 0 and otm_pe_oi > 0 and otm_pe_oi > 1.5 * otm_ce_oi or otm_ce_oi < 0:
            signal = "BUY"
        elif itm_pe_oi > 0 and itm_ce_oi < 0 and otm_ce_oi > 0 and otm_ce_oi > 1.5 * otm_pe_oi or otm_pe_oi < 0:
            signal = "SELL"
        else:
            signal = "SIDEWAYS"
        st.write(f"ITM :  PE OI: {itm_pe_oi}, CE OI: {itm_ce_oi}")
        st.write(f"OTM :  PE OI: {otm_pe_oi}, CE OI: {otm_ce_oi}")
        if otm_pe_oi > 1.5 * otm_ce_oi:
            st.write("50% more otm pe compared to ce")
        elif otm_ce_oi > 1.5 * otm_pe_oi:
            st.write("50% more otm ce compared to pe")

    return signal


def select_strikes_based_on_oi(chain):
    """
    Given the raw option chain list of dicts, return:
    - For BUY signal: (sell_pe_row, buy_pe_row)
    - For SELL signal: (sell_ce_row, buy_ce_row)
    """
    df = pd.DataFrame(chain)
    if df.empty:
        return None, None, None, None

    ce = df[df["option_type"] == "CE"].copy()
    pe = df[df["option_type"] == "PE"].copy()

    # Ensure numeric OI change exists
    pe["oich_numeric"] = pd.to_numeric(pe.get("oich", 0), errors="coerce").fillna(0)
    ce["oich_numeric"] = pd.to_numeric(ce.get("oich", 0), errors="coerce").fillna(0)

    # SELL_PE: the PE strike with highest increase in OI
    if not pe.empty:
        sell_pe_idx = pe["oich_numeric"].idxmax()
        sell_pe_row = pe.loc[sell_pe_idx].to_dict()
    else:
        sell_pe_row = None

    # SELL_CE: the CE strike with highest increase in OI
    if not ce.empty:
        sell_ce_idx = ce["oich_numeric"].idxmax()
        sell_ce_row = ce.loc[sell_ce_idx].to_dict()
    else:
        sell_ce_row = None

    # For BUY signal: find a PE with LTP approx half of sell_pe row's LTP
    buy_pe_row = None
    if sell_pe_row is not None:
        target_half = sell_pe_row.get("ltp", 0) / 2
        pe_candidates = pe.copy()
        pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target_half).abs()
        if not pe_candidates.empty:
            buy_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

    # For SELL signal: find a CE with LTP approx half of sell_ce row's LTP
    buy_ce_row = None
    if sell_ce_row is not None:
        target_half = sell_ce_row.get("ltp", 0) / 2
        ce_candidates = ce.copy()
        ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target_half).abs()
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
    ids = st.session_state.get("last_signal_order_ids", [])
    if not ids:
        return False
    try:
        ob = fyers.orderbook().get("orderBook", [])
        return any(o.get("id") in ids and o.get("status") in [4,6] for o in ob)
    except:
        return False


def handle_paper_trade(signal, sell_pe, buy_pe, sell_ce, buy_ce):
    orders = []
    lot = 75
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
    current_signal = st.session_state.get("last_signal")

    # Skip if signal is the same as last time (no duplicate orders)
    if current_signal == signal:
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

        # Enter SELL basket (Buy CE half-price & Sell CE highest OI-change)
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

        # Enter BUY basket (Buy PE half-price & Sell PE highest OI-change)
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
        st.session_state["last_signal"] = signal

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

        # Enter SELL basket (Buy CE half-price & Sell CE highest OI-change)
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
                "qty": lot, 
                "type": 2, 
                "side": -1,  # Sell CE
                "productType": "MARGIN",
                "validity": "DAY"
            })
            st.write(f"[LIVE] BUY {buy_ce['symbol']} & SELL {sell_ce['symbol']}")

    # Case 2: SELL → BUY (Exit SELL, Enter BUY)
    elif current_signal == "SELL" and signal == "BUY":
        # Exit all SELL positions first
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
            st.write("[LIVE] Exiting SELL positions for BUY signal")
        except Exception as e:
            st.error(f"Failed to exit positions: {e}")
            return []

        # Enter BUY basket (Buy PE half-price & Sell PE highest OI-change)
        if buy_pe and sell_pe:
            orders.append({
                "symbol": buy_pe["symbol"], 
                "qty": lot, 
                "type": 2, 
                "side": 1,  # Buy PE
                "productType": "MARGIN",
                "validity": "DAY"
            })
            orders.append({
                "symbol": sell_pe["symbol"], 
                "qty": lot, 
                "type": 2, 
                "side": -1,  # Sell PE
                "productType": "MARGIN",
                "validity": "DAY"
            })
            st.write(f"[LIVE] BUY {buy_pe['symbol']} & SELL {sell_pe['symbol']}")

    # Case 3: New signal (no existing position)
    elif current_signal is None and signal in ["BUY", "SELL"]:
        if signal == "BUY" and buy_pe and sell_pe:
            orders.append({
                "symbol": buy_pe["symbol"], 
                "qty": lot, 
                "type": 2, 
                "side": 1,  # Buy PE
                "productType": "MARGIN",
                "validity": "DAY"
            })
            orders.append({
                "symbol": sell_pe["symbol"], 
                "qty": lot, 
                "type": 2, 
                "side": -1,  # Sell PE
                "productType": "MARGIN",
                "validity": "DAY"
            })
            st.write(f"[LIVE] BUY {buy_pe['symbol']} & SELL {sell_pe['symbol']}")
        elif signal == "SELL" and buy_ce and sell_ce:
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
                "qty": lot, 
                "type": 2, 
                "side": -1,  # Sell CE
                "productType": "MARGIN",
                "validity": "DAY"
            })
            st.write(f"[LIVE] BUY {buy_ce['symbol']} & SELL {sell_ce['symbol']}")

    # Case 4: SIDEWAYS (do nothing)
    elif signal == "SIDEWAYS":
        st.write("[LIVE] SIDEWAYS: Holding positions")
        return []

    # Execute orders
    placed_ids = []
    for order in orders:
        oid, status = place_order_and_check(fyers, order)
        if oid is None or status in [1, 5]:  # Rejected/Canceled
            st.error(f"Order failed: {order['symbol']}")
            # Cancel all pending orders
            for pid in placed_ids:
                cancel_order(fyers, pid)
            return []
        placed_ids.append(oid)

    # Update last_signal if successful
    if placed_ids:
        st.session_state["last_signal"] = signal
        st.success(f"[LIVE] {signal} basket executed!")
    
    return placed_ids


def format_and_show(chain, title, ltp, show_signals=False):
    df = pd.DataFrame(chain)
    if df.empty:
        st.info(f"No data for {title}")
        return None
    ce = df[df["option_type"] == "CE"]
    pe = df[df["option_type"] == "PE"]
    cols = ["symbol", "strike_price", "oi", "volume", "ltp", "ask", "bid", "ltpch", "ltpchp", "oich", "oichp", "prev_oi"]
    ce_df = ce[cols].rename(columns={"strike_price": "Strike", "oi": "OI", "volume": "Vol", "ltpch": "LTPCh", "ltpchp": "LTPChP", "oich": "OICh", "oichp": "OIChP", "prev_oi": "PrevOI"})
    pe_df = pe[cols].rename(columns={"strike_price": "Strike", "oi": "OI", "volume": "Vol", "ltpch": "LTPCh", "ltpchp": "LTPChP", "oich": "OICh", "oichp": "OIChP", "prev_oi": "PrevOI"})
    merged = pd.merge(
        ce_df.add_prefix("CE_"),
        pe_df.add_prefix("PE_"),
        left_on="CE_Strike", right_on="PE_Strike",
        how="outer"
    ).rename(columns={"CE_Strike": "Strike"}).drop("PE_Strike", axis=1)
    atm = None
    if ltp is not None and not merged["Strike"].dropna().empty:
        atm = min(merged["Strike"].dropna(), key=lambda x: abs(x - ltp))
    st.subheader(title)
    if show_signals and atm is not None:
        signal = compute_signals(merged, atm)
        st.metric("PCR Signal", signal)  # Show just the PCR signal
    styled = merged.style.apply(
        lambda row: ["background:yellow" if row["Strike"] == atm else "" for _ in row],
        axis=1
    )
    st.dataframe(styled)
    st.caption(f"ATM: {atm} • LTP: {ltp}")
    if show_signals and atm is not None:
        return signal
    return None

def show_signal_history():
    st.subheader("📜 Signal History")
    st_autorefresh(interval=60_000 )  # 60 000 ms = 60 seconds
    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    
    
    if not st.session_state.get("signal_history", []):
        st.info("No signal history available yet")
        return
    
    # Create and display dataframe
    history_df = pd.DataFrame(st.session_state.signal_history)
    history_df = history_df.sort_values("timestamp", ascending=False)
    
    # Format the DataFrame for better display
    formatted_df = history_df.copy()
    formatted_df["underlying_ltp"] = formatted_df["underlying_ltp"].apply(lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x)
    
    st.dataframe(formatted_df)
    
    # Download button
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download Signal History",
        data=csv,
        file_name="signal_history.csv",
        mime="text/csv"
    )

def show_paper_trading_page(cid, token):
    # update the paper trade dashboard @ 30 seconds
    count = st_autorefresh(interval=30000, key="paper_refresh")
    st.subheader("📝 Paper Trading Dashboard")
    
    # Display PnL summary
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Realized PnL", f"₹{st.session_state.paper_pnl['realized']:.2f}")
    col2.metric("📊 Unrealized PnL", f"₹{st.session_state.paper_pnl['unrealized']:.2f}")
    total_pnl = st.session_state.paper_pnl['realized'] + st.session_state.paper_pnl['unrealized']
    col3.metric("💵 Total PnL", f"₹{total_pnl:.2f}", delta=f"{total_pnl:.2f}")
    
    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Current positions
    st.subheader("📊 Current Positions")
    if st.session_state.paper_positions:
        positions_list = []
        for symbol, position in st.session_state.paper_positions.items():
            # Calculate current PnL for each position
            if position["action"] == "BUY":  # Long position
                pnl = (position["current_price"] - position["entry_price"]) * position["qty"]
            else:  # Short position
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
    
    # Trade history
    st.subheader("📜 Trade History")
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        
        # Reorder columns for better readability
        if not history_df.empty:
            desired_columns = ["timestamp", "symbol", "action", "qty", "price", "signal", "type", "pnl"]
            available_columns = [col for col in desired_columns if col in history_df.columns]
            history_df = history_df[available_columns]
            
            # Format timestamp
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
            history_df["timestamp"] = history_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Sort by timestamp
            history_df = history_df.sort_values("timestamp", ascending=False)
            
            # Format prices and PnL
            if "price" in history_df:
                history_df["price"] = history_df["price"].apply(lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x)
            if "pnl" in history_df:
                history_df["pnl"] = history_df["pnl"].apply(lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(history_df)
        
        # Download button
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
    
    # Reset button
    if st.button("🔄 Reset Paper Trading"):
        st.session_state.paper_trades = []
        st.session_state.paper_positions = {}
        st.session_state.paper_pnl = {"realized": 0.0, "unrealized": 0.0}
        st.session_state.trade_history = []
        st.session_state["last_signal"] = None
        st.success("Paper trading data reset!")


def main():
    st.title("Fyers Algo Trading")
    
    # Initialize session state
    if "cid" not in st.session_state:
        st.session_state.cid = ""
    if "token" not in st.session_state:
        st.session_state.token = ""
    # Add to SESSION STATE INITIALIZATION
    if "signal_history" not in st.session_state:
        st.session_state.signal_history = []
    
    # Settings sidebar
    with st.sidebar:
        st.subheader("⚙️ Trading Settings")
        auto_trade = st.checkbox("Enable Auto Trade", value=False)
        paper_trade = st.checkbox("📝 Enable Paper Trading", value=False)
        
        # Navigation buttons
        st.subheader("📊 Navigation")
        if st.button("📈 Signal History"):
            st.session_state.page = "signal_history"
        if st.button("🔙 Back to Trading"):
            st.session_state.page = "trading"
        
        # Paper trading page navigation
        if paper_trade:
            if st.button("📊 Paper Trading Dashboard"):
                st.session_state.page = "paper_trading"
        
        st.subheader("🔐 API Credentials")
        cid = st.text_input("Client ID", value=st.session_state.cid)
        token = st.text_input("Access Token", type="password", value=st.session_state.token)
        sym = st.text_input("Symbol (e.g. NSE:NIFTY50-INDEX)", "NSE:NIFTY50-INDEX")
        
        # Store credentials in session state
        st.session_state.cid = cid
        st.session_state.token = token
        
        # Display rate limit info
        st.subheader("📊 API Usage")
        oc = st.session_state.get("option_chain_api_count", 0)
        qc = st.session_state.get("quote_api_count", 0)
        st.write(f"Option Chain API: {oc}")
        st.write(f"Quote API: {qc}")
        st.write(f"Total: {oc+qc}/500 per minute")
    
    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "trading"
    
    # AUTO-REFRESH PAPER TRADING DATA
    # This will run regardless of current page
    if paper_trade and st.session_state.paper_positions:
        update_position_prices(cid, token)
        update_unrealized_pnl(cid, token)

        # Page routing
    if st.session_state.page == "signal_history":
        show_signal_history()
        return

    
    # Paper trading page
    if st.session_state.page == "paper_trading" and paper_trade:
        show_paper_trading_page(cid, token)
        return
    
    # Main trading page
    st.subheader("📈 Live Trading")
    if not (cid and token and sym):
        st.info("Enter credentials in the sidebar.")
        return
    
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    enforce_rate_limit()
    st.session_state["option_chain_api_count"] += 1
    
    meta = fy.optionchain(data={"symbol": sym, "timestamp": ""})
    if meta.get("code") != 200:
        st.error("Failed to fetch expiry data")
        return
    
    exp = sorted([(e["date"], e.get("expiry")) for e in meta["data"]["expiryData"]],
                 key=lambda x: datetime.strptime(x[0], "%d-%m-%Y"))
    today = datetime.today().date()
    dates = [datetime.strptime(d, "%d-%m-%Y").date() for d, _ in exp]
    idx = next((i for i, d in enumerate(dates) if d >= today), 0)
    curr_date, curr_ts = exp[idx]
    
    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    
    ltp = get_underlying_ltp(cid, token, sym)
    if ltp:
        st.success(f"📊 Underlying LTP: {ltp}")
    
    chain = get_option_chain_data(cid, token, sym, expiry_ts=curr_ts)
    if chain:
        # Display the option chain and PCR signal
        signal = format_and_show(chain, f"Current Expiry: {curr_date}", ltp, show_signals=True)
                # RECORD SIGNAL TO HISTORY
        if signal and ltp:
            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.signal_history.append({
                "timestamp": timestamp,
                "signal": signal,
                "underlying_ltp": ltp,
                "expiry_date": curr_date
            })
        
        # Select strikes based on highest OI change and half-price logic
        sell_pe, buy_pe, sell_ce, buy_ce = select_strikes_based_on_oi(chain)
        
        # Validate that we have valid pairs for BUY or SELL
        valid_buy = (sell_pe is not None and buy_pe is not None)
        valid_sell = (sell_ce is not None and buy_ce is not None)
        
        if signal == "BUY" and not valid_buy:
            st.warning("Missing option data for BUY signal. Unable to place trade.")
        elif signal == "SELL" and not valid_sell:
            st.warning("Missing option data for SELL signal. Unable to place trade.")
        elif signal in ("BUY", "SELL"):
            # PAPER TRADE: Execute immediately if paper trading is enabled
            if paper_trade:
                st.write(f"📢 Signal detected: {signal} - Attempting paper trade")
                ids = handle_paper_trade(signal, sell_pe, buy_pe, sell_ce, buy_ce)
                st.session_state["last_signal_order_ids"] = ids
            
            # REAL TRADE: Only execute if auto trade is enabled
            elif auto_trade:
                if st.session_state["last_signal"] == signal and has_open_orders_for_last_signal(fy):
                    st.info(f"Basket for {signal} already open.")
                else:
                    ids = handle_basket_orders_atomic(signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
                    st.session_state["last_signal"] = signal
                    st.session_state["last_signal_order_ids"] = ids
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
    
    # time.sleep(60)
    # st.rerun()

if __name__ == "__main__":
    logging.basicConfig(filename="trading_debug.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    main()
