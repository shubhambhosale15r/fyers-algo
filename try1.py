import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from fyers_apiv3 import fyersModel
import logging

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

def compute_signals(merged_df):
    ce_ltpch_sum = merged_df["CE_LTPCh"].sum()
    pe_ltpch_sum = merged_df["PE_LTPCh"].sum()
    st.write(f"CE LTP change sum: {ce_ltpch_sum}, PE LTP change sum: {pe_ltpch_sum}")
    if ce_ltpch_sum > pe_ltpch_sum and ce_ltpch_sum > 0:
        pure_price_signal = "BUY"
    elif pe_ltpch_sum > ce_ltpch_sum and pe_ltpch_sum > 0:
        pure_price_signal = "SELL"
    else:
        pure_price_signal = "SIDEWAYS"
    st.write(f"Pure price signal: {pure_price_signal}")

    merged_df["normalized_ce_bid"] = merged_df["CE_bid"] * merged_df["CE_Vol"] / merged_df["CE_Vol"].sum()
    merged_df["normalized_pe_bid"] = merged_df["PE_bid"] * merged_df["PE_Vol"] / merged_df["PE_Vol"].sum()
    merged_df["normalized_ce_ask"] = merged_df["CE_ask"] * merged_df["CE_Vol"] / merged_df["CE_Vol"].sum()
    merged_df["normalized_pe_ask"] = merged_df["PE_ask"] * merged_df["PE_Vol"] / merged_df["PE_Vol"].sum()

    ce_bid_sum = merged_df["normalized_ce_bid"].sum()
    pe_bid_sum = merged_df["normalized_pe_bid"].sum()
    ce_ask_sum = merged_df["normalized_ce_ask"].sum()
    pe_ask_sum = merged_df["normalized_pe_ask"].sum()
    st.write(f"CE bid sum: {ce_bid_sum}, PE bid sum: {pe_bid_sum}")
    st.write(f"CE ask sum: {ce_ask_sum}, PE ask sum: {pe_ask_sum}")

    if ce_bid_sum > pe_bid_sum and ce_ask_sum > pe_ask_sum:
        normalized_signal = "BUY"
    elif pe_bid_sum > ce_bid_sum and pe_ask_sum > ce_ask_sum:
        normalized_signal = "SELL"
    else:
        normalized_signal = "SIDEWAYS"
    st.write(f"Normalized signal: {normalized_signal}")

    if pure_price_signal == "BUY" and normalized_signal == "BUY":
        final_signal = 'BUY'
    elif pure_price_signal == "SELL" and normalized_signal == "SELL":
        final_signal = "SELL"
    else:
        final_signal = "SIDEWAYS"
    st.write(f"Final trading signal: {final_signal}")

    return pure_price_signal, normalized_signal, final_signal

def get_atm_and_half_price_option(chain, ltp):
    df = pd.DataFrame(chain)
    if df.empty:
        return None, None, None, None, None, None

    ce = df[df["option_type"] == "CE"].copy()
    pe = df[df["option_type"] == "PE"].copy()
    ce = ce[["symbol", "strike_price", "oi", "volume", "ltp", "ask", "bid", "ltpch", "ltpchp", "oich", "oichp", "prev_oi"]]
    pe = pe[["symbol", "strike_price", "oi", "volume", "ltp", "ask", "bid", "ltpch", "ltpchp", "oich", "oichp", "prev_oi"]]

    strikes = df["strike_price"].dropna().unique()
    strikes = sorted(strikes)
    atm_strike = min(strikes, key=lambda x: abs(x - ltp))
    atm_ce = ce[ce["strike_price"] == atm_strike]
    atm_pe = pe[pe["strike_price"] == atm_strike]
    if atm_ce.empty or atm_pe.empty:
        return None, None, None, None, None, None
    atm_ce_row = atm_ce.iloc[0].to_dict()
    atm_pe_row = atm_pe.iloc[0].to_dict()

    half_pe_row = None
    pe_candidates = pe[pe["strike_price"] < atm_strike].copy()
    if not pe_candidates.empty:
        target_half_pe = atm_pe_row["ltp"] / 2
        pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target_half_pe).abs()
        half_pe_row = pe_candidates.sort_values("ltp_diff").iloc[0].to_dict()

    half_ce_row = None
    ce_candidates = ce[ce["strike_price"] > atm_strike].copy()
    if not ce_candidates.empty:
        target_half_ce = atm_ce_row["ltp"] / 2
        ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target_half_ce).abs()
        half_ce_row = ce_candidates.sort_values("ltp_diff").iloc[0].to_dict()

    return atm_strike, atm_ce_row, atm_pe_row, half_ce_row, half_pe_row, df

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
        st.write(f"Order placed for {order['symbol']} ({'BUY' if order['side'] == 1 else 'SELL'}):", resp)
        logging.info(f"Order placed for {order['symbol']} ({'BUY' if order['side'] == 1 else 'SELL'}): {resp}")
        if isinstance(resp, dict) and resp.get("s") == "ok":
            order_id = resp.get("id")
            # Wait for order to show up
            time.sleep(2)
            status_code = get_order_status_by_id(fyers, order_id)
            if status_code is None:
                st.warning(f"Order {order_id} not found in order book!")
                return None, None
            if status_code in [5, 1]:
                st.error(f"Order {order_id} was REJECTED or CANCELED!")
                return None, order_id
            st.info(f"Order {order_id} status code: {status_code}")
            return order_id, status_code
        else:
            st.error(f"Order failed for {order['symbol']}! Response: {resp}")
            return None, None
    except Exception as e:
        st.error(f"Exception while placing order for {order['symbol']}: {e}")
        logging.exception(f"Order error for {order['symbol']}: {e}")
        return None, None

def has_open_orders_for_last_signal(fyers):
    order_ids = st.session_state.get("last_signal_order_ids", [])
    if not order_ids:
        return False
    try:
        orderbook = fyers.orderbook()
        for o in orderbook.get("orderBook", []):
            if o.get("id") in order_ids and o.get("status") in [4, 6]:
                return True
    except Exception as e:
        st.warning(f"Could not check for open orders: {e}")
    return False

def handle_basket_orders_atomic(signal, atm_strike, atm_ce, atm_pe, half_ce, half_pe, fyers):
    orders = []
    lot_size = 75
    if signal == "BUY":
        if not (atm_pe and half_pe):
            st.error("ATM PE or half-price PE not found.")
            return []
        orders.append({
            "symbol": half_pe["symbol"],
            "qty": lot_size,
            "type": 2,
            "side": 1,
            "productType": "MARGIN",
            "validity": "DAY",
        })
        orders.append({
            "symbol": atm_pe["symbol"],
            "qty": lot_size,
            "type": 2,
            "side": -1,
            "productType": "MARGIN",
            "validity": "DAY",
        })
        st.write(f"BUY {half_pe['symbol']} (half price PE), SELL {atm_pe['symbol']} (ATM PE)")
    elif signal == "SELL":
        if not (atm_ce and half_ce):
            st.error("ATM CE or half-price CE not found.")
            return []
        orders.append({
            "symbol": half_ce["symbol"],
            "qty": lot_size,
            "type": 2,
            "side": 1,
            "productType": "MARGIN",
            "validity": "DAY",
        })
        orders.append({
            "symbol": atm_ce["symbol"],
            "qty": lot_size,
            "type": 2,
            "side": -1,
            "productType": "MARGIN",
            "validity": "DAY",
        })
        st.write(f"BUY {half_ce['symbol']} (half price CE), SELL {atm_ce['symbol']} (ATM CE)")
    elif signal == "SIDEWAYS":
        try:
            posbook = fyers.positions()
            for pos in posbook.get("netPositions", []):
                if pos.get("segment", "").upper() == "NFO" and pos.get("qty", 0) != 0:
                    qty = abs(int(pos["qty"]))
                    side = -1 if int(pos["qty"]) > 0 else 1
                    orders.append({
                        "symbol": pos["symbol"],
                        "qty": qty,
                        "type": 2,
                        "side": side,
                        "productType": "MARGIN",
                        "validity": "DAY",
                    })
            if orders:
                st.write("Exiting all F&O positions...")
            else:
                st.info("No open F&O positions to exit.")
        except Exception as e:
            st.error(f"Failed to fetch/exit F&O positions: {e}")
            logging.exception(f"Positions fetch/exit error: {e}")
            return []
        for order in orders:
            place_order_and_check(fyers, order)
        st.session_state["last_signal"] = None
        st.session_state["last_signal_order_ids"] = []
        return []
    else:
        st.info("No valid signal. No basket order created.")
        return []

    placed_order_ids = []
    order_statuses = []
    for idx, order in enumerate(orders):
        order_id, status_code = place_order_and_check(fyers, order)
        if order_id is not None and status_code is not None and status_code not in [1, 5]:
            placed_order_ids.append(order_id)
            order_statuses.append(status_code)
            continue
        else:
            st.error("Basket execution failed! Cancelling all placed orders.")
            for oid in placed_order_ids:
                cancel_order(fyers, oid)
            st.session_state["last_signal_order_ids"] = []
            return []
    st.success("All basket orders executed successfully!")
    return placed_order_ids

def format_and_show(chain, title, ltp, show_signals=False):
    df = pd.DataFrame(chain)
    if df.empty:
        st.info(f"No data for {title}")
        return None, None, None, None

    ce = df[df["option_type"] == "CE"]
    pe = df[df["option_type"] == "PE"]

    cols = [
        "symbol", "strike_price", "oi", "volume", "ltp",
        "ask", "bid", "ltpch", "ltpchp", "oich", "oichp", "prev_oi"
    ]
    ce_df = ce[cols].rename(columns={"strike_price": "Strike", "oi": "OI", "volume": "Vol",
                                    "ltpch": "LTPCh", "ltpchp": "LTPChP", "oich": "OICh",
                                    "oichp": "OIChP", "prev_oi": "PrevOI"})
    pe_df = pe[cols].rename(columns={"strike_price": "Strike", "oi": "OI", "volume": "Vol",
                                    "ltpch": "LTPCh", "ltpchp": "LTPChP", "oich": "OICh",
                                    "oichp": "OIChP", "prev_oi": "PrevOI"})

    merged = pd.merge(
        ce_df.add_prefix("CE_"),
        pe_df.add_prefix("PE_"),
        left_on="CE_Strike", right_on="PE_Strike", how="outer"
    ).rename(columns={"CE_Strike": "Strike"}).drop("PE_Strike", axis=1)

    st.subheader(title)

    if show_signals:
        pure_sig, norm_sig, final_sig = compute_signals(merged)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pure-Price Signal", pure_sig)
        c2.metric("Normalized Signal", norm_sig)
        c3.metric("Final Signal", final_sig)
    else:
        pure_sig = norm_sig = final_sig = None

    if ltp is not None:
        atm = min(merged["Strike"].dropna(), key=lambda x: abs(x - ltp))
        styled = merged.style.apply(
            lambda row: ["background: yellow" if row["Strike"] == atm else "" for _ in row], axis=1
        )
        st.dataframe(styled)
        st.caption(f"ATM Strike: {atm}  •  Underlying LTP: {ltp}")
    else:
        st.dataframe(merged)
        atm = None

    return pure_sig, norm_sig, final_sig, atm

def main():
    st.title("Fyers Option Chain Viewer — One Open Basket Per Signal")
    cid = st.text_input("Client ID")
    token = st.text_input("Access Token", type="password")
    sym = st.text_input("Symbol (e.g. NSE:NIFTY50-INDEX)", "NSE:NIFTY50-INDEX")
    if not (cid and token and sym):
        st.info("Enter Client ID, Access Token, and Symbol to proceed.")
        return

    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    enforce_rate_limit()
    st.session_state["option_chain_api_count"] += 1
    meta = fy.optionchain(data={"symbol": sym, "timestamp": ""})
    if meta.get("code") != 200:
        st.error("Failed to fetch expiryData: " + meta.get("message", ""))
        return

    expiry_data = meta["data"]["expiryData"]
    expiries = sorted([(e["date"], e.get("expiry")) for e in expiry_data],
                      key=lambda x: datetime.strptime(x[0], "%d-%m-%Y"))

    today = datetime.today().date()
    dates = [datetime.strptime(d, "%d-%m-%Y").date() for d, _ in expiries]
    curr_idx = next((i for i, d in enumerate(dates) if d >= today), 0)

    curr_date, curr_ts = expiries[curr_idx]

    ist = pytz.timezone("Asia/Kolkata")
    st.write(f"Data updated at IST: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")

    ltp = get_underlying_ltp(cid, token, sym)
    if ltp is not None:
        st.success(f"Underlying LTP: {ltp}")

    curr_chain = get_option_chain_data(cid, token, sym, expiry_ts=curr_ts)
    if curr_chain is not None:
        pure, norm, final, atm = format_and_show(curr_chain, f"Current Expiry: {curr_date}", ltp, show_signals=True)

        atm_strike, atm_ce, atm_pe, half_ce, half_pe, options_df = get_atm_and_half_price_option(curr_chain, ltp)

        if final in ("BUY", "SELL"):
            if st.session_state.get("last_signal") == final and has_open_orders_for_last_signal(fy):
                st.info(f"A basket order for '{final}' is already open. Not placing a new one.")
            else:
                order_ids = handle_basket_orders_atomic(final, atm_strike, atm_ce, atm_pe, half_ce, half_pe, fy)
                st.session_state["last_signal"] = final
                st.session_state["last_signal_order_ids"] = order_ids
        elif final == "SIDEWAYS":
            handle_basket_orders_atomic(final, atm_strike, atm_ce, atm_pe, half_ce, half_pe, fy)
        else:
            st.info("No actionable signal or no ATM found for basket order.")

    oc = st.session_state["option_chain_api_count"]
    qc = st.session_state["quote_api_count"]
    st.write(f"Option Chain API calls: {oc}  •  Quote API calls: {qc}  •  Total: {oc+qc}")

    time.sleep(60)
    st.rerun()

if __name__ == "__main__":
    logging.basicConfig(filename="fyers_order_debug.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    main()