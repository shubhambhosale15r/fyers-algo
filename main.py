import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from fyers_apiv3 import fyersModel
import logging
import uuid

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
    itm_pe_oi = merged_df[merged_df["Strike"].isin(pe_itm_strikes)]["PE_OI"].sum()
    itm_ce_oi = merged_df[merged_df["Strike"].isin(ce_itm_strikes)]["CE_OI"].sum()
    otm_pe_oi = merged_df[merged_df["Strike"].isin(pe_otm_strikes)]["PE_OI"].sum()
    otm_ce_oi = merged_df[merged_df["Strike"].isin(ce_otm_strikes)]["CE_OI"].sum()

    # Avoid division by zero
    itm_pcr = itm_pe_oi / itm_ce_oi if itm_ce_oi > 0 else float('inf')
    otm_pcr = otm_pe_oi / otm_ce_oi if otm_ce_oi > 0 else float('inf')

    # Generate signals based on PCR values
    if itm_pcr < 0.8 and otm_pcr > 1.2:
        signal = "BUY"
    elif itm_pcr > 1.2 and otm_pcr < 0.8:
        signal = "SELL"
    else:
        signal = "SIDEWAYS"

    # Display PCR values
    st.write(f"ITM PCR: {itm_pcr:.4f} (PE OI: {itm_pe_oi}, CE OI: {itm_ce_oi})")
    st.write(f"OTM PCR: {otm_pcr:.4f} (PE OI: {otm_pe_oi}, CE OI: {otm_ce_oi})")
    
    return signal


def get_atm_and_half_price_option(chain, ltp):
    df = pd.DataFrame(chain)
    if df.empty:
        return None, None, None, None, None, None

    ce = df[df["option_type"] == "CE"].copy()
    pe = df[df["option_type"] == "PE"].copy()
    ce = ce[["symbol", "strike_price", "oi", "volume", "ltp", "ask", "bid", "ltpch", "ltpchp", "oich", "oichp", "prev_oi"]]
    pe = pe[["symbol", "strike_price", "oi", "volume", "ltp", "ask", "bid", "ltpch", "ltpchp", "oich", "oichp", "prev_oi"]]

    strikes = sorted(df["strike_price"].dropna().unique().tolist())
    atm_strike = min(strikes, key=lambda x: abs(x - ltp))

    atm_ce_row = ce[ce["strike_price"] == atm_strike].iloc[0].to_dict()
    atm_pe_row = pe[pe["strike_price"] == atm_strike].iloc[0].to_dict()

    half_pe_row = None
    pe_cand = pe[pe["strike_price"] < atm_strike].copy()
    if not pe_cand.empty:
        pe_cand["ltp_diff"] = (pe_cand["ltp"] - atm_pe_row["ltp"]/2).abs()
        half_pe_row = pe_cand.nsmallest(1, "ltp_diff").iloc[0].to_dict()

    half_ce_row = None
    ce_cand = ce[ce["strike_price"] > atm_strike].copy()
    if not ce_cand.empty:
        ce_cand["ltp_diff"] = (ce_cand["ltp"] - atm_ce_row["ltp"]/2).abs()
        half_ce_row = ce_cand.nsmallest(1, "ltp_diff").iloc[0].to_dict()

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


def handle_paper_trade(signal, atm_strike, atm_ce, atm_pe, half_ce, half_pe):
    orders = []
    lot = 75
    timestamp = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
    
    if signal == "BUY":
        # Buy half_pe, Sell atm_pe
        buy_symbol = half_pe["symbol"]
        sell_symbol = atm_pe["symbol"]
        orders.append({"symbol": buy_symbol, "qty": lot, "side": 1, "action": "BUY"})
        orders.append({"symbol": sell_symbol, "qty": lot, "side": -1, "action": "SELL"})
        st.write(f"[PAPER] BUY {buy_symbol} & SELL {sell_symbol}")
        
    elif signal == "SELL":
        # Buy half_ce, Sell atm_ce
        buy_symbol = half_ce["symbol"]
        sell_symbol = atm_ce["symbol"]
        orders.append({"symbol": buy_symbol, "qty": lot, "side": 1, "action": "BUY"})
        orders.append({"symbol": sell_symbol, "qty": lot, "side": -1, "action": "SELL"})
        st.write(f"[PAPER] BUY {buy_symbol} & SELL {sell_symbol}")
        
    elif signal == "SIDEWAYS":
        # Close all positions
        if st.session_state.paper_positions:
            for symbol, position in st.session_state.paper_positions.items():
                # Close with opposite action
                close_action = "SELL" if position["action"] == "BUY" else "BUY"
                close_side = -1 if position["side"] == 1 else 1
                orders.append({
                    "symbol": symbol,
                    "qty": position["qty"],
                    "side": close_side,
                    "action": close_action
                })
            st.write("[PAPER] Exiting all positions...")
        else:
            st.write("[PAPER] No positions to exit")
            return []
    
    # Execute paper trades
    placed_ids = []
    for order in orders:
        # Get current LTP for the symbol
        ltp = get_symbol_ltp(st.session_state.cid, st.session_state.token, order["symbol"])
        if ltp is None:
            st.error(f"Failed to get LTP for {order['symbol']}")
            continue
            
        # Create trade record
        trade_id = str(uuid.uuid4())[:8]
        trade = {
            "id": trade_id,
            "timestamp": timestamp,
            "symbol": order["symbol"],
            "qty": order["qty"],
            "action": order["action"],
            "price": ltp,
            "signal": signal,
            "type": "OPEN" if signal in ["BUY", "SELL"] else "CLOSE"
        }
        
        # Update positions and trade history
        update_paper_positions(trade)
        st.session_state.trade_history.append(trade)
        placed_ids.append(trade_id)
        
        st.write(f"[PAPER] {order['action']} {order['qty']} of {order['symbol']} at {ltp}")
    
    return placed_ids


def update_paper_positions(trade):
    symbol = trade["symbol"]
    
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
        else:
            st.warning(f"[PAPER] No position found to close for {symbol}")


def update_unrealized_pnl(cid, token):
    if not st.session_state.paper_positions:
        st.session_state.paper_pnl["unrealized"] = 0.0
        return
        
    total_unrealized = 0.0
    symbols = set(pos["symbol"] for pos in st.session_state.paper_positions.values())
    
    # Get current LTPs for all positions
    for symbol in symbols:
        ltp = get_symbol_ltp(cid, token, symbol)
        if ltp is None:
            continue
            
        # Update all positions for this symbol
        for key, position in st.session_state.paper_positions.items():
            if position["symbol"] == symbol:
                position["current_price"] = ltp
                
                # Calculate position PnL
                if position["action"] == "BUY":  # Long position
                    pnl = (ltp - position["entry_price"]) * position["qty"]
                else:  # Short position
                    pnl = (position["entry_price"] - ltp) * position["qty"]
                    
                total_unrealized += pnl
    
    st.session_state.paper_pnl["unrealized"] = total_unrealized


def handle_basket_orders_atomic(signal, atm_strike, atm_ce, atm_pe, half_ce, half_pe, fyers):
    orders = []
    lot=75
    if signal=="BUY":
        orders.append({"symbol": half_pe["symbol"], "qty": lot, "type":2, "side":1, "productType":"MARGIN","validity":"DAY"})
        orders.append({"symbol": atm_pe["symbol"],  "qty": lot, "type":2, "side":-1, "productType":"MARGIN","validity":"DAY"})
        st.write(f"BUY {half_pe['symbol']} & SELL {atm_pe['symbol']}")
    elif signal=="SELL":
        orders.append({"symbol": half_ce["symbol"], "qty": lot, "type":2, "side":1, "productType":"MARGIN","validity":"DAY"})
        orders.append({"symbol": atm_ce["symbol"],  "qty": lot, "type":2, "side":-1, "productType":"MARGIN","validity":"DAY"})
        st.write(f"BUY {half_ce['symbol']} & SELL {atm_ce['symbol']}")
    elif signal=="SIDEWAYS":
        try:
            pos= fyers.positions().get("netPositions", [])
            for p in pos:
                if p.get("segment","").upper()=="NFO" and p.get("qty",0)!=0:
                    side = -1 if p['qty']>0 else 1
                    orders.append({"symbol":p['symbol'],"qty":abs(int(p['qty'])),"type":2,"side":side,"productType":"MARGIN","validity":"DAY"})
            if orders: st.write("Exiting positions...")
        except Exception as e:
            st.error(f"Exit error: {e}")
            return []
        for o in orders:
            place_order_and_check(fyers,o)
        st.session_state["last_signal"] = None
        st.session_state["last_signal_order_ids"]=[]
        return []

    placed_ids=[]
    for o in orders:
        oid, status = place_order_and_check(fyers, o)
        if oid is None or status in [1,5]:
            for pid in placed_ids: cancel_order(fyers, pid)
            st.session_state["last_signal_order_ids"]=[]
            return []
        placed_ids.append(oid)

    st.success("Basket executed")
    return placed_ids


def format_and_show(chain, title, ltp, show_signals=False):
    df=pd.DataFrame(chain)
    if df.empty:
        st.info(f"No data for {title}")
        return None
    ce=df[df["option_type"]=="CE"]
    pe=df[df["option_type"]=="PE"]
    cols=["symbol","strike_price","oi","volume","ltp","ask","bid","ltpch","ltpchp","oich","oichp","prev_oi"]
    ce_df=ce[cols].rename(columns={"strike_price":"Strike","oi":"OI","volume":"Vol","ltpch":"LTPCh","ltpchp":"LTPChP","oich":"OICh","oichp":"OIChP","prev_oi":"PrevOI"})
    pe_df=pe[cols].rename(columns={"strike_price":"Strike","oi":"OI","volume":"Vol","ltpch":"LTPCh","ltpchp":"LTPChP","oich":"OICh","oichp":"OIChP","prev_oi":"PrevOI"})
    merged=pd.merge(ce_df.add_prefix("CE_"),pe_df.add_prefix("PE_"),left_on="CE_Strike",right_on="PE_Strike",how="outer").rename(columns={"CE_Strike":"Strike"}).drop("PE_Strike",axis=1)
    atm=None
    if ltp is not None:
        atm=min(merged["Strike"].dropna(),key=lambda x:abs(x-ltp))
    st.subheader(title)
    if show_signals and atm is not None:
        signal = compute_signals(merged, atm)
        st.metric("PCR Signal", signal)  # Show just the PCR signal
    styled=merged.style.apply(lambda row:["background:yellow" if row["Strike"]==atm else "" for _ in row],axis=1)
    st.dataframe(styled)
    st.caption(f"ATM: {atm} • LTP: {ltp}")
    return signal


def show_paper_trading_page(cid, token):
    st.subheader("📝 Paper Trading Dashboard")
    
    # Update unrealized PnL
    update_unrealized_pnl(cid, token)
    
    # Display PnL summary
    col1, col2 = st.columns(2)
    col1.metric("💰 Realized PnL", f"₹{st.session_state.paper_pnl['realized']:.2f}")
    col2.metric("📊 Unrealized PnL", f"₹{st.session_state.paper_pnl['unrealized']:.2f}")
    st.metric("💵 Total PnL", f"₹{st.session_state.paper_pnl['realized'] + st.session_state.paper_pnl['unrealized']:.2f}")
    
    # Current positions
    st.subheader("📊 Current Positions")
    if st.session_state.paper_positions:
        positions_list = []
        for symbol, position in st.session_state.paper_positions.items():
            # Calculate current PnL for each position
            if position["action"] == "BUY":
                pnl = (position["current_price"] - position["entry_price"]) * position["qty"]
            else:
                pnl = (position["entry_price"] - position["current_price"]) * position["qty"]
                
            positions_list.append({
                "Symbol": position["symbol"],
                "Action": position["action"],
                "Qty": position["qty"],
                "Entry Price": position["entry_price"],
                "Current Price": position["current_price"],
                "PnL": pnl,
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
        st.success("Paper trading data reset!")


def main():
    st.title("Fyers Algo Trading")
    
    # Initialize session state
    if "cid" not in st.session_state:
        st.session_state.cid = ""
    if "token" not in st.session_state:
        st.session_state.token = ""
    
    # Settings sidebar
    with st.sidebar:
        st.subheader("⚙️ Trading Settings")
        auto_trade = st.checkbox("Enable Auto Trade", value=False)
        paper_trade = st.checkbox("📝 Enable Paper Trading", value=False)
        
        # Paper trading page navigation
        if paper_trade:
            if st.button("📊 Go to Paper Trading Dashboard"):
                st.session_state.page = "paper_trading"
            if st.button("🔙 Back to Trading"):
                st.session_state.page = "trading"
        else:
            st.session_state.page = "trading"
        
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
    
    # Paper trading page
    if st.session_state.page == "paper_trading" and paper_trade:
        show_paper_trading_page(cid, token)
        return
    
    # Main trading page
    st.subheader("📈 Live Trading")
    if not(cid and token and sym): 
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
        signal = format_and_show(chain, f"Current Expiry: {curr_date}", ltp, show_signals=True)
        strike, atm_ce, atm_pe, half_ce, half_pe, _ = get_atm_and_half_price_option(chain, ltp)
        
        # Check if we have valid option data
        if not half_pe or not atm_pe or not half_ce or not atm_ce:
            st.warning("Couldn't find required option strikes. Unable to place trades.")
        elif signal in ("BUY", "SELL"):
            if not auto_trade:
                st.info(f"Auto trade disabled; signal: {signal}.")
            else:
                if paper_trade:
                    # Handle paper trading
                    if st.session_state["last_signal"] == signal and st.session_state.paper_positions:
                        st.info(f"Paper basket for {signal} already open.")
                    else:
                        ids = handle_paper_trade(signal, strike, atm_ce, atm_pe, half_ce, half_pe)
                        st.session_state["last_signal"] = signal
                        st.session_state["last_signal_order_ids"] = ids
                else:
                    # Handle real trading
                    if st.session_state["last_signal"] == signal and has_open_orders_for_last_signal(fy):
                        st.info(f"Basket for {signal} already open.")
                    else:
                        ids = handle_basket_orders_atomic(signal, strike, atm_ce, atm_pe, half_ce, half_pe, fy)
                        st.session_state["last_signal"] = signal
                        st.session_state["last_signal_order_ids"] = ids
        elif signal == "SIDEWAYS":
            if not auto_trade: 
                st.info("Auto trade disabled; sideways.")
            else: 
                if paper_trade:
                    handle_paper_trade(signal, strike, atm_ce, atm_pe, half_ce, half_pe)
                else:
                    handle_basket_orders_atomic(signal, strike, atm_ce, atm_pe, half_ce, half_pe, fy)
        else: 
            st.info("No action.")
    
    time.sleep(180)
    st.rerun()

if __name__ == "__main__":
    logging.basicConfig(filename="trading_debug.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    main()
