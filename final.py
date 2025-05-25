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


def compute_signals(merged_df, atm_strike):
    strikes = sorted(merged_df["Strike"].dropna().unique().tolist())
    try:
        idx = strikes.index(atm_strike)
    except ValueError:
        st.error(f"ATM strike {atm_strike} not in available strikes")
        return "SIDEWAYS", "SIDEWAYS", "SIDEWAYS"

    lower_strikes = strikes[max(0, idx-5):idx]
    upper_strikes = strikes[idx+1:idx+6]

    ce_itm = merged_df[merged_df["Strike"].isin(lower_strikes)].copy()
    pe_itm = merged_df[merged_df["Strike"].isin(upper_strikes)].copy()

    ce_ltpch_sum = ce_itm["CE_LTPCh"].sum()
    pe_ltpch_sum = pe_itm["PE_LTPCh"].sum()
    st.write(f"(5×ITM) CE ΔLTP sum: {ce_ltpch_sum}, PE ΔLTP sum: {pe_ltpch_sum}")
    if ce_ltpch_sum > pe_ltpch_sum and ce_ltpch_sum > 0:
        pure_price_signal = "BUY"
    elif pe_ltpch_sum > ce_ltpch_sum and pe_ltpch_sum > 0:
        pure_price_signal = "SELL"
    else:
        pure_price_signal = "SIDEWAYS"
    st.write(f"Pure price signal: {pure_price_signal}")

    def normalize(df, price_col, vol_col):
        total_vol = df[vol_col].sum()
        return (df[price_col] * df[vol_col]).sum() / total_vol if total_vol else 0

    ce_bid_sum = normalize(ce_itm, "CE_bid", "CE_Vol")
    pe_bid_sum = normalize(pe_itm, "PE_bid", "PE_Vol")
    ce_ask_sum = normalize(ce_itm, "CE_ask", "CE_Vol")
    pe_ask_sum = normalize(pe_itm, "PE_ask", "PE_Vol")

    st.write(f"(5×ITM) CE bid avg: {ce_bid_sum:.2f}, PE bid avg: {pe_bid_sum:.2f}")
    st.write(f"(5×ITM) CE ask avg: {ce_ask_sum:.2f}, PE ask avg: {pe_ask_sum:.2f}")

    if ce_bid_sum > pe_bid_sum and ce_ask_sum > pe_ask_sum:
        normalized_signal = "BUY"
    elif pe_bid_sum > ce_bid_sum and pe_ask_sum > ce_ask_sum:
        normalized_signal = "SELL"
    else:
        normalized_signal = "SIDEWAYS"
    st.write(f"Normalized signal: {normalized_signal}")

    if pure_price_signal == normalized_signal and pure_price_signal in ("BUY", "SELL"):
        final_signal = pure_price_signal
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
        return None,None,None,None
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
        p,n,f=compute_signals(merged,atm)
        c1,c2,c3=st.columns(3)
        c1.metric("Pure-Price",p)
        c2.metric("Normalized",n)
        c3.metric("Final",f)
    styled=merged.style.apply(lambda row:["background:yellow" if row["Strike"]==atm else "" for _ in row],axis=1)
    st.dataframe(styled)
    st.caption(f"ATM: {atm} • LTP: {ltp}")
    return p,n,f,atm


def main():
    st.title("Fyers Algo")
    auto_trade=st.checkbox("Enable Auto Trade",value=False)
    cid=st.text_input("Client ID")
    token=st.text_input("Access Token",type="password")
    sym=st.text_input("Symbol (e.g. NSE:NIFTY50-INDEX)","NSE:NIFTY50-INDEX")
    if not(cid and token and sym): st.info("Enter credentials.");return
    fy=fyersModel.FyersModel(client_id=cid,token=token,is_async=False,log_path="")
    enforce_rate_limit();st.session_state["option_chain_api_count"]+=1
    meta=fy.optionchain(data={"symbol":sym,"timestamp":""})
    if meta.get("code")!=200:st.error("Failed to fetch expiryData");return
    exp=sorted([(e["date"],e.get("expiry")) for e in meta["data"]["expiryData"]],key=lambda x:datetime.strptime(x[0],"%d-%m-%Y"))
    today=datetime.today().date();dates=[datetime.strptime(d,"%d-%m-%Y").date() for d,_ in exp]
    idx=next((i for i,d in enumerate(dates) if d>=today),0);curr_date,curr_ts=exp[idx]
    ist=pytz.timezone("Asia/Kolkata");st.write(f"Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")
    ltp=get_underlying_ltp(cid,token,sym);
    if ltp:st.success(f"Underlying LTP: {ltp}")
    chain=get_option_chain_data(cid,token,sym,expiry_ts=curr_ts)
    if chain:
        pure,norm,final,atm=format_and_show(chain,f"Current Expiry: {curr_date}",ltp,show_signals=True)
        strike,atm_ce,atm_pe,half_ce,half_pe,_=get_atm_and_half_price_option(chain,ltp)
        if final in ("BUY","SELL"):
            if not auto_trade:
                st.info(f"Auto trade disabled; signal: {final}.")
            else:
                if st.session_state["last_signal"]==final and has_open_orders_for_last_signal(fy):
                    st.info(f"Basket for {final} already open.")
                else:
                    ids=handle_basket_orders_atomic(final,strike,atm_ce,atm_pe,half_ce,half_pe,fy)
                    st.session_state["last_signal"]=final;st.session_state["last_signal_order_ids"]=ids
        elif final=="SIDEWAYS":
            if not auto_trade: st.info("Auto trade disabled; sideways.")
            else: handle_basket_orders_atomic(final,strike,atm_ce,atm_pe,half_ce,half_pe,fy)
        else: st.info("No action.")
    oc, qc = st.session_state["option_chain_api_count"], st.session_state["quote_api_count"]
    #st.write(f"APIs: {oc}+{qc} = {oc+qc}")
    time.sleep(5); st.rerun()

if __name__=="__main__":
    logging.basicConfig(filename="fyers_order_debug.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    main()
