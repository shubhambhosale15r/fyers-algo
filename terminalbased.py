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
# def load_credentials(session_state):
#     session_state.cid = os.getenv("FYERS_CID", session_state.cid)
#     session_state.token = os.getenv("FYERS_TOKEN", session_state.token)
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
        print(f"❌ {error_msg}")
        return None
    ltp = resp["d"][0]["v"]["lp"]
    print(f"✅ [DEBUG] Symbol LTP: {ltp}")
    return ltp

# Signal computation with detailed debug logs

# def compute_signals(
#     merged_df,
#     atm_strike,
#     ltp,
#     min_range_pct: float = 1.0,
#     tolerance_pct: float = 1.0,
#     fyers_api=None,
# ):
#     """
#     Compute intraday trading signals using ITM options only.

#     Logic:
#       1. Define price range around ATM strike.
#       2. Fetch Fyers quotes for ITM strikes.
#       3. Update LTP, ATP, highs in dataframe.
#       4. Compute moneyflow, direction ratio, and strength.
#       5. Generate trading signal (BUY / SELL / SIDEWAYS).

#     Returns:
#         str: Signal -> "BUY", "SELL", or "SIDEWAYS"
#     """
#     print(f"🔍 [DEBUG] Computing signals for ATM strike: {atm_strike}, LTP: {ltp}")

#     try:
#         # ----------------------------
#         # Helper
#         # ----------------------------
#         def safe_val(series, default=0.0):
#             """Safe extraction of first value from Series."""
#             return float(series.iloc[0]) if not series.empty and not pd.isna(series.iloc[0]) else float(default)

#         # ----------------------------
#         # 1. Price range around ATM
#         # ----------------------------
#         atm_row = merged_df[merged_df["Strike"] == atm_strike]
#         atm_ce_price = safe_val(atm_row["CE_ltp"])
#         atm_pe_price = safe_val(atm_row["PE_ltp"])
#         atm_ce_volume = safe_val(atm_row["CE_Vol"])
#         atm_pe_volume = safe_val(atm_row["PE_Vol"])

#         atm_ce_moneyflow = atm_ce_price * atm_ce_volume
#         atm_pe_moneyflow = atm_pe_price * atm_pe_volume
#         price_range = atm_ce_price + atm_pe_price

#         min_range_abs = float(ltp) * (min_range_pct / 100.0)

#         if price_range >= min_range_abs:
#             lower_bound = atm_strike - price_range
#             upper_bound = atm_strike + price_range
#             range_type = f"ATM+CE Range (±{price_range:.2f} points)"
#         else:
#             lower_bound = ltp * (1 - min_range_pct / 100.0)
#             upper_bound = ltp * (1 + min_range_pct / 100.0)
#             range_type = f"{min_range_pct}% Underlying Range"

#         def round_to_nearest_strike(value, strikes):
#             """Round to the nearest available strike price."""
#             if not strikes:
#                 return value
#             return min(strikes, key=lambda x: abs(x - value))

#         strikes = sorted(merged_df["Strike"].dropna().unique())
#         lower_bound = round_to_nearest_strike(lower_bound, strikes)
#         upper_bound = round_to_nearest_strike(upper_bound, strikes)

#         range_strikes = [s for s in strikes if lower_bound <= s <= upper_bound]
#         if not range_strikes:
#             print(f"⚠️ [DEBUG] No strikes in range {lower_bound:.2f}-{upper_bound:.2f}")
#             return "SIDEWAYS"

#         # Work on a copy to avoid polluting original
#         range_df = merged_df[merged_df["Strike"].isin(range_strikes)].copy()

#         # ----------------------------
#         # 2. Update with Fyers quotes
#         # ----------------------------
#         if fyers_api and {"CE_symbol", "PE_symbol"} <= set(range_df.columns):
#             try:
#                 symbols = list(set(range_df["CE_symbol"].dropna().tolist() + range_df["PE_symbol"].dropna().tolist()))
#                 symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
#                 quotes = {}

#                 # Batch fetch (50 at a time)
#                 for i in range(0, len(symbols), 50):
#                     batch = symbols[i:i + 50]
#                     response = fyers_api.quotes({"symbols": ",".join(batch)})
#                     if response.get("code") == 200:
#                         quotes.update({item["n"]: item["v"] for item in response["d"] if item["s"] == "ok"})

#                 # Default flags
#                 range_df["CE_vs_atp"] = "BELOW"
#                 range_df["PE_vs_atp"] = "BELOW"
#                 range_df["ce_strength"] = "LOW"
#                 range_df["pe_strength"] = "LOW"

#                 # Iterate rows
#                 for idx, row in range_df.iterrows():
#                     # CE updates
#                     if row["CE_symbol"] in quotes:
#                         ce_data = quotes[row["CE_symbol"]]
#                         ce_ltp = float(ce_data.get("lp", 0) or 0)
#                         atp = float(ce_data.get("atp", 0) or 0)
#                         ce_high = float(ce_data.get("high_price", 0) or 0)

#                         if atp > 0:
#                             range_df.at[idx, "CE_vs_atp"] = "ABOVE" if ce_ltp > atp else "BELOW"
#                         if ce_high > 0 and ce_ltp > 0:
#                             range_df.at[idx, "CE_high"] = ce_high
#                             range_df.at[idx, "CE_high_diff"] = ce_high - ce_ltp
#                         range_df.at[idx, "CE_ltp"] = ce_ltp

#                     # PE updates
#                     if row["PE_symbol"] in quotes:
#                         pe_data = quotes[row["PE_symbol"]]
#                         pe_ltp = float(pe_data.get("lp", 0) or 0)
#                         atp = float(pe_data.get("atp", 0) or 0)
#                         pe_high = float(pe_data.get("high_price", 0) or 0)

#                         if atp > 0:
#                             range_df.at[idx, "PE_vs_atp"] = "ABOVE" if pe_ltp > atp else "BELOW"
#                         if pe_high > 0 and pe_ltp > 0:
#                             range_df.at[idx, "PE_high"] = pe_high
#                             range_df.at[idx, "PE_high_diff"] = pe_high - pe_ltp
#                         range_df.at[idx, "PE_ltp"] = pe_ltp

#                 # Recompute moneyflows with safe dtypes
#                 range_df["CE_moneyflow"] = range_df["CE_ltp"].astype(float) * range_df["CE_Vol"].astype(float)
#                 range_df["PE_moneyflow"] = range_df["PE_ltp"].astype(float) * range_df["PE_Vol"].astype(float)

#             except Exception as e:
#                 logging.warning(f"Fyers quotes error: {str(e)}")
#                 print(f"⚠️ [DEBUG] Fyers quotes error: {str(e)}")

#         # ----------------------------
#         # 3. ITM Strikes selection
#         # ----------------------------
#         atm_idx = range_strikes.index(atm_strike) if atm_strike in range_strikes else len(range_strikes) // 2
#         num_strikes = min(len(range_strikes[:atm_idx]), len(range_strikes[atm_idx + 1:]))
#         itm_ce_strikes = range_strikes[atm_idx - num_strikes:atm_idx]
#         itm_pe_strikes = range_strikes[atm_idx + 1:atm_idx + 1 + num_strikes]

#         # ----------------------------
#         # 4. Money flow & Direction
#         # ----------------------------
#         ce_net = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_moneyflow"].sum()
#         pe_net = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_moneyflow"].sum()

#         net_flow = (ce_net - pe_net) / (ce_net + pe_net) if (ce_net + pe_net) != 0 else 0.0

#         ce_above_atp = range_df[(range_df["Strike"].isin(itm_ce_strikes)) & (range_df["CE_vs_atp"] == "ABOVE")].shape[0]
#         pe_above_atp = range_df[(range_df["Strike"].isin(itm_pe_strikes)) & (range_df["PE_vs_atp"] == "ABOVE")].shape[0]
#         direction_ratio = (ce_above_atp - pe_above_atp) / max(len(itm_ce_strikes) + len(itm_pe_strikes), 1)

#         # ----------------------------
#         # 5. Strength metric
#         # ----------------------------
#         range_df["CE_high_pct"] = np.where(
#             range_df["CE_high"] > 0,
#             range_df["CE_high_diff"] / range_df["CE_high"],
#             0,
#         )
#         range_df["PE_high_pct"] = np.where(
#             range_df["PE_high"] > 0,
#             range_df["PE_high_diff"] / range_df["PE_high"],
#             0,
#         )

#         ce_high_pct_avg = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_high_pct"].mean()
#         pe_high_pct_avg = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_high_pct"].mean()

#         strength = (ce_high_pct_avg - pe_high_pct_avg) / max(ce_high_pct_avg + pe_high_pct_avg, 1e-6)

#         # ----------------------------
#         # 6. Signal generation
#         # ----------------------------
#         signal = "SIDEWAYS"
#         if direction_ratio >= 0.5 and strength < 0 and net_flow > 0:      #0.2 mean 50% more    0.3 mean 100% more
#             print("📈 [DEBUG] Market is in BUY entry found entering the trade")
#             signal = "BUY"
#         elif direction_ratio <= -0.5 and strength > 0 and net_flow < 0:
#             print("📉 [DEBUG] Market is in SELL entry found entering the trade")
#             signal = "SELL"

#         # ----------------------------
#         # 7. Debug Summary
#         # ----------------------------
#         print(f"\n📊 [DEBUG] {range_type}")
#         print(f"📊 [DEBUG] Range: {lower_bound:.2f} to {upper_bound:.2f}")
#         print(f"📊 [DEBUG] ITM CE Strikes: {list(itm_ce_strikes)}")
#         print(f"📊 [DEBUG] ITM PE Strikes: {list(itm_pe_strikes)}")
#         print("📊 [DEBUG] ce_net", ce_net)
#         print("📊 [DEBUG] pe_net", pe_net)
#         print(f"📊 [DEBUG] Net Money Flow: {net_flow:.2f}")
#         print(
#             f"📊 [DEBUG] Direction Ratio: {direction_ratio:.2f} "
#             f"| CE Above ATP: {ce_above_atp} | PE Above ATP: {pe_above_atp} "
#             f"| Total Strikes: {len(itm_ce_strikes) + len(itm_pe_strikes)}"
#         )
#         print(
#             f"📊 [DEBUG] Avg CE % below High: {ce_high_pct_avg:.2%} | "
#             f"Avg PE % below High: {pe_high_pct_avg:.2%} | Strength: {strength:.2f}"
#         )

#         return signal

#     except Exception as e:
#         logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
#         print(f"❌ [DEBUG] Error in compute_signals: {str(e)}")
#         return "SIDEWAYS"

# ----------------------------
# Global rolling signal buffer
# ----------------------------
# signal_history = []
# MAX_HISTORY = 3  # Last 5 signals
# MAJORITY_THRESHOLD = 3  # Majority count to consider

# def majority_signal(history, threshold=MAJORITY_THRESHOLD):
#     """
#     Determine majority signal from the history.
#     Returns the majority signal if count >= threshold, else SIDEWAYS.
#     """
#     if not history:
#         return "SIDEWAYS"
#     counter = Counter(history)
#     most_common_signal, count = counter.most_common(1)[0]
#     if count >= threshold:
#         return most_common_signal
#     return "SIDEWAYS"


# def compute_signals(
#     merged_df,
#     atm_strike,
#     ltp,
#     min_range_pct: float = 1.0,
#     tolerance_pct: float = 1.0,
#     fyers_api=None,
# ):
#     """
#     Compute intraday trading signals using ITM options only.
#     Returns: "BUY", "SELL", or "SIDEWAYS" based on rolling majority.
#     """
#     print(f"🔍 [DEBUG] Computing signals for ATM strike: {atm_strike}, LTP: {ltp}")

#     try:
#         # ----------------------------
#         # Helper
#         # ----------------------------
#         def safe_val(series, default=0.0):
#             return float(series.iloc[0]) if not series.empty and not pd.isna(series.iloc[0]) else float(default)

#         # ----------------------------
#         # 1. Price range around ATM
#         # ----------------------------
#         atm_row = merged_df[merged_df["Strike"] == atm_strike]
#         atm_ce_price = safe_val(atm_row["CE_ltp"])
#         atm_pe_price = safe_val(atm_row["PE_ltp"])
#         atm_ce_volume = safe_val(atm_row["CE_Vol"])
#         atm_pe_volume = safe_val(atm_row["PE_Vol"])

#         atm_ce_moneyflow = atm_ce_price * atm_ce_volume
#         atm_pe_moneyflow = atm_pe_price * atm_pe_volume
#         price_range = atm_ce_price + atm_pe_price
#         min_range_abs = float(ltp) * (min_range_pct / 100.0)

#         if price_range >= min_range_abs:
#             lower_bound = atm_strike - price_range
#             upper_bound = atm_strike + price_range
#             range_type = f"ATM+CE Range (±{price_range:.2f} points)"
#         else:
#             lower_bound = ltp * (1 - min_range_pct / 100.0)
#             upper_bound = ltp * (1 + min_range_pct / 100.0)
#             range_type = f"{min_range_pct}% Underlying Range"

#         def round_to_nearest_strike(value, strikes):
#             if not strikes:
#                 return value
#             return min(strikes, key=lambda x: abs(x - value))

#         strikes = sorted(merged_df["Strike"].dropna().unique())
#         lower_bound = round_to_nearest_strike(lower_bound, strikes)
#         upper_bound = round_to_nearest_strike(upper_bound, strikes)

#         range_strikes = [s for s in strikes if lower_bound <= s <= upper_bound]
#         if not range_strikes:
#             print(f"⚠️ [DEBUG] No strikes in range {lower_bound:.2f}-{upper_bound:.2f}")
#             return "SIDEWAYS"

#         range_df = merged_df[merged_df["Strike"].isin(range_strikes)].copy()

#         # ----------------------------
#         # 2. Update with Fyers quotes
#         # ----------------------------
#         if fyers_api and {"CE_symbol", "PE_symbol"} <= set(range_df.columns):
#             try:
#                 symbols = list(set(range_df["CE_symbol"].dropna().tolist() + range_df["PE_symbol"].dropna().tolist()))
#                 symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
#                 quotes = {}

#                 for i in range(0, len(symbols), 50):
#                     batch = symbols[i:i + 50]
#                     response = fyers_api.quotes({"symbols": ",".join(batch)})
#                     if response.get("code") == 200:
#                         quotes.update({item["n"]: item["v"] for item in response["d"] if item["s"] == "ok"})

#                 range_df["CE_vs_atp"] = "BELOW"
#                 range_df["PE_vs_atp"] = "BELOW"

#                 for idx, row in range_df.iterrows():
#                     if row["CE_symbol"] in quotes:
#                         ce_data = quotes[row["CE_symbol"]]
#                         ce_ltp = float(ce_data.get("lp", 0) or 0)
#                         atp = float(ce_data.get("atp", 0) or 0)
#                         ce_high = float(ce_data.get("high_price", 0) or 0)
#                         if atp > 0:
#                             range_df.at[idx, "CE_vs_atp"] = "ABOVE" if ce_ltp > atp else "BELOW"
#                         if ce_high > 0 and ce_ltp > 0:
#                             range_df.at[idx, "CE_high"] = ce_high
#                             range_df.at[idx, "CE_high_diff"] = ce_high - ce_ltp
#                         range_df.at[idx, "CE_ltp"] = ce_ltp

#                     if row["PE_symbol"] in quotes:
#                         pe_data = quotes[row["PE_symbol"]]
#                         pe_ltp = float(pe_data.get("lp", 0) or 0)
#                         atp = float(pe_data.get("atp", 0) or 0)
#                         pe_high = float(pe_data.get("high_price", 0) or 0)
#                         if atp > 0:
#                             range_df.at[idx, "PE_vs_atp"] = "ABOVE" if pe_ltp > atp else "BELOW"
#                         if pe_high > 0 and pe_ltp > 0:
#                             range_df.at[idx, "PE_high"] = pe_high
#                             range_df.at[idx, "PE_high_diff"] = pe_high - pe_ltp
#                         range_df.at[idx, "PE_ltp"] = pe_ltp

#                 range_df["CE_moneyflow"] = range_df["CE_ltp"].astype(float) * range_df["CE_Vol"].astype(float)
#                 range_df["PE_moneyflow"] = range_df["PE_ltp"].astype(float) * range_df["PE_Vol"].astype(float)

#             except Exception as e:
#                 logging.warning(f"Fyers quotes error: {str(e)}")
#                 print(f"⚠️ [DEBUG] Fyers quotes error: {str(e)}")

#         # ----------------------------
#         # 3. ITM Strikes selection
#         # ----------------------------
#         atm_idx = range_strikes.index(atm_strike) if atm_strike in range_strikes else len(range_strikes) // 2
#         num_strikes = min(len(range_strikes[:atm_idx]), len(range_strikes[atm_idx + 1:]))
#         itm_ce_strikes = range_strikes[atm_idx - num_strikes:atm_idx]
#         itm_pe_strikes = range_strikes[atm_idx + 1:atm_idx + 1 + num_strikes]

#         # ----------------------------
#         # 4. Money flow & Direction
#         # ----------------------------
#         ce_net = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_moneyflow"].sum()
#         pe_net = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_moneyflow"].sum()
#         net_flow = (ce_net - pe_net) / (ce_net + pe_net) if (ce_net + pe_net) != 0 else 0.0

#         ce_above_atp = range_df[(range_df["Strike"].isin(itm_ce_strikes)) & (range_df["CE_vs_atp"] == "ABOVE")].shape[0]
#         pe_above_atp = range_df[(range_df["Strike"].isin(itm_pe_strikes)) & (range_df["PE_vs_atp"] == "ABOVE")].shape[0]
#         direction_ratio = (ce_above_atp - pe_above_atp) / max(len(itm_ce_strikes) + len(itm_pe_strikes), 1)

#         range_df["CE_high_pct"] = np.where(range_df["CE_high"] > 0, range_df["CE_high_diff"] / range_df["CE_high"], 0)
#         range_df["PE_high_pct"] = np.where(range_df["PE_high"] > 0, range_df["PE_high_diff"] / range_df["PE_high"], 0)
#         ce_high_pct_avg = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_high_pct"].mean()
#         pe_high_pct_avg = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_high_pct"].mean()
#         strength = (ce_high_pct_avg - pe_high_pct_avg) / max(ce_high_pct_avg + pe_high_pct_avg, 1e-6)

#         # ----------------------------
#         # 5. Signal generation (raw)
#         # ----------------------------
#         raw_signal = "SIDEWAYS"
#         if direction_ratio >= 0.5 and strength <= -0.33 and net_flow >= 0.33:
#             raw_signal = "BUY"
#         elif direction_ratio <= -0.5 and strength >= 0.33 and net_flow < -0.33:
#             raw_signal = "SELL"

#         # ----------------------------
#         # 6. Update rolling buffer and get majority
#         # ----------------------------
#         global signal_history
#         signal_history.append(raw_signal)
#         if len(signal_history) > MAX_HISTORY:
#             signal_history.pop(0)
#         print(f"📊 [DEBUG] Signal History: {signal_history}")

#         final_signal = majority_signal(signal_history)

#         # ----------------------------
#         # 7. Debug Summary
#         # ----------------------------
#         print(f"\n📊 [DEBUG] {range_type}")
#         print(f"📊 [DEBUG] ITM CE Strikes: {list(itm_ce_strikes)}")
#         print(f"📊 [DEBUG] ITM PE Strikes: {list(itm_pe_strikes)}")
#         print(f"📊 [DEBUG] ce_net: {ce_net},📊 [DEBUG] pe_net: {pe_net}")
#         print(f"📊 Net Flow: {net_flow:.2f}, Direction Ratio: {direction_ratio:.2f}, Strength: {strength:.2f}")
#         print(f"📊 Raw Signal: {raw_signal}, Majority Signal: {final_signal}")
        
#         return final_signal

#     except Exception as e:
#         logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
#         print(f"❌ [DEBUG] Error in compute_signals: {str(e)}")
#         return "SIDEWAYS"


# ----------------------------
# Global rolling signal buffer
# ----------------------------
signal_history = []

MAX_HISTORY = 3  # Last 3 signals
MAJORITY_THRESHOLD = 3  # Majority count to consider

def majority_signal(history, threshold=MAJORITY_THRESHOLD):
    """
    Determine majority signal from the history.
    Returns the majority signal if count >= threshold, else SIDEWAYS.
    """
    if not history:
        return "SIDEWAYS"
    counter = Counter(history)
    most_common_signal, count = counter.most_common(1)[0]
    if count >= threshold:
        return most_common_signal
    return "SIDEWAYS"

def compute_signals(
    merged_df,
    atm_strike,
    ltp,
    min_range_pct: float = 1.0,
    tolerance_pct: float = 1.0,
    fyers_api=None,
):
    """
    Compute intraday trading signals using ITM options only.
    Returns: "BUY", "SELL", or "SIDEWAYS" based on rolling majority.
    """
    print(f"🔍 [DEBUG] Computing signals for ATM strike: {atm_strike}, LTP: {ltp}")

    try:
        # ----------------------------
        # Helper
        # ----------------------------
        def safe_val(series, default=0.0):
            return float(series.iloc[0]) if not series.empty and not pd.isna(series.iloc[0]) else float(default)

        # ----------------------------
        # 1. Price range around ATM
        # ----------------------------
        atm_row = merged_df[merged_df["Strike"] == atm_strike]
        atm_ce_price = safe_val(atm_row["CE_ltp"])
        atm_pe_price = safe_val(atm_row["PE_ltp"])

        price_range = atm_ce_price + atm_pe_price
        min_range_abs = float(ltp) * (min_range_pct / 100.0)

        if price_range >= min_range_abs:
            lower_bound = atm_strike - price_range
            upper_bound = atm_strike + price_range
            range_type = f"ATM+CE Range (±{price_range:.2f} points)"
        else:
            lower_bound = ltp * (1 - min_range_pct / 100.0)
            upper_bound = ltp * (1 + min_range_pct / 100.0)
            range_type = f"{min_range_pct}% Underlying Range"

        strikes = sorted(merged_df["Strike"].dropna().unique())

        def round_to_nearest_strike(value, strikes):
            if not strikes:
                return value
            return min(strikes, key=lambda x: abs(x - value))

        lower_bound = round_to_nearest_strike(lower_bound, strikes)
        upper_bound = round_to_nearest_strike(upper_bound, strikes)

        range_strikes = [s for s in strikes if lower_bound <= s <= upper_bound]
        if not range_strikes:
            print(f"⚠️ [DEBUG] No strikes in range {lower_bound:.2f}-{upper_bound:.2f}")
            return "SIDEWAYS"

        range_df = merged_df[merged_df["Strike"].isin(range_strikes)].copy()

        # ----------------------------
        # 2. Update with Fyers quotes
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

                range_df["CE_moneyflow"] = range_df["CE_ltp"].astype(float) * range_df["CE_Vol"].astype(float)
                range_df["PE_moneyflow"] = range_df["PE_ltp"].astype(float) * range_df["PE_Vol"].astype(float)

            except Exception as e:
                logging.warning(f"Fyers quotes error: {str(e)}")
                print(f"⚠️ [DEBUG] Fyers quotes error: {str(e)}")

        # ----------------------------
        # 3. ITM Strikes selection
        # ----------------------------
        atm_idx = range_strikes.index(atm_strike) if atm_strike in range_strikes else len(range_strikes) // 2
        num_strikes = min(len(range_strikes[:atm_idx]), len(range_strikes[atm_idx + 1:]))
        itm_ce_strikes = range_strikes[atm_idx - num_strikes:atm_idx]
        itm_pe_strikes = range_strikes[atm_idx + 1:atm_idx + 1 + num_strikes]

        # ----------------------------
        # 4. Money flow & Direction
        # ----------------------------
        ce_net = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_moneyflow"].sum()
        pe_net = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_moneyflow"].sum()
        net_flow = (ce_net - pe_net) / (ce_net + pe_net) if (ce_net + pe_net) != 0 else 0.0

        ce_above_atp = range_df[(range_df["Strike"].isin(itm_ce_strikes)) & (range_df["CE_vs_atp"] == "ABOVE")].shape[0]
        pe_above_atp = range_df[(range_df["Strike"].isin(itm_pe_strikes)) & (range_df["PE_vs_atp"] == "ABOVE")].shape[0]
        direction_ratio = (ce_above_atp - pe_above_atp) / max(len(itm_ce_strikes) + len(itm_pe_strikes), 1)

        # ----------------------------
        # 5. Range placement logic
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

        ce_placement_avg = range_df[range_df["Strike"].isin(itm_ce_strikes)]["CE_range_placement"].mean()
        pe_placement_avg = range_df[range_df["Strike"].isin(itm_pe_strikes)]["PE_range_placement"].mean()

                # ----------------------------
        # 4c. Placement thresholds
        # ----------------------------
        CE_HIGH_THRESHOLD = 0.6
        CE_LOW_THRESHOLD = 0.4
        PE_HIGH_THRESHOLD = 0.6
        PE_LOW_THRESHOLD = 0.4

        # ----------------------------
        # 6. Signal generation (raw)
        # ----------------------------
        raw_signal = "SIDEWAYS"
        if direction_ratio >= 0.5   and ce_placement_avg > CE_HIGH_THRESHOLD and pe_placement_avg < PE_LOW_THRESHOLD: #and net_flow > 0.33
            raw_signal = "BUY"
        elif direction_ratio <= -0.5 and  pe_placement_avg > PE_HIGH_THRESHOLD and ce_placement_avg < CE_LOW_THRESHOLD: #and net_flow < -0.33 
            raw_signal = "SELL"

        # ----------------------------
        # 7. Update rolling buffer and get majority
        # ----------------------------
        global signal_history
        signal_history.append(raw_signal)
        
        if len(signal_history) > MAX_HISTORY:
            signal_history.pop(0)
        print(f"📊 [DEBUG] Signal History: {signal_history}")
        final_signal = majority_signal(signal_history)

        # ----------------------------
        # 8. Debug Summary
        # ----------------------------
        print(f"\n📊 [DEBUG] {range_type}")
        print(f"📊 [DEBUG] ITM CE Strikes: {list(itm_ce_strikes)}")
        print(f"📊 [DEBUG] ITM PE Strikes: {list(itm_pe_strikes)}")
        print(f"📊 [DEBUG] ce_net: {ce_net}, pe_net: {pe_net}")
        print(f"📊 [DEBUG] CE Above ATP: {ce_above_atp}, PE Above ATP: {pe_above_atp}, total: {len(itm_ce_strikes) + len(itm_pe_strikes)}")
        print(f"📊 [DEBUG] Net Flow: {net_flow:.2f}, Direction Ratio: {direction_ratio}, CE Placement Strength: {ce_placement_avg}, PE Placement Strength: {pe_placement_avg}")
        print(f"📊 [DEBUG] Raw Signal: {raw_signal}, Majority Signal: {final_signal}")

        return final_signal

    except Exception as e:
        logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
        print(f"❌ [DEBUG] Error in compute_signals: {str(e)}")
        return "SIDEWAYS"
# sell itm and buy atm
# Optimized strike selection
def select_strikes_atm_half_price(chain, atm_strike):
    print(f"🔍 [DEBUG] Selecting strikes for ATM Buy / ITM Sell: {atm_strike}")
    try:
        df = pd.DataFrame(chain)
        if df.empty or atm_strike is None:
            print("⚠️ [DEBUG] No data or ATM strike for selection")
            return None, None, None, None

        ce = df[df["option_type"] == "CE"].copy()
        pe = df[df["option_type"] == "PE"].copy()

        # ATM Buy legs
        atm_pe_row = pe[pe["strike_price"] == atm_strike]
        atm_ce_row = ce[ce["strike_price"] == atm_strike]

        buy_pe_row = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
        buy_ce_row = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None

        # ITM Sell legs = approx 2 × ATM price
        sell_pe_row, sell_ce_row = None, None

        if buy_pe_row and "ltp" in buy_pe_row and buy_pe_row["ltp"] > 0:
            target = buy_pe_row["ltp"] * 2
            pe_candidates = pe[pe["strike_price"] != atm_strike].copy()
            pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
            if not pe_candidates.empty:
                sell_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        if buy_ce_row and "ltp" in buy_ce_row and buy_ce_row["ltp"] > 0:
            target = buy_ce_row["ltp"] * 2
            ce_candidates = ce[ce["strike_price"] != atm_strike].copy()
            ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target).abs()
            if not ce_candidates.empty:
                sell_ce_row = ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        print(f"✅ [DEBUG] Strike selection complete:")
        print(f"   - BUY PE: {buy_pe_row['symbol'] if buy_pe_row else 'None'}")
        print(f"   - SELL PE: {sell_pe_row['symbol'] if sell_pe_row else 'None'}")
        print(f"   - BUY CE: {buy_ce_row['symbol'] if buy_ce_row else 'None'}")
        print(f"   - SELL CE: {sell_ce_row['symbol'] if sell_ce_row else 'None'}")

        return buy_pe_row, sell_pe_row, buy_ce_row, sell_ce_row

    except Exception as e:
        logging.error(f"Error in select_strikes_buy_atm_sell_itm: {str(e)}", exc_info=True)
        print(f"❌ [DEBUG] Error in select_strikes_buy_atm_sell_itm: {str(e)}")
        return None, None, None, None
#ATM sell and otm hedge
# # Optimized strike selection
# def select_strikes_atm_half_price(chain, atm_strike):
#     print(f"🔍 [DEBUG] Selecting strikes for ATM: {atm_strike}")
#     try:
#         df = pd.DataFrame(chain)
#         if df.empty or atm_strike is None:
#             print("⚠️ [DEBUG] No data or ATM strike for selection")
#             return None, None, None, None

#         ce = df[df["option_type"] == "CE"].copy()
#         pe = df[df["option_type"] == "PE"].copy()

#         # Find ATM PE and CE rows
#         atm_pe_row = pe[pe["strike_price"] == atm_strike]
#         atm_ce_row = ce[ce["strike_price"] == atm_strike]

#         sell_pe_row = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
#         sell_ce_row = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None

#         # Find half-price strikes with vectorized operations
#         buy_pe_row = None
#         if sell_pe_row and "ltp" in sell_pe_row and sell_pe_row["ltp"] > 0:
#             target = sell_pe_row["ltp"] / 2
#             pe_candidates = pe[pe["strike_price"] != atm_strike].copy()
#             pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
#             if not pe_candidates.empty:
#                 buy_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

#         buy_ce_row = None
#         if sell_ce_row and "ltp" in sell_ce_row and sell_ce_row["ltp"] > 0:
#             target = sell_ce_row["ltp"] / 2
#             ce_candidates = ce[ce["strike_price"] != atm_strike].copy()
#             ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target).abs()
#             if not ce_candidates.empty:
#                 buy_ce_row = ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

#         print(f"✅ [DEBUG] Strike selection complete:")
#         print(f"   - SELL PE: {sell_pe_row['symbol'] if sell_pe_row else 'None'}")
#         print(f"   - BUY PE: {buy_pe_row['symbol'] if buy_pe_row else 'None'}")
#         print(f"   - SELL CE: {sell_ce_row['symbol'] if sell_ce_row else 'None'}")
#         print(f"   - BUY CE: {buy_ce_row['symbol'] if buy_ce_row else 'None'}")

#         return sell_pe_row, buy_pe_row, sell_ce_row, buy_ce_row

#     except Exception as e:
#         logging.error(f"Error in select_strikes_atm_half_price: {str(e)}", exc_info=True)
#         print(f"❌ [DEBUG] Error in select_strikes_atm_half_price: {str(e)}")
#         return None, None, None, None

# Otm sell and otm hedge
# Optimized strike selection
# def select_strikes_atm_half_price(chain, atm_strike):
#     print(f"🔍 [DEBUG] Selecting strikes for ATM: {atm_strike}")
#     try:
#         df = pd.DataFrame(chain)
#         if df.empty or atm_strike is None:
#             print("⚠️ [DEBUG] No data or ATM strike for selection")
#             return None, None, None, None

#         ce = df[df["option_type"] == "CE"].copy()
#         pe = df[df["option_type"] == "PE"].copy()

#         # --- Get ATM CE/PE rows ---
#         atm_pe_row = pe[pe["strike_price"] == atm_strike]
#         atm_ce_row = ce[ce["strike_price"] == atm_strike]

#         atm_pe = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
#         atm_ce = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None

#         # --- Find SELL PE (half of ATM PE premium) ---
#         sell_pe_row, buy_pe_row = None, None
#         if atm_pe and "ltp" in atm_pe and atm_pe["ltp"] > 0:
#             target_sell = atm_pe["ltp"] / 2
#             pe_candidates = pe[pe["strike_price"] != atm_strike].copy()
#             pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target_sell).abs()
#             if not pe_candidates.empty:
#                 sell_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

#                 # --- Buy hedge for PE (half of sell leg premium) ---
#                 if sell_pe_row and "ltp" in sell_pe_row and sell_pe_row["ltp"] > 0:
#                     target_buy = sell_pe_row["ltp"] / 2
#                     hedge_candidates = pe[pe["strike_price"] != sell_pe_row["strike_price"]].copy()
#                     hedge_candidates["ltp_diff"] = (hedge_candidates["ltp"] - target_buy).abs()
#                     if not hedge_candidates.empty:
#                         buy_pe_row = hedge_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

#         # --- Find SELL CE (half of ATM CE premium) ---
#         sell_ce_row, buy_ce_row = None, None
#         if atm_ce and "ltp" in atm_ce and atm_ce["ltp"] > 0:
#             target_sell = atm_ce["ltp"] / 2
#             ce_candidates = ce[ce["strike_price"] != atm_strike].copy()
#             ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target_sell).abs()
#             if not ce_candidates.empty:
#                 sell_ce_row = ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

#                 # --- Buy hedge for CE (half of sell leg premium) ---
#                 if sell_ce_row and "ltp" in sell_ce_row and sell_ce_row["ltp"] > 0:
#                     target_buy = sell_ce_row["ltp"] / 2
#                     hedge_candidates = ce[ce["strike_price"] != sell_ce_row["strike_price"]].copy()
#                     hedge_candidates["ltp_diff"] = (hedge_candidates["ltp"] - target_buy).abs()
#                     if not hedge_candidates.empty:
#                         buy_ce_row = hedge_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

#         print(f"✅ [DEBUG] Strike selection complete:")
#         print(f"   - SELL PE: {sell_pe_row['symbol'] if sell_pe_row else 'None'}")
#         print(f"   - BUY PE: {buy_pe_row['symbol'] if buy_pe_row else 'None'}")
#         print(f"   - SELL CE: {sell_ce_row['symbol'] if sell_ce_row else 'None'}")
#         print(f"   - BUY CE: {buy_ce_row['symbol'] if buy_ce_row else 'None'}")

#         return sell_pe_row, buy_pe_row, sell_ce_row, buy_ce_row

#     except Exception as e:
#         logging.error(f"Error in select_strikes_atm_half_price: {str(e)}", exc_info=True)
#         print(f"❌ [DEBUG] Error in select_strikes_atm_half_price: {str(e)}")
#         return None, None, None, None


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
def has_open_orders_for_last_signal(fyers, session_state):
    print(f"🔍 [DEBUG] Checking open orders for last signal")
    ids = session_state.live_last_signal_order_ids
    if not ids:
        print("✅ [DEBUG] No order IDs for last signal")
        return False
    try:
        ob = fyers.orderbook().get("orderBook", [])
        has_open = any(o.get("id") in ids and o.get("status") == 2 for o in ob)
        print(f"✅ [DEBUG] Open orders for last signal: {has_open}")
        return has_open
    except Exception as e:
        logging.warning(f"Could not check open orders: {e}")
        print(f"❌ [DEBUG] Error checking open orders: {e}")
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

        if signal not in ["BUY", "SELL"]:
            print("📊 [DEBUG] Signal = SIDEWAYS → No trading action.")
            return []

        # Map entry legs for the new signal
        signal_map = {
            "BUY": {"enter_hedge": buy_pe, "enter_short": sell_pe},
            "SELL": {"enter_hedge": buy_ce, "enter_short": sell_ce}
        }

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
        # 1) FLATTEN EVERYTHING on SIGNAL CHANGE
        # ============================================================
        # ============================================================
        # EXIT LOGIC: SHORTS FIRST, THEN LONGS/HEGES
        # ============================================================
        if current_signal in ["BUY", "SELL"] and current_signal != signal and net_positions:
            print(f"⚠️ [DEBUG] Signal changed ({current_signal} → {signal}). "
                f"Attempting to exit ALL open positions step by step...")

            try:
                # ---------- Step 1: Exit SHORT legs ----------
                exit_shorts = {
                    "segment": [11],          # Equity Derivatives
                    "side": [-1],             # Shorts only
                    "productType": ["ALL"]    # Try ALL first
                }
                print(f"📤 [DEBUG] Sending SHORT exit request: {exit_shorts}")
                resp_short = fyers.exit_positions(data=exit_shorts)
                print(f"📥 [DEBUG] SHORT exit response: {resp_short}")

                if resp_short.get("code") != 200:
                    # Fallback using specific productTypes
                    product_types = list(set([p.get('productType', 'MARGIN') for p in net_positions if p.get("netQty",0) < 0]))
                    fallback_short = {"segment": [11], "side": [-1], "productType": product_types}
                    print(f"📤 [DEBUG] SHORT exit fallback request: {fallback_short}")
                    resp_short = fyers.exit_positions(data=fallback_short)
                    print(f"📥 [DEBUG] SHORT fallback response: {resp_short}")

                # ---------- Step 2: Exit LONG/HEDGE legs ----------
                exit_longs = {
                    "segment": [11],
                    "side": [1],              # Longs only
                    "productType": ["ALL"]
                }
                print(f"📤 [DEBUG] Sending LONG exit request: {exit_longs}")
                resp_long = fyers.exit_positions(data=exit_longs)
                print(f"📥 [DEBUG] LONG exit response: {resp_long}")

                if resp_long.get("code") != 200:
                    product_types = list(set([p.get('productType', 'MARGIN') for p in net_positions if p.get("netQty",0) > 0]))
                    fallback_long = {"segment": [11], "side": [1], "productType": product_types}
                    print(f"📤 [DEBUG] LONG exit fallback request: {fallback_long}")
                    resp_long = fyers.exit_positions(data=fallback_long)
                    print(f"📥 [DEBUG] LONG fallback response: {resp_long}")

                # ---------- Step 3: Verify flattening ----------
                if not _wait_until_flat(timeout=60, interval=2):
                    print("❌ [DEBUG] Flattening failed after stepwise exit. Triggering EMERGENCY exit.")
                    # emergency_exit_positions(fyers, net_positions)
                    return placed_ids

                print("✅ [DEBUG] All positions successfully exited step by step.")

            except Exception as e:
                print(f"❌ [DEBUG] Exception during exit sequence: {str(e)}")
                # emergency_exit_positions(fyers, net_positions)
                return placed_ids
        # ============================================================
        # 2) ENTER NEW BASKET (only if nothing is open)
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

        # Basket entry
        enter_hedge = signal_map[signal]["enter_hedge"]
        enter_short = signal_map[signal]["enter_short"]

        print(f"📥 [DEBUG] Preparing new {signal} basket entry:")
        print(f"  - Hedge leg (BUY): {enter_hedge}")
        print(f"  - Short leg (SELL): {enter_short}")

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

        print(f"📤 [DEBUG] Sending basket order request: {entry_basket}")
        entry_resp = fyers.place_basket_orders(data=entry_basket)
        print(f"📥 [DEBUG] Basket order response: {entry_resp}")

        if entry_resp and entry_resp.get("s") == "ok":
            for item in entry_resp.get("data", []):
                body = item.get("body", {})
                if body.get("s") == "ok" and body.get("id"):
                    placed_ids.append(body["id"])
            session_state.last_signal = signal
            print(f"✅ [DEBUG] Successfully entered new {signal} basket orders → IDs: {placed_ids}")
        else:
            msg = entry_resp.get("message", "Unknown error") if isinstance(entry_resp, dict) else "Unknown error"
            print(f"❌ [DEBUG] Basket order FAILED. Reason: {msg}")

        return placed_ids

    except Exception as e:
        print(f"❌ [DEBUG] Fatal error in handle_basket_orders_atomic: {str(e)}")
        return []
    finally:
        session_state._hb_lock = False
        print("🔓 [DEBUG] Lock released.\n")
#=============================================
# if the above func dont work as expected try this below one
#===============================================
# def handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers):
#     """
#     Robust atomic-ish rebalancer using Fyers responses.
#     - Dynamic exit logic: identify segments from current positions, exit SHORTS then LONGS.
#     - Retries & fallback for exit_positions.
#     - Verifies via positions() after each exit step and uses _wait_until_flat as final safety.
#     - Basket per-leg handling and orderbook reconciliation.
#     Returns list of placed order IDs (accepted/confirmed).
#     """
#     print(f"\n🔍 [DEBUG {datetime.now().isoformat()}] Handling basket orders for signal: {signal}")

#     # ---------- simple lock + watchdog ----------
#     LOCK_WATCHDOG_SECONDS = 120  # if a lock is older than this, force release
#     now_ts = time.time()
#     if getattr(session_state, "_hb_lock", False):
#         lock_time = getattr(session_state, "_hb_lock_ts", None)
#         if lock_time and (now_ts - lock_time) > LOCK_WATCHDOG_SECONDS:
#             print("⚠️ [DEBUG] Lock appears stale → forcing release before continuing.")
#             session_state._hb_lock = False
#             session_state._hb_lock_ts = None
#         else:
#             print("⚠️ [DEBUG] Another rebalance is already running. Skipping this cycle.")
#             return []

#     session_state._hb_lock = True
#     session_state._hb_lock_ts = now_ts

#     placed_ids: List[str] = []
#     try:
#         lot = Config.LOT_SIZE
#         current_signal = getattr(session_state, "last_signal", None)
#         print(f"🔄 [DEBUG] Current signal: {current_signal}, Incoming signal: {signal}")

#         if signal not in ["BUY", "SELL"]:
#             print("📊 [DEBUG] Signal = SIDEWAYS → No trading action.")
#             return []

#         # map legs
#         signal_map = {
#             "BUY": {"enter_hedge": buy_pe, "enter_short": sell_pe},
#             "SELL": {"enter_hedge": buy_ce, "enter_short": sell_ce}
#         }

#         # ---------- safe API wrappers ----------
#         def safe_positions():
#             """Return list of netPositions or an empty list on error."""
#             try:
#                 resp = fyers.positions()
#             except Exception as e:
#                 print(f"❌ [DEBUG] positions() exception: {e}")
#                 return []
#             if not isinstance(resp, dict):
#                 print(f"❌ [DEBUG] positions() returned non-dict: {resp}")
#                 return []
#             if resp.get("s") == "ok" and "netPositions" in resp:
#                 return resp.get("netPositions", []) or []
#             # handle an OK with missing netPositions as empty
#             print(f"⚠️ [DEBUG] positions() unexpected payload: {resp}")
#             return []

#         def safe_exit_positions(payload, retries=2, backoff=1.5):
#             """Call fyers.exit_positions with retries. Return response dict or last resp."""
#             attempt = 0
#             last_resp = None
#             while attempt <= retries:
#                 try:
#                     resp = fyers.exit_positions(data=payload)
#                 except Exception as e:
#                     print(f"❌ [DEBUG] exit_positions exception (attempt {attempt}): {e}")
#                     resp = None
#                 last_resp = resp
#                 if isinstance(resp, dict) and resp.get("s") == "ok" and resp.get("code") == 200:
#                     return resp
#                 attempt += 1
#                 sleep_for = backoff * attempt
#                 print(f"⏳ [DEBUG] exit_positions attempt {attempt} failed/returned {resp}. Retrying after {sleep_for:.1f}s...")
#                 time.sleep(sleep_for)
#             print(f"❌ [DEBUG] exit_positions failed after {retries+1} attempts. Last resp: {last_resp}")
#             return last_resp

#         def safe_place_basket(entry_basket, retries=1):
#             """Place basket orders and return the raw response (or None)."""
#             attempt = 0
#             last_resp = None
#             while attempt <= retries:
#                 try:
#                     resp = fyers.place_basket_orders(data=entry_basket)
#                 except Exception as e:
#                     print(f"❌ [DEBUG] place_basket_orders exception (attempt {attempt}): {e}")
#                     resp = None
#                 last_resp = resp
#                 if isinstance(resp, dict) and resp.get("s") == "ok" and "data" in resp:
#                     return resp
#                 attempt += 1
#                 print(f"⏳ [DEBUG] place_basket_orders retry in {2*attempt}s (attempt {attempt})")
#                 time.sleep(2 * attempt)
#             print(f"❌ [DEBUG] place_basket_orders failed after retries. Last resp: {last_resp}")
#             return last_resp

#         def safe_orderbook():
#             try:
#                 resp = fyers.orderbook()
#             except Exception as e:
#                 print(f"❌ [DEBUG] orderbook() exception: {e}")
#                 return {}
#             if isinstance(resp, dict) and resp.get("s") == "ok" and "orderBook" in resp:
#                 return resp
#             print(f"⚠️ [DEBUG] orderbook() unexpected payload: {resp}")
#             return resp or {}

#         # ---------- wait until flat (with backoff) ----------
#         def _wait_until_flat(timeout=90, interval=2):
#             """Wait until positions show no netQty or until timeout. Returns True if flat."""
#             print(f"⏳ [DEBUG] Waiting up to {timeout}s for positions to flatten...")
#             end = time.time() + timeout
#             attempt = 0
#             while time.time() < end:
#                 attempt += 1
#                 net_positions = safe_positions()
#                 still_open = [p for p in net_positions if abs(p.get("netQty", 0)) != 0]
#                 if not still_open:
#                     print("✅ [DEBUG] All positions are flat now.")
#                     return True
#                 else:
#                     readable = [(p.get("symbol"), p.get("netQty")) for p in still_open]
#                     print(f"⏳ [DEBUG] Still open positions (attempt {attempt}): {readable}")
#                 to_sleep = min(interval * (1 + attempt * 0.2), 6)
#                 time.sleep(to_sleep)
#             print("❌ [DEBUG] Timeout: positions did not flatten in time.")
#             return False

#         # ---------- initial snapshot ----------
#         net_positions = [p for p in safe_positions() if abs(p.get("netQty", 0)) != 0]
#         print(f"🔍 [DEBUG] Found {len(net_positions)} active positions at start:")
#         for p in net_positions:
#             print(f"  - {p.get('symbol')}: netQty={p.get('netQty')} side={p.get('side')} productType={p.get('productType')} segment={p.get('segment')}")

#         # ============================================================
#         # EXIT LOGIC: SHORTS FIRST, THEN LONGS (dynamic segments + verification)
#         # ============================================================
#         if current_signal in ["BUY", "SELL"] and current_signal != signal and net_positions:
#             print(f"⚠️ [DEBUG] Signal changed ({current_signal} → {signal}). Attempting to exit ALL open positions...")

#             try:
#                 def exit_side(side_val: int) -> bool:
#                     """Try to exit all positions for one side (short or long). Returns True if cleared."""
#                     # Determine segments relevant to this side from net_positions
#                     segs = list({p.get("segment") for p in net_positions if (p.get("netQty", 0) < 0 if side_val == -1 else p.get("netQty", 0) > 0)})
#                     if not segs:
#                         print(f"ℹ️ [DEBUG] No segments found for side {side_val} → nothing to exit.")
#                         return True  # nothing to exit

#                     payload = {"segment": [int(s) for s in segs if s is not None], "side": [side_val], "productType": ["ALL"]}
#                     print(f"📤 [DEBUG] Sending exit request for side {side_val}: {payload}")
#                     resp = safe_exit_positions(payload)

#                     # If exit_positions didn't return a clear success, fallback to productType-specific attempt
#                     if not (isinstance(resp, dict) and resp.get("s") == "ok" and resp.get("code") == 200):
#                         # collect productTypes for the side
#                         product_types = list({p.get("productType") for p in net_positions if (p.get("netQty", 0) < 0 if side_val == -1 else p.get("netQty", 0) > 0)})
#                         # use unique non-null productTypes
#                         product_types = [pt for pt in product_types if pt]
#                         if product_types:
#                             fallback = {"segment": [int(s) for s in segs if s is not None], "side": [side_val], "productType": product_types}
#                             print(f"📤 [DEBUG] Exit fallback request for side {side_val}: {fallback}")
#                             resp = safe_exit_positions(fallback)
#                         else:
#                             print(f"⚠️ [DEBUG] No productTypes discovered for fallback for side {side_val}.")

#                     # Verify by checking positions for that side specifically
#                     side_positions = [p for p in safe_positions() if (p.get("netQty", 0) < 0 if side_val == -1 else p.get("netQty", 0) > 0)]
#                     if side_positions:
#                         print(f"⚠️ [DEBUG] After exit attempt, positions still remain for side {side_val}: {[(p['symbol'], p['netQty']) for p in side_positions]}")
#                         return False
#                     print(f"✅ [DEBUG] Exit for side {side_val} appears to have cleared positions.")
#                     return True

#                 ok_shorts = exit_side(-1)
#                 ok_longs = exit_side(1)

#                 if not _wait_until_flat(timeout=90, interval=3):
#                     print("❌ [DEBUG] Flattening failed after exit attempts. Aborting this cycle.")
#                     return placed_ids

#                 if ok_shorts and ok_longs:
#                     print("✅ [DEBUG] All positions exited successfully.")
#                 else:
#                     print("⚠️ [DEBUG] Some exit attempts may not have cleared all positions but flatten check passed/failed accordingly.")

#             except Exception as e:
#                 print(f"❌ [DEBUG] Exception during exit sequence: {e}")
#                 return placed_ids

#         # ============================================================
#         # Enter new basket only if nothing is open or if the open
#         # positions match the intended basket (idempotency)
#         # ============================================================
#         current_positions = [p for p in safe_positions() if abs(p.get("netQty", 0)) != 0]
#         if current_positions:
#             # Check if current positions already correspond to intended basket
#             idx_map = {p.get("symbol"): p for p in current_positions}

#             enter_hedge = signal_map[signal]["enter_hedge"]
#             enter_short = signal_map[signal]["enter_short"]

#             hedge_sym = enter_hedge.get("symbol")
#             short_sym = enter_short.get("symbol")

#             already_has_hedge = hedge_sym in idx_map and idx_map[hedge_sym].get("netQty", 0) > 0
#             already_has_short = short_sym in idx_map and idx_map[short_sym].get("netQty", 0) < 0

#             if already_has_hedge and already_has_short:
#                 print("ℹ️ [DEBUG] Intended basket already present in positions → skipping new entry and updating last_signal.")
#                 session_state.last_signal = signal
#                 return placed_ids
#             else:
#                 print("⚠️ [DEBUG] Some positions are STILL open and do not match intended basket → Skipping new entry to avoid conflicts.")
#                 for p in current_positions:
#                     print(f"  - OPEN: {p.get('symbol')} {p.get('netQty')}")
#                 return placed_ids

#         # ---------- prepare entry basket ----------
#         enter_hedge = signal_map[signal]["enter_hedge"]
#         enter_short = signal_map[signal]["enter_short"]

#         print(f"📥 [DEBUG] Preparing new {signal} basket entry:")
#         print(f"  - Hedge leg (BUY): {enter_hedge}")
#         print(f"  - Short leg (SELL): {enter_short}")

#         entry_basket = [
#             {
#                 "symbol": enter_hedge["symbol"],
#                 "qty": lot,
#                 "type": 2,
#                 "side": 1,
#                 "productType": "MARGIN",
#                 "validity": "DAY",
#             },
#             {
#                 "symbol": enter_short["symbol"],
#                 "qty": lot,
#                 "type": 2,
#                 "side": -1,
#                 "productType": "MARGIN",
#                 "validity": "DAY",
#             }
#         ]

#         # ---------- place basket ----------
#         print(f"📤 [DEBUG] Sending basket order request: {entry_basket}")
#         entry_resp = safe_place_basket(entry_basket)
#         print(f"📥 [DEBUG] Basket order raw response: {entry_resp}")

#         # analyze per-leg results
#         tentative_ids: List[str] = []
#         failed_legs_info: List[Dict[str, Any]] = []
#         if isinstance(entry_resp, dict) and entry_resp.get("s") == "ok" and "data" in entry_resp:
#             for idx, leg in enumerate(entry_resp.get("data", [])):
#                 statusCode = leg.get("statusCode")
#                 body = leg.get("body", {}) if isinstance(leg, dict) else {}
#                 if isinstance(body, dict) and body.get("s") == "ok" and body.get("id"):
#                     tentative_ids.append(body.get("id"))
#                     print(f"✅ [DEBUG] Basket leg {idx} accepted → id={body.get('id')}")
#                 else:
#                     failed_legs_info.append({"idx": idx, "statusCode": statusCode, "body": body})
#                     print(f"⚠️ [DEBUG] Basket leg {idx} failed or returned error: {body}")

#             # verify via orderbook that tentative_ids are present and check their statuses
#             if tentative_ids:
#                 print(f"🔎 [DEBUG] Verifying tentative order IDs in orderbook: {tentative_ids}")
#                 ob_ok = False
#                 ob_end = time.time() + 30  # 30s to reconcile orderbook
#                 while time.time() < ob_end:
#                     ob = safe_orderbook()
#                     order_book_list = ob.get("orderBook", []) if isinstance(ob, dict) else []
#                     found_ids = {o.get("id"): o for o in order_book_list if o.get("id")}
#                     placed_confirmed = []
#                     for tid in tentative_ids:
#                         o = found_ids.get(tid)
#                         if o:
#                             remaining = o.get("remainingQuantity", None)
#                             filled = o.get("filledQty", None)
#                             status = o.get("status", None)
#                             print(f"  - [OB] id={tid} status={status} filledQty={filled} remaining={remaining}")
#                             placed_confirmed.append(tid)
#                     if placed_confirmed:
#                         placed_ids = placed_confirmed
#                         ob_ok = True
#                         break
#                     time.sleep(2)
#                 if not ob_ok:
#                     placed_ids = tentative_ids
#                     print("⚠️ [DEBUG] orderbook reconciliation timed out — accepting tentative IDs but please reconcile manually.")
#             else:
#                 print("❌ [DEBUG] No legs accepted by Fyers in place_basket_orders response.")
#         else:
#             print(f"❌ [DEBUG] Basket order call failed or returned unexpected payload: {entry_resp}")

#         # On success (even partial) update last_signal so we don't re-enter the same basket
#         if placed_ids:
#             session_state.last_signal = signal
#             print(f"✅ [DEBUG] Successfully entered new {signal} basket orders → IDs: {placed_ids}")
#         else:
#             print(f"⚠️ [DEBUG] No placed order IDs returned. No update to last_signal.")

#         return placed_ids

#     except Exception as e:
#         print(f"❌ [DEBUG] Fatal error in handle_basket_orders_atomic: {e}")
#         return []
#     finally:
#         session_state._hb_lock = False
#         session_state._hb_lock_ts = None
#         print(f"🔓 [DEBUG {datetime.now().isoformat()}] Lock released.\n")


# Data display functions with improved formatting
def format_and_show(chain, title, ltp, show_signals=False, fyers_api=None):
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
        signal = compute_signals(merged, atm, ltp, fyers_api=fyers_api)
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
        print("\n" + "=" * 50)
        print("📈 FYERS ALGO TRADING - LIVE AUTO TRADE")
        print("=" * 50)
        print("1. 📈 Start Live Trading")
        print("2. ⚙️ Settings")
        print("3. 🚪 Exit")
        print("=" * 50)

        choice = get_user_input("Select option", input_type=int)

        if choice == 1:  # Live Trading
            clear_screen()
            print("\n" + "=" * 50)
            print("📈 LIVE AUTO TRADING")
            print("=" * 50)

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
                    print('='*50)
                    print(f"\n🔄 [DEBUG] Starting new iteration at {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Update data
                    ltp, chain, curr_date = update_all_data(session_state, session_state.cid, session_state.token, sym)

                    ist = pytz.timezone("Asia/Kolkata")
                    print(f"🕒 Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")

                    if ltp:
                        print(f"✅ Underlying LTP: {ltp}")

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
                            print("⚠️ Missing option data for BUY signal. Unable to place trade.")
                        elif signal == "SELL" and not valid_sell:
                            print("⚠️ Missing option data for SELL signal. Unable to place trade.")
                        elif signal in ("BUY", "SELL"):
                            if session_state.live_last_signal == signal and has_open_orders_for_last_signal(fy, session_state):
                                print(f"ℹ️ Basket for {signal} already open.")
                            else:
                                ids = handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fy)
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