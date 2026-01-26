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
import matplotlib.pyplot as plt
import csv
import math


# Constants
class Config:
    REFRESH_INTERVAL = 60  # seconds
    API_RATE_LIMIT = 200
    API_WINDOW = 60  # seconds
    LOT_SIZE = 130  # nifty lot size
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    EXIT_ON_SIDEWAYS = True

    # Added defaults required by compute_score and strike selection
    ROLLING_WINDOW = 30
    STRIKE_GAP = 50  # Nifty strike gap


# Initialize logging
logging.basicConfig(
    filename="trading_app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_next_aligned_time(interval_seconds):
    """Get next time aligned to the interval"""
    now = datetime.now()
    current_seconds = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    intervals_passed = current_seconds // interval_seconds
    next_aligned_seconds = (intervals_passed + 1) * interval_seconds

    next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=next_aligned_seconds)
    return next_time

def wait_until_next_interval(interval_seconds):
    """Wait until next aligned interval"""
    next_time = get_next_aligned_time(interval_seconds)
    wait_seconds = (next_time - datetime.now()).total_seconds()

    if wait_seconds > 0:
        print(f"⏳ Next update at {next_time.strftime('%H:%M:%S')} (waiting {wait_seconds:.1f}s)")
        time.sleep(wait_seconds)

# Data Logging Class (PNL CSV removed per request)
class DataLogger:
    def __init__(self, filename="trading_data.csv"):
        self.filename = filename
        # PNL CSV logic removed intentionally
        self._ensure_header()

    def _ensure_header(self):
        """Create trading data file with headers if it doesn't exist"""
        try:
            with open(self.filename, 'r') as f:
                pass
        except FileNotFoundError:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'underlying_ltp', 'signal',
                    'atm_strike', 'expiry_date', 'dominant_option',
                    'option_score', 'underlying_score'
                ])

    def log_data(self, timestamp, underlying_ltp, signal, atm_strike,
                 expiry_date, dominant_option='',
                    option_score=0.0, underlying_score=0.0):
        """Log trading data to CSV file with Z-score metrics"""
        try:
            # Ensure consistent types / safe string coercion
            row = [
                str(timestamp),
                float(underlying_ltp) if underlying_ltp is not None else "",
                str(signal) if signal is not None else "",
                int(atm_strike) if (atm_strike is not None and not pd.isna(atm_strike)) else "",
                str(expiry_date) if expiry_date is not None else "",
                str(dominant_option) if dominant_option is not None else "",
                float(option_score) if option_score is not None else "",
                float(underlying_score) if underlying_score is not None else ""

            ]
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"✅ Data logged: {timestamp} - LTP: {underlying_ltp}, Signal: {signal}")
        except Exception as e:
            logging.exception("Error logging trading data")
            print(f"❌ Error logging data: {e}")

    def read_data(self, limit=None):
        """Read logged trading data from CSV file"""
        try:
            df = pd.read_csv(self.filename)
            if limit:
                return df.tail(limit)
            return df
        except Exception as e:
            logging.exception("Error reading trading data")
            print(f"❌ Error reading data: {e}")
            return pd.DataFrame()


# Terminal-based Live Dashboard (PNL display removed)
class TerminalDashboard:
    def __init__(self, data_logger):
        self.data_logger = data_logger
        self.last_update = None

    def display_live_dashboard(self):
        """Display live dashboard in terminal"""
        try:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')

            # Read latest data
            df = self.data_logger.read_data(limit=20)

            print("\n" + "=" * 100)
            print("📊 LIVE TRADING DASHBOARD")
            print("=" * 100)
            print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 100)

            if df.empty:
                print("🔄 Waiting for trading data...")
                return

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # Display current status
            latest = df.iloc[-1]
            # Use safe access with defaults if columns missing or NaN
            ltp = latest.get('underlying_ltp', 0) if isinstance(latest, pd.Series) else 0
            print(f"\n📈 CURRENT STATUS:")
            try:
                print(f"   Nifty LTP: {float(ltp):.2f}")
            except Exception:
                print(f"   Nifty LTP: {ltp}")
            print(f"   Signal: {latest.get('signal', '')}")
            print(f"   ATM Strike: {latest.get('atm_strike', '')}")
            print(f"   Expiry Date: {latest.get('expiry_date', '')}")

            # Display recent signals
            print(f"\n🔄 RECENT SIGNALS (Last 5):")
            recent_signals = df.tail(5)[['timestamp', 'underlying_ltp', 'signal']]
            for _, row in recent_signals.iterrows():
                signal_color = {
                    'BUY': '🟢',
                    'SELL': '🔴',
                    'SIDEWAYS': '🟡',
                    'HOLD': '⚪'
                }.get(row['signal'], '⚪')

                print(
                    f"   {row['timestamp'].strftime('%H:%M:%S')} - {signal_color} {row['signal']} - LTP: {row['underlying_ltp']:.2f}")

            # P&L CSV removed — do not display P&L summary

            # Display signal statistics
            signal_counts = df['signal'].value_counts()
            print(f"\n📊 SIGNAL STATISTICS:")
            for signal, count in signal_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {signal}: {count} times ({percentage:.1f}%)")

            print("\n" + "=" * 100)
            print(f"🔄 Next update in {Config.REFRESH_INTERVAL} seconds... (Press Ctrl+C to stop)")
            print("=" * 100)

        except Exception as e:
            logging.exception("Error displaying dashboard")
            print(f"❌ Error displaying dashboard: {e}")


# Plotting Functions (PNL plotting removed)
def generate_static_plot():
    """Generate a static plot that can be viewed manually"""
    try:
        data_logger = DataLogger()
        df = data_logger.read_data()

        if df.empty:
            print("❌ No data available for plotting")
            return

        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Create the plot
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))

        # Plot Nifty LTP
        ax1.plot(df['timestamp'], df['underlying_ltp'], 'b-', label='Nifty LTP', linewidth=2)
        ax1.set_ylabel('Nifty LTP', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)

        # Add signals
        buy_signals = df[df['signal'] == 'BUY']
        sell_signals = df[df['signal'] == 'SELL']
        sideways_signals = df[df['signal'] == 'SIDEWAYS']

        if not buy_signals.empty:
            ax1.scatter(buy_signals['timestamp'], buy_signals['underlying_ltp'],
                        color='green', s=100, label='BUY', marker='^', alpha=0.8)

        if not sell_signals.empty:
            ax1.scatter(sell_signals['timestamp'], sell_signals['underlying_ltp'],
                        color='red', s=100, label='SELL', marker='v', alpha=0.8)

        if not sideways_signals.empty:
            ax1.scatter(sideways_signals['timestamp'], sideways_signals['underlying_ltp'],
                        color='orange', s=80, label='SIDEWAYS', marker='s', alpha=0.6)

        ax1.legend()
        ax1.set_title('Nifty LTP vs Trading Signals')

        plt.tight_layout()

        # Save plot
        plot_filename = f"trading_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Plot saved as: {plot_filename}")
        return plot_filename

    except Exception as e:
        logging.exception("Error generating plot")
        print(f"❌ Error generating plot: {e}")
        return None


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
        self.signal_buffer = []  # For majority signal calculation
        self.cid = ""
        self.token = ""
        self._hb_lock = False
        self.trade_tracker = {}  # kept for compatibility
        self.last_trade_time = None      # cooldown anchor (set after SIDEWAYS exit)
        self.opening_straddle = None     # opening ATM straddle value
        self.opening_date = None         # date when opening straddle was captured
        self.cooldown_warned = False

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
            print("✅ Credentials loaded from JSON file")
    except FileNotFoundError:
        # Fall back to environment variables if file doesn't exist
        print("⚠️ JSON file not found, checking environment variables")
        # use consistent env var names
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

    if window.total_seconds() > Config.API_WINDOW:
        session_state.api_window_start = now
        session_state.option_chain_api_count = 0
        session_state.quote_api_count = 0
    elif total_calls >= Config.API_RATE_LIMIT:
        wait = Config.API_WINDOW - int(window.total_seconds())
        if wait > 0:
            logging.warning(f"Approaching rate limit, waiting {wait}s...")
            print(f"⚠️ Approaching rate limit, waiting {wait}s...")
            time.sleep(wait)
        session_state.api_window_start = datetime.now()
        session_state.option_chain_api_count = 0
        session_state.quote_api_count = 0


# API calls with retry logic
@handle_errors
def get_option_chain_data(session_state, cid, token, sym, expiry_ts=""):
    # NOTE: Removed lru_cache usage because session_state is not hashable and caching caused issues.
    print(f"🔍 [DEBUG] Fetching option chain data for {sym}, expiry: {expiry_ts or 'nearest'}")
    enforce_rate_limit(session_state)
    session_state.option_chain_api_count += 1
    fy = fyersModel.FyersModel(client_id=cid, token=token, is_async=False, log_path="")
    resp = fy.optionchain(data={"symbol": sym, "timestamp": expiry_ts})
    if not resp or resp.get("code") != 200:
        error_msg = f"Option chain error ({expiry_ts or 'nearest'}): {resp.get('message', 'Unknown') if isinstance(resp, dict) else resp}"
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
        error_msg = f"Quotes API error ({resp.get('code') if isinstance(resp, dict) else 'N/A'}): {resp.get('message', 'Unknown') if isinstance(resp, dict) else resp}"
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
        error_msg = f"Quotes API error for {symbol}: {resp.get('message', 'Unknown') if isinstance(resp, dict) else resp}"
        logging.error(error_msg)
        print(f"❌ {error_msg}")
        return None
    ltp = resp["d"][0]["v"]["lp"]
    print(f"✅ [DEBUG] Symbol LTP: {ltp}")
    return ltp


MAX_HISTORY = 2  # last N signals for majority check
MAJORITY_THRESHOLD = 2  # number of same signals to confirm majority


def majority_signal(history, threshold=MAJORITY_THRESHOLD):
    """
    Determine majority signal from the history.
    Returns the majority signal if all signals in buffer are the same, else HOLD.
    """
    if not history or len(history) < threshold:
        return "HOLD"

    if all(signal == history[0] for signal in history):
        return history[0]

    return "HOLD"

def compute_signals(
        merged_df,
        atm_strike,
        ltp,
        fyers_api=None,
        session_state=None
):
    """
    Dominance-based intraday signal logic
    with ATP–LTP directional permission filter
    PLUS:
    - Opening ATM straddle × 0.6 regime gate
    - 15-minute cooldown (AFTER EXIT)
    """

    print("\n" + "=" * 90)
    print("🚀 [DEBUG] compute_signals STARTED")
    print(f"📌 LTP: {ltp}, Input ATM Strike: {atm_strike}")
    print("=" * 90)

    try:

        COOLDOWN_MINUTES = 5
        STRADDLE_MULTIPLIER = 0.6
        now = datetime.now()

        metrics = {
            'signal': "HOLD",
            'raw_signal': "HOLD",
            'dominant_option': None,
            'option_score': 0,
            'underlying_score': 0,
            'history': []
        }

        if fyers_api is None:
            print("❌ [DEBUG] fyers_api is None → returning HOLD")
            return "HOLD", metrics

        # ======================================================
        # STEP 1: CHECK CURRENT OPEN POSITION
        # ======================================================
        current_position = None
        try:
            pos_resp = fyers_api.positions()
            if pos_resp.get("s") == "ok":
                for p in pos_resp.get("netPositions", []):
                    qty = p.get("netQty", 0)
                    sym = (p.get("symbol") or "").upper()
                    print(f"🔎 [DEBUG] Position → Symbol: {sym}, Qty: {qty}")
                    if qty > 0:
                        if sym.endswith("CE"):
                            current_position = "CE"
                        elif sym.endswith("PE"):
                            current_position = "PE"
        except Exception as e:
            print(f"❌ [DEBUG] Position fetch error: {e}")

        print(f"✅ [DEBUG] Current Position Detected: {current_position}")

        # ======================================================
        # STEP 1.5: CAPTURE OPENING STRADDLE (ONCE PER DAY)
        # ======================================================
        if session_state is not None:
            opening_date = session_state.get("opening_date")
            if opening_date != now.date():
                print("🧭 [DEBUG] Capturing opening straddle")

                q = fyers_api.quotes({"symbols": "NSE:NIFTY50-INDEX"})
                if q.get("s") != "ok":
                    print("❌ [DEBUG] Failed to fetch NIFTY open quote")
                    return "HOLD", metrics

                nifty_open = q["d"][0]["v"]["open_price"]
                session_state.day_open = nifty_open

                atm_open = min(
                    merged_df["Strike"],
                    key=lambda x: abs(x - nifty_open)
                )

                atm_row = merged_df[merged_df["Strike"] == atm_open].iloc[0]
                ce_symbol = atm_row.get("CE_symbol")
                pe_symbol = atm_row.get("PE_symbol")

                if not ce_symbol or not pe_symbol:
                    print("❌ [DEBUG] ATM symbols missing at open")
                    return "HOLD", metrics

                q_opt = fyers_api.quotes({
                    "symbols": f"{ce_symbol},{pe_symbol}"
                })

                if q_opt.get("s") != "ok":
                    print("❌ [DEBUG] Failed to fetch ATM straddle quotes")
                    return "HOLD", metrics

                ce_open = q_opt["d"][0]["v"]["open_price"]
                pe_open = q_opt["d"][1]["v"]["open_price"]

                opening_straddle = ce_open + pe_open

                session_state.opening_straddle = opening_straddle
                session_state.opening_date = now.date()

                print(f"📊 [DEBUG] Opening straddle = {opening_straddle:.2f}")
            else:
                print("🟩 [DEBUG] Opening straddle already captured")

        # ======================================================
        # STEP 2: HELPER FUNCTIONS
        # ======================================================
        def fetch_history(symbol):
            today = datetime.now().strftime("%Y-%m-%d")
            print(f"📥 [DEBUG] Fetching history for {symbol} on {today}")
            resp = fyers_api.history({
                "symbol": symbol,
                "resolution": "1",
                "date_format": 1,
                "range_from": today,
                "range_to": today,
                "cont_flag": 1
            })

            if resp.get("s") != "ok" or not resp.get("candles"):
                print(f"❌ [DEBUG] History fetch failed for {symbol}")
                return None

            print(f"📊 [DEBUG] History fetched for {symbol}, candles={len(resp['candles'])}")
            return pd.DataFrame(
                resp["candles"],
                columns=["epoch", "open", "high", "low", "close", "volume"]
            )

        def compute_score(df, label=""):
            """
            Computes momentum score for a series of prices.
            R3, R10, R30 = returns over 3, 10, 30 periods
            M = average return
            score = normalized by recent volatility (sigma)
            """
            closes = df["close"]
            if len(closes) < 32:
                print(f"❌ [DEBUG] Not enough candles for {label}, len={len(closes)}")
                return None

            # Momentum over different periods
            R3 = closes.iloc[-2] / closes.iloc[-5] - 1
            R10 = closes.iloc[-2] / closes.iloc[-12] - 1
            R30 = closes.iloc[-2] / closes.iloc[-32] - 1
            M = (R3 + R10 + R30) / 3

            # Volatility (sigma) based on % changes
            sigma = closes.pct_change().std()
            score = M / sigma if sigma and sigma > 0 else 0.0

            print(
                f"📈 [DEBUG] {label} | R3={R3:.4f}, R10={R10:.4f}, R30={R30:.4f}, M={M:.4f}, σ={sigma:.4f}, SCORE={score:.4f}")
            return {"score": score, "M": M, "sigma": sigma}

        def is_dominant(opt_score, und_score, opt_M, sigma_opt, sigma_und, eps=1e-6):
            """
            Determines if an option is dominant using actual noise barrier.
            - opt_score: option momentum score
            - und_score: underlying momentum score
            - opt_M: option average momentum
            - sigma_opt: volatility of option
            - sigma_und: volatility of underlying

            Dominance is confirmed if option outperforms underlying by more than combined noise.
            """
            # Combined noise of option + underlying

            sigma_diff = 1.0

            # Dominant if:
            # 1. Option momentum is positive
            # 2. Option outperforms underlying beyond noise
            dominant = (opt_M > 0) and ((abs(opt_score) - abs(und_score)) > sigma_diff) and abs(opt_score) > abs(underlying_score)

            print(
                f"🏁 [DEBUG] Dominance Check → "
                f"OptScore={opt_score:.4f}, UndScore={und_score:.4f}, M={opt_M:.4f}, "
                f"σ_opt={sigma_opt:.4f}, σ_und={sigma_und:.4f}, NoiseBarrier={sigma_diff:.4f}, "
                f"Dominant={dominant}"
            )

            return dominant

        def fetch_quotes_avg_atp_ltp(symbols):
            print(f"📥 [DEBUG] Fetching quotes for symbols: {symbols}")
            try:
                resp = fyers_api.quotes({"symbols": ",".join(symbols)})
                if resp.get("s") != "ok":
                    print("❌ [DEBUG] Quotes fetch failed")
                    return None, None

                atps, ltps = [], []
                for d in resp.get("d", []):
                    v = d.get("v", {})
                    atp = v.get("atp")
                    ltp = v.get("lp")
                    if atp is not None and ltp is not None:
                        atps.append(atp)
                        ltps.append(ltp)
                if not atps or not ltps:
                    print("❌ [DEBUG] No valid ATP/LTP found in quotes")
                    return None, None
                avg_atp, avg_ltp = sum(atps) / len(atps), sum(ltps) / len(ltps)
                print(f"📊 [DEBUG] Quotes averages → avg_ATP={avg_atp}, avg_LTP={avg_ltp}")
                return avg_atp, avg_ltp
            except Exception as e:
                print(f"❌ [DEBUG] Quotes exception: {e}")
                return None, None

        # ======================================================
        # STEP 3: UNDERLYING SCORE
        # ======================================================
        print("\n🧭 [STEP 3] Computing underlying (Nifty) score")
        udf = fetch_history("NSE:NIFTY50-INDEX")
        if udf is None:
            print("❌ [DEBUG] Unable to fetch underlying history → HOLD")
            return "HOLD", metrics

        u = compute_score(udf, "UNDERLYING")
        if u is None:
            print("❌ [DEBUG] Unable to compute underlying score → HOLD")
            return "HOLD", metrics

        underlying_score = u["score"]
        metrics["underlying_score"] = underlying_score
        print(f"✅ [DEBUG] Underlying Score: {underlying_score:.4f}")

        # ======================================================
        # STEP 3.5: STRADDLE REGIME GATE
        # ======================================================
        if session_state is not None and session_state.get("opening_straddle"):
            day_high = udf["high"].max()
            day_low = udf["low"].min()
            realized_range = day_high - day_low
            threshold = STRADDLE_MULTIPLIER * session_state.opening_straddle
            print(f"📏 [DEBUG] Regime Gate → DayHigh={day_high:.2f}, DayLow={day_low:.2f}, Range={realized_range:.2f}, Threshold={threshold:.2f}")
            if realized_range < threshold:
                print("🛑 [DEBUG] Regime BLOCK → HOLD")
                return "HOLD", metrics
            else:
                print("✅ [DEBUG] Regime PASSED")

        # ======================================================
        # STEP 4: ATM ± 1 STRIKE SELECTION
        # ======================================================
        print("\n🎯 [STEP 4] Selecting ATM ± 1 strikes")
        atm = min(merged_df["Strike"], key=lambda x: abs(x - ltp))
        print(f"📌 [DEBUG] Calculated ATM Strike: {atm}")

        strike_df = merged_df[
            merged_df["Strike"].isin([atm, atm + Config.STRIKE_GAP, atm - Config.STRIKE_GAP])
        ].copy()
        print(f"📊 [DEBUG] Selected strikes DF size: {len(strike_df)}")
        if strike_df.empty:
            print("❌ [DEBUG] No strikes available → HOLD")
            return "HOLD", metrics

        # ======================================================
        # STEP 5: OPTION DOMINANCE
        # ======================================================
        print("\n🏎️ [STEP 5] Evaluating CE & PE dominance")

        dominant_options = []
        evaluated_options = []  # ✅ ADDED (tracks ALL evaluated strikes)

        for _, row in strike_df.iterrows():
            strike = row["Strike"]
            for opt_type in ["CE", "PE"]:
                symbol = row.get(f"{opt_type}_symbol")
                if pd.isna(symbol):
                    continue

                print(f"🔍 [DEBUG] Evaluating {symbol} at strike {strike}")

                df_opt = fetch_history(symbol)
                if df_opt is None:
                    print(f"❌ [DEBUG] History not available for {symbol}")
                    continue

                s = compute_score(df_opt, symbol)
                if s is None:
                    print(f"❌ [DEBUG] Score not computed for {symbol}")
                    continue

                # ✅ TRACK EVERY evaluated option (dominant or not)
                evaluated_options.append({
                    "symbol": symbol,
                    "type": opt_type,
                    "score": s["score"]
                })

                if is_dominant(
                        s["score"],  # option score
                        underlying_score,  # underlying score
                        s["M"],  # option momentum M
                        s["sigma"],  # sigma_opt
                        u["sigma"]  # sigma_und
                ):
                    dominant_options.append({
                        "symbol": symbol,
                        "type": opt_type,
                        "score": s["score"]
                    })
                    print(f"✅ [DEBUG] {symbol} added as dominant")
                else:
                    print(f"❌ [DEBUG] {symbol} NOT dominant")

        dominant_options.sort(key=lambda x: x["score"], reverse=True)
        print(f"📊 [DEBUG] Total dominant options found: {len(dominant_options)}")

        top_option = dominant_options[0] if dominant_options else None
        rank1_type = top_option["type"] if top_option else None
        rank1_symbol = top_option["symbol"] if top_option else None
        print(f"🏆 [DEBUG] Rank-1 Option: {rank1_symbol}, Type: {rank1_type}")
        metrics['dominant_option'] = rank1_symbol
        metrics['option_score'] = top_option["score"] if top_option else 0.0

        # ======================================================
        # STEP 5.5: ATP–LTP DIRECTIONAL GATE
        # ======================================================
        print("\n🧪 [STEP 5.5] Applying ATP–LTP gate")

        ce_symbols, pe_symbols = [], []
        for _, row in strike_df.iterrows():
            if pd.notna(row.get("CE_symbol")):
                ce_symbols.append(row["CE_symbol"])
            if pd.notna(row.get("PE_symbol")):
                pe_symbols.append(row["PE_symbol"])

        ce_atp_avg, ce_ltp_avg = fetch_quotes_avg_atp_ltp(ce_symbols)
        pe_atp_avg, pe_ltp_avg = fetch_quotes_avg_atp_ltp(pe_symbols)

        allow_buy = ce_atp_avg is not None and ce_ltp_avg is not None and ce_ltp_avg > ce_atp_avg
        allow_sell = pe_atp_avg is not None and pe_ltp_avg is not None and pe_ltp_avg > pe_atp_avg
        print(f"📊 [DEBUG] ATP–LTP Gate → BUY_ALLOWED={allow_buy}, SELL_ALLOWED={allow_sell}")

        # ======================================================
        # STEP 6: RAW SIGNAL DECISION
        # ======================================================
        print("\n🧭 [STEP 6] Computing raw signal")

        raw_signal = "HOLD"

        # ---- TOTAL strikes evaluated ----
        total_ce = len([opt for opt in evaluated_options if opt["type"] == "CE"])
        total_pe = len([opt for opt in evaluated_options if opt["type"] == "PE"])

        # ---- DOMINANT strikes ----
        dominant_types = [opt["type"] for opt in dominant_options]
        ce_count = dominant_types.count("CE")
        pe_count = dominant_types.count("PE")

        print(
            f"📊 [DEBUG] Strike Summary → "
            f"Total CE={total_ce}, Total PE={total_pe} | "
            f"Dominant CE={ce_count}, Dominant PE={pe_count}"
        )

        # ======================================================
        # ENTRY LOGIC (no open position)
        # ======================================================
        if current_position is None:

            if total_ce > 0 and ce_count == total_ce and allow_buy:
                raw_signal = "BUY"

            elif total_pe > 0 and pe_count == total_pe and allow_sell:
                raw_signal = "SELL"

            else:
                print("❌ [DEBUG] Dominance NOT unanimous across strikes")

        # ======================================================
        # HOLD / EXIT LOGIC (position open)
        # ======================================================
        else:

            if current_position == "CE":
                raw_signal = "HOLD" if ce_count == total_ce else "SIDEWAYS"

            elif current_position == "PE":
                raw_signal = "HOLD" if pe_count == total_pe else "SIDEWAYS"

        print(
            f"📌 [DEBUG] Raw signal: {raw_signal} | "
            f"Position={current_position}"
        )

        # ======================================================
        # STEP 7: MAJORITY CONFIRMATION
        # ======================================================
        print("\n🧭 [STEP 7] Majority signal confirmation")
        history = list(session_state.get("signal_history", [])) if session_state else []
        history.append(raw_signal)
        history = history[-MAX_HISTORY:]
        if session_state is not None:
            session_state.signal_history = history

        confirmed_signal = majority_signal(history, threshold=MAJORITY_THRESHOLD)
        metrics["signal"] = confirmed_signal
        metrics["history"] = history
        print(f"🎯 [DEBUG] Signal History: {history}")
        print(f"🎯 [DEBUG] Confirmed signal: {confirmed_signal}")

        print("=" * 90 + "\n")

        # ======================================================
        # STEP 7.5: COOLDOWN STATUS (VISIBILITY — ALWAYS)
        # ======================================================
        if session_state is not None:
            last_trade_time = session_state.get("last_trade_time")

            if last_trade_time:
                elapsed_mins = (now - last_trade_time).total_seconds() / 60
                remaining_mins = max(0, COOLDOWN_MINUTES - elapsed_mins)

                if elapsed_mins < COOLDOWN_MINUTES:
                    print(
                        f"⏳ [DEBUG] Cooldown ACTIVE → "
                        f"elapsed={elapsed_mins:.2f} min | remaining={remaining_mins:.2f} min"
                    )
                else:
                    print("✅ [DEBUG] Cooldown completed → trading allowed")

            else:
                if not getattr(session_state, "cooldown_warned", False):
                    print("⚠️ [DEBUG] Cooldown anchor not set yet (no SIDEWAYS exit so far)")
                    session_state.cooldown_warned = True

        # ======================================================
        # STEP 8: COOLDOWN ENFORCEMENT (LOGIC — BUY/SELL ONLY)
        # ======================================================
        if session_state is not None and confirmed_signal in ("BUY", "SELL"):
            last_trade_time = session_state.get("last_trade_time")
            if last_trade_time:
                elapsed_mins = (now - last_trade_time).total_seconds() / 60
                remaining_mins = max(0, COOLDOWN_MINUTES - elapsed_mins)
                print(
                    f"⏱️ [DEBUG] Cooldown elapsed = {elapsed_mins:.2f} min, "
                    f"remaining = {remaining_mins:.2f} min"
                )
                if elapsed_mins < COOLDOWN_MINUTES:
                    print("🛑 [DEBUG] Cooldown ACTIVE → HOLD")
                    confirmed_signal = "HOLD"
                    metrics["signal"] = "HOLD"

        return confirmed_signal, metrics

    except Exception as e:
        logging.error("Error in compute_signals", exc_info=True)
        print(f"❌ [DEBUG] Exception: {e}")
        return "SIDEWAYS", metrics



def select_strikes_atm_half_price(chain, atm_strike):
    """
    Strategy:
      - BUY ATM (both CE & PE)
      - SELL OTM where LTP ≈  ATM LTP / 4
    """
    print(f"🔍 [DEBUG] Selecting strikes for ATM Strike: {atm_strike}")
    try:
        df = pd.DataFrame(chain)
        if df.empty or atm_strike is None:
            print("⚠️ [DEBUG] No data or ATM strike for selection")
            return None, None, None, None

        ce = df[df["option_type"] == "CE"].copy()
        pe = df[df["option_type"] == "PE"].copy()

        # ---------------- ATM Buy legs ----------------
        atm_pe_row = pe[pe["strike_price"] == atm_strike]
        atm_ce_row = ce[ce["strike_price"] == atm_strike]

        buy_pe_row = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
        buy_ce_row = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None

        # ---------------- OTM Sell legs (half price) ----------------
        sell_pe_row, sell_ce_row = None, None

        if buy_pe_row and "ltp" in buy_pe_row and buy_pe_row["ltp"] > 0:
            target = buy_pe_row["ltp"] / 4
            # Only consider OTM PEs (strike < ATM strike)
            pe_candidates = pe[pe["strike_price"] < atm_strike].copy()
            pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
            if not pe_candidates.empty:
                sell_pe_row = pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        if buy_ce_row and "ltp" in buy_ce_row and buy_ce_row["ltp"] > 0:
            target = buy_ce_row["ltp"] / 4
            # Only consider OTM CEs (strike > ATM strike)
            ce_candidates = ce[ce["strike_price"] > atm_strike].copy()
            ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target).abs()
            if not ce_candidates.empty:
                sell_ce_row = ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()

        print(f"✅ [DEBUG] Strike selection complete:")
        print(f"   - BUY PE : {buy_pe_row['symbol'] if buy_pe_row else 'None'} @ {buy_pe_row['ltp'] if buy_pe_row else 'N/A'}")
        # print(f"   - SELL PE : {sell_pe_row['symbol'] if sell_pe_row else 'None'} @ {sell_pe_row['ltp'] if sell_pe_row else 'N/A'}")
        print(f"   - BUY CE : {buy_ce_row['symbol'] if buy_ce_row else 'None'} @ {buy_ce_row['ltp'] if buy_ce_row else 'N/A'}")
        # print(f"   - SELL CE : {sell_ce_row['symbol'] if sell_ce_row else 'None'} @ {sell_ce_row['ltp'] if sell_ce_row else 'N/A'}")


        return buy_pe_row, sell_pe_row, buy_ce_row, sell_ce_row

    except Exception as e:
        logging.error(f"Error in select_strikes_atm_half_price: {str(e)}", exc_info=True)
        print(f"❌ [DEBUG] Error in select_strikes_atm_half_price: {str(e)}")
        return None, None, None, None


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


def has_open_orders_for_last_signal(fyers, session_state):
    print(f"🔍 [DEBUG] Checking open orders for last signal")

    # First check if we have any actual positions
    try:
        pos_resp = fyers.positions()
        if pos_resp.get("s") == "ok":
            net_positions = [p for p in pos_resp.get("netPositions", [])
                             if abs(p.get("netQty", 0)) != 0]
            if not net_positions:
                print(f"✅ [DEBUG] No active positions found")
                session_state.last_signal_order_ids = []  # Clear stale order IDs
                return False
            else:
                print(f"📊 [DEBUG] Found {len(net_positions)} active positions")
                for p in net_positions:
                    print(f"  - {p.get('symbol')}: {p.get('netQty')} units")
                return True
    except Exception as e:
        print(f"❌ [DEBUG] Error checking positions: {e}")

    # Fallback to order book check
    ids = session_state.live_last_signal_order_ids
    if not ids:
        print("✅ [DEBUG] No order IDs for last signal")
        return False

    try:
        ob = fyers.orderbook()
        if ob and "orderBook" in ob:
            # Check for open orders (status 2 = open, 1 = completed, 5 = cancelled)
            open_orders = [o for o in ob["orderBook"]
                           if o.get("id") in ids and o.get("status") == 2]
            has_open = len(open_orders) > 0
            print(f"✅ [DEBUG] Open orders for last signal: {has_open} ({len(open_orders)} orders)")
            return has_open
        print("⚠️ [DEBUG] No order book data")
        return False
    except Exception as e:
        logging.warning(f"Could not check open orders: {e}")
        print(f"❌ [DEBUG] Error checking open orders: {e}")
        return False


def handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers, data_logger):
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
        # print(f"📡 [DEBUG] Initial positions API response: {pos_resp}")
        net_positions = []
        if pos_resp.get("s") == "ok":
            net_positions = [p for p in pos_resp.get("netPositions", [])
                             if abs(p.get("netQty", 0)) != 0]

        print(f"🔍 [DEBUG] Found {len(net_positions)} active positions at start:")
        for p in net_positions:
            print(f"  - {p.get('symbol')}: {p.get('netQty')} units")

        # ============================================================
        # 1) Handle HOLD signal (maintain existing positions)
        # ============================================================
        if signal == "HOLD":
            print("📊 [DEBUG] HOLD signal → Maintaining existing positions.")
            # Do nothing, just keep existing positions
            return []

        # ============================================================
        # 2) Handle SIDEWAYS signal (exit all positions)
        # ============================================================
        if signal == "SIDEWAYS" and Config.EXIT_ON_SIDEWAYS == True:
            print("📊 [DEBUG] SIDEWAYS signal -> EXIT_ON_SIDEWAYS == True -> Exiting all positions.")

            if not net_positions:
                print("✅ [DEBUG] No positions to exit for SIDEWAYS signal.")
                session_state.last_signal = signal
                return []

            try:
                # NOTE: PNL CSV logic removed. We keep prints for traceability.
                timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                for position in net_positions:
                    symbol = position.get('symbol', '')
                    net_qty = position.get('netQty', 0)
                    avg_price = position.get('avgPrice', 0)
                    print(f"ℹ️ Exiting position {symbol} qty={net_qty} avg_price={avg_price} at {timestamp}")

                # Exit all positions
                exit_shorts = {"segment": [11], "side": [-1], "productType": ["MARGIN"]}
                print(f"📤 [DEBUG] Sending SHORT exit request: {exit_shorts}")
                resp_short = fyers.exit_positions(data=exit_shorts)
                print(f"📥 [DEBUG] SHORT exit response: {resp_short}")

                exit_longs = {"segment": [11], "side": [1], "productType": ["MARGIN"]}
                print(f"📤 [DEBUG] Sending LONG exit request: {exit_longs}")
                resp_long = fyers.exit_positions(data=exit_longs)
                print(f"📥 [DEBUG] LONG exit response: {resp_long}")

                # Verify flattening
                if not _wait_until_flat(timeout=60, interval=2):
                    print("❌ [DEBUG] Flattening failed after exit. Manual check needed.")
                    return placed_ids

                print("✅ [DEBUG] All positions successfully exited for SIDEWAYS signal.")
                session_state.last_signal = signal
                session_state.last_trade_time = datetime.now()
                print("🕒 [DEBUG] Cooldown started after SIDEWAYS exit")
                return []

            except Exception as e:
                print(f"❌ [DEBUG] Exception during SIDEWAYS exit sequence: {str(e)}")
                return placed_ids
        if signal == "SIDEWAYS" and Config.EXIT_ON_SIDEWAYS == False:
            print("📊 [DEBUG] SIDEWAYS signal -> EXIT_ON_SIDEWAYS == False -> Holding all positions.")
            return []
        # ============================================================
        # 3) FLATTEN on SIGNAL CHANGE (BUY/SELL to different BUY/SELL)
        # ============================================================
        if current_signal in ["BUY", "SELL"] and current_signal != signal and net_positions:
            print(f"⚠️ [DEBUG] Signal changed ({current_signal} → {signal}). "
                  f"Attempting to exit ALL open positions step by step...")

            try:
                timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                for position in net_positions:
                    symbol = position.get('symbol', '')
                    net_qty = position.get('netQty', 0)
                    avg_price = position.get('avgPrice', 0)
                    print(f"ℹ️ Exiting position {symbol} qty={net_qty} avg_price={avg_price} at {timestamp}")

                # Step 1: Exit SHORT legs
                exit_shorts = {"segment": [11], "side": [-1], "productType": ["MARGIN"]}
                print(f"📤 [DEBUG] Sending SHORT exit request: {exit_shorts}")
                resp_short = fyers.exit_positions(data=exit_shorts)
                print(f"📥 [DEBUG] SHORT exit response: {resp_short}")

                # Step 2: Exit LONG legs
                exit_longs = {"segment": [11], "side": [1], "productType": ["MARGIN"]}
                print(f"📤 [DEBUG] Sending LONG exit request: {exit_longs}")
                resp_long = fyers.exit_positions(data=exit_longs)
                print(f"📥 [DEBUG] LONG exit response: {resp_long}")

                # Step 3: Verify flattening
                if not _wait_until_flat(timeout=60, interval=2):
                    print("❌ [DEBUG] Flattening failed after exit. Manual check needed.")
                    return placed_ids

                print("✅ [DEBUG] All positions successfully exited for signal change.")

            except Exception as e:
                print(f"❌ [DEBUG] Exception during exit sequence: {str(e)}")
                return placed_ids

        # ============================================================
        # 4) ENTER NEW BASKET for BUY/SELL signals (ATM BUY only per request)
        # ============================================================
        signal_map = {
            "BUY": {"enter_hedge": buy_ce, "enter_short": None},
            "SELL": {"enter_hedge": buy_pe, "enter_short": None}
        }
        pos_resp = fyers.positions()
        # print(f"📡 [DEBUG] Positions before entry: {pos_resp}")
        current_positions = []
        if pos_resp.get("s") == "ok":
            current_positions = [p for p in pos_resp.get("netPositions", [])
                                 if abs(p.get("netQty", 0)) != 0]

        if current_positions:
            print("⚠️ [DEBUG] Some positions are STILL open → Skipping new entry.")
            for p in current_positions:
                print(f"  - OPEN: {p.get('symbol')} {p.get('netQty')}")
            return placed_ids

        # Basket entry - build legs conditionally (ATM buy only)
        enter_hedge = signal_map[signal]["enter_hedge"]
        enter_short = signal_map[signal]["enter_short"]

        print(f"📥 [DEBUG] Preparing new {signal} basket entry:")
        print(f"  - ATM BUY leg: {enter_hedge}")
        if enter_short:
            print(f"  - (Would have been SELL leg): {enter_short}")

        entry_basket = []
        if enter_hedge:
            # validate shape before accessing keys
            if not isinstance(enter_hedge, dict) or "symbol" not in enter_hedge:
                print("❌ [DEBUG] Invalid ATM buy leg data. Aborting entry.")
                return placed_ids
            entry_basket.append({
                "symbol": enter_hedge["symbol"],
                "qty": lot,
                "type": 2,
                "side": 1,  # BUY
                "productType": "MARGIN",
                "validity": "DAY",
            })

        if enter_short:
            if not isinstance(enter_short, dict) or "symbol" not in enter_short:
                print("❌ [DEBUG] Invalid short leg data. Skipping short leg.")
            else:
                entry_basket.append({
                    "symbol": enter_short["symbol"],
                    "qty": lot,
                    "type": 2,
                    "side": -1,  # SELL
                    "productType": "MARGIN",
                    "validity": "DAY",
                })

        if not entry_basket:
            print("❌ [DEBUG] No valid legs to submit for basket entry. Aborting.")
            return placed_ids

        print(f"📤 [DEBUG] Sending basket order request: {entry_basket}")
        entry_resp = fyers.place_basket_orders(data=entry_basket)
        print(f"📥 [DEBUG] Basket order response: {entry_resp}")

        if entry_resp and entry_resp.get("s") == "ok":
            for item in entry_resp.get("data", []):
                body = item.get("body", {})
                if body.get("s") == "ok" and body.get("id"):
                    placed_ids.append(body["id"])

            # NOTE: PNL CSV logic removed. Keep a trace print instead.
            timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
            if enter_hedge and isinstance(enter_hedge, dict) and "ltp" in enter_hedge:
                print(f"ℹ️ Logged ENTRY event (no PNL CSV): {signal} {enter_hedge.get('symbol')} @ {enter_hedge.get('ltp')} at {timestamp}")

            session_state.last_signal = signal
            print(f"✅ [DEBUG] Successfully entered new {signal} basket orders → IDs: {placed_ids}")
        else:
            msg = entry_resp.get("message", "Unknown error") if isinstance(entry_resp, dict) else "Unknown error"
            print(f"❌ [DEBUG] Basket order FAILED. Reason: {msg}")

        return placed_ids

    except Exception as e:
        logging.exception("Fatal error in handle_basket_orders_atomic")
        print(f"❌ [DEBUG] Fatal error in handle_basket_orders_atomic: {str(e)}")
        return []
    finally:
        session_state._hb_lock = False
        print("🔓 [DEBUG] Lock released.\n")


# Data display functions with improved formatting
def format_and_show(chain, title, ltp, show_signals=False, fyers_api=None, session_state=None):
    print(f"🔍 [DEBUG] Formatting and displaying data for: {title}")
    df = pd.DataFrame(chain)
    if df.empty:
        print(f"ℹ️ [DEBUG] No data for {title}")
        return None, None, None

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
        signal, metrics = compute_signals(merged, atm, ltp, fyers_api=fyers_api, session_state=session_state)
        # Display results with improved formatting
        print(f"✅ Signal: {signal}")

    # Display the dataframe
    print(f"📊 ATM Strike: {atm} | Underlying LTP: {ltp}")
    # print(merged.to_string(index=False))

    if show_signals and atm is not None:
        return signal, atm, metrics
    return None, atm, None


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

    # Initialize data logger and dashboard
    data_logger = DataLogger()
    dashboard = TerminalDashboard(data_logger)

    # Default symbol
    sym = 'NSE:NIFTY50-INDEX'

    while True:
        clear_screen()
        print("\n" + "=" * 100)
        print("📈 FYERS ALGO TRADING - LIVE AUTO TRADE")
        print("=" * 50)
        print("1. 📈 Start Live Trading (with Auto Dashboard)")
        print("2. ⚙️ Settings")
        print("3. 📊 Generate Static Plot")
        print("4. 🚪 Exit")
        print("=" * 50)

        choice = get_user_input("Select option", input_type=int)

        if choice == 1:  # Live Trading
            clear_screen()
            print("\n" + "=" * 100)
            print("📈 LIVE AUTO TRADING")
            print("=" * 100)

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
            print("📊 Live dashboard will update automatically every cycle...")

            try:
                iteration_count = 0
                while True:
                    iteration_count += 1

                    # Display dashboard every iteration
                    dashboard.display_live_dashboard()

                    print(f"\n🔄 [DEBUG] Starting iteration {iteration_count} at {datetime.now().strftime('%H:%M:%S')}")

                    # Update data
                    ltp, chain, curr_date = update_all_data(session_state, session_state.cid, session_state.token, sym)

                    ist = pytz.timezone("Asia/Kolkata")
                    print(f"🕒 Data Updated: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')}")

                    if ltp:
                        print(f"✅ Underlying LTP: {ltp}")

                    if chain:
                        fy = fyersModel.FyersModel(client_id=session_state.cid, token=session_state.token,
                                                   is_async=False,
                                                   log_path="")
                        signal, atm, metrics = format_and_show(chain, f"Current Expiry: {curr_date}", ltp,
                                                               show_signals=True,
                                                               fyers_api=fy, session_state=session_state)

                        # Log data with metrics
                        if signal and ltp:
                            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
                            # metrics may be None in some edge cases; guard access
                            if not metrics:
                                metrics = {}
                            data_logger.log_data(
                                timestamp=timestamp,
                                underlying_ltp=ltp,
                                atm_strike=atm,
                                expiry_date=curr_date,
                                dominant_option=metrics.get("dominant_option", ""),
                                option_score=metrics.get("option_score", 0.0),
                                underlying_score=metrics.get("underlying_score", 0.0),
                                signal=signal
                            )

                        # Select strikes
                        buy_pe, sell_pe, buy_ce, sell_ce = select_strikes_atm_half_price(chain, atm)

                        # VALIDATION CHANGED: we now only require the ATM BUY leg for each signal
                        valid_buy = (buy_ce is not None)   # For BUY signal, ATM CE buy leg must exist
                        valid_sell = (buy_pe is not None)  # For SELL signal, ATM PE buy leg must exist

                        # Trading logic - handle all signal types including HOLD
                        if signal == "HOLD":
                            print("📊 HOLD signal → Maintaining existing positions.")
                            # No action needed, just continue holding

                        elif signal == "BUY" and not valid_buy:
                            print("⚠️ Missing ATM CE buy data for BUY signal. Unable to place trade.")
                        elif signal == "SELL" and not valid_sell:
                            print("⚠️ Missing ATM PE buy data for SELL signal. Unable to place trade.")
                        elif signal in ("BUY", "SELL"):
                            if session_state.live_last_signal == signal and has_open_orders_for_last_signal(fy,
                                                                                                            session_state):
                                print(f"ℹ️ ATM BUY for {signal} already open.")
                            else:
                                # Note: function signature expects (session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers, data_logger)
                                ids = handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce,
                                                                  buy_ce, fy, data_logger)
                                session_state.live_last_signal = signal
                                session_state.live_last_signal_order_ids = ids
                        elif signal == "SIDEWAYS":
                            handle_basket_orders_atomic(session_state, signal, sell_pe, buy_pe, sell_ce, buy_ce, fy,
                                                        data_logger)
                        else:
                            print("ℹ️ No action.")

                    print(f"⏳ Waiting {Config.REFRESH_INTERVAL} seconds before next update...")
                    # time.sleep(Config.REFRESH_INTERVAL)
                    wait_until_next_interval(Config.REFRESH_INTERVAL)

            except KeyboardInterrupt:
                print("\n⏹️ Live trading stopped by user")
                input("Press Enter to continue...")

            except Exception as e:
                logging.exception("Error in live trading loop")
                print(f"❌ Error in live trading: {str(e)}")
                input("Press Enter to continue...")

        elif choice == 2:  # Settings
            settings_menu(session_state)

        elif choice == 3:  # Generate Static Plot
            print("📊 Generating static plot...")
            plot_file = generate_static_plot()
            if plot_file:
                print(f"✅ Plot saved as: {plot_file}")
            input("Press Enter to continue...")

        elif choice == 4:  # Exit
            print("👋 Exiting... Goodbye!")
            break

        else:
            print("❌ Invalid option. Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()