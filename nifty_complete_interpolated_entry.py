import time
import numpy as np
from datetime import datetime
import pytz
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import logging
import os
import json
from collections import deque
import csv
import threading
import traceback

# ============================================================
# PAPER TRADE FLAG
# ============================================================
PAPER_MODE = True   # Change to False for live trading

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    SAMPLE_INTERVAL = 1
    OPTION_CHAIN_INTERVAL = 30

    WINDOW_SECONDS = 300        # 5 minutes (rolling window)
    WINDOW_BARS = WINDOW_SECONDS
    REQUIRED_BARS = WINDOW_SECONDS + 1

    ATM_RANGE = 2
    LOT_SIZE = 195
    MAX_DAILY_ENTRIES = 100
    API_RATE_LIMIT = 200
    API_WINDOW = 60
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    WS_HEARTBEAT_TIMEOUT = 10
    MIN_TRAILING_DISTANCE = 0.5

    # Entry execution confirmation
    ENTRY_PRICE_RANGE_PERCENT = 1.0      # ±1% around interpolated price at signal
    ENTRY_WAIT_TIMEOUT_SECONDS = 30      # cancel pending signal after this many seconds
    OPTION_PRICE_STALE_SECONDS = 5       # reject stale option ticks for interpolation/execution

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    filename="trading_app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_debug(msg):
    logging.debug(msg)

def log_console(msg):
    print(msg, flush=True)
    logging.info(msg)

IST = pytz.timezone("Asia/Kolkata")

def now_ist():
    return datetime.now(IST)

def timestamp_ist():
    return now_ist().strftime("%Y-%m-%d %H:%M:%S")

# ============================================================
# ERROR HANDLER
# ============================================================
def handle_errors(func):
    def wrapper(*args, **kwargs):
        for retry in range(Config.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_debug(f"{func.__name__} attempt {retry+1} failed: {e}\n{traceback.format_exc()}")
                if retry < Config.MAX_RETRIES-1:
                    time.sleep(Config.RETRY_DELAY)
                else:
                    log_debug(f"{func.__name__} permanently failed")
                    return None, None, None, None, []
    return wrapper

# ============================================================
# DATA LOGGER (only needed fields)
# ============================================================
class DataLogger:
    def __init__(self, filename="trading_data.csv"):
        self.filename = filename
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'ltp', 'straddle',
                    'rolling_max_ltp', 'rolling_min_ltp',
                    'rolling_max_straddle',
                    'held_symbol', 'held_entry_premium', 'held_current_premium',
                    'trailing_offset', 'trailing_stop',
                    'held_pnl', 'total_paper_pnl', 'daily_entries', 'bars_collected',
                    'decision_reason'
                ])

    def log_data(self, timestamp, ltp, straddle,
                 rolling_max_ltp, rolling_min_ltp,
                 rolling_max_straddle,
                 active_positions, total_pnl, daily_entries, bars_collected,
                 decision_reason):

        held_s = ''
        held_e = 0.0
        held_c = 0.0
        trail_off = 0.0
        trail_stop = 0.0
        held_p = 0.0

        if len(active_positions) > 0:
            pos = active_positions[0]
            held_s = pos.get('symbol', '')
            held_e = pos.get('entry_premium', 0.0) or 0.0
            held_c = pos.get('current_premium', held_e) or 0.0
            trail_off = pos.get('trailing_offset', 0.0) or 0.0
            # Compute current trailing stop based on recorded extreme
            entry_type = pos.get('entry_type')
            if entry_type == 'CE':
                trail_high = pos.get('trail_high', pos.get('entry_ltp', 0.0))
                trail_stop = trail_high - trail_off
            else:  # PE
                trail_low = pos.get('trail_low', pos.get('entry_ltp', 0.0))
                trail_stop = trail_low + trail_off
            if held_e > 0 and held_c > 0:
                held_p = (held_c - held_e) * Config.LOT_SIZE

        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                str(timestamp),
                round(ltp, 2) if ltp else 0.0,
                round(straddle, 2) if straddle else 0.0,
                round(rolling_max_ltp, 2),
                round(rolling_min_ltp, 2),
                round(rolling_max_straddle, 2),
                held_s,
                round(held_e, 2),
                round(held_c, 2),
                round(trail_off, 2),
                round(trail_stop, 2),
                round(held_p, 2),
                round(total_pnl, 2),
                daily_entries,
                bars_collected,
                decision_reason
            ])

# ============================================================
# SESSION STATE (added live_ltp_time for freshness tracking)
# ============================================================
class SessionState:
    def __init__(self):
        self.lock = threading.RLock()
        self.option_chain_api_count = 0
        self.api_window_start = now_ist()
        self.cid = ""
        self.token = ""
        self.daily_entries = 0
        self.position_state = "FLAT"      # FLAT / WAITING / ACTIVE
        self.pending_signal = None
        self.eod_square_off_triggered = False
        self.warmup_start_time = None

        self.sold_time = {}
        self.symbol_histories = {}
        self.live_ltp = {}
        self.live_ltp_time = {}          # NEW: track timestamp of last LTP update
        self.live_vwap = {}
        self.live_volume = {}

        self.option_chain = []
        self.fut_price = None
        self.expiry_date = None
        self.atm_strike = None
        self.option_symbols = []

        self.strike_to_symbols = {}

        self.active_positions = []
        self.india_vix = 15.0
        self.paper_realized_pnl = 0.0

        self.decision_reason = ""
        self.running = False
        self._order_lock = False

        # Rolling histories – only what we actually use
        self.ltp_history = deque(maxlen=Config.REQUIRED_BARS)
        self.straddle_history = deque(maxlen=Config.REQUIRED_BARS)
        self.bars_collected = 0

        # Rolling extremes updated every tick
        self.rolling_max_ltp = 0.0
        self.rolling_min_ltp = 0.0
        self.rolling_max_straddle = 0.0

        self.current_ltp = 0.0
        self.current_straddle = 0.0

        self.straddle_data_live = False
        self.last_data_time = None

    def start_warmup(self):
        with self.lock:
            if self.warmup_start_time is None:
                self.warmup_start_time = time.time()
                log_console(f"⏳ Warmup started. Need {Config.REQUIRED_BARS} bars of data.")

    def is_warming_up(self):
        return self.bars_collected < Config.REQUIRED_BARS

    def warmup_remaining(self):
        return max(0, Config.REQUIRED_BARS - self.bars_collected)

    def update_ws(self, symbol, ltp, vwap, volume):
        with self.lock:
            self.last_data_time = time.time()
            if ltp is not None:
                # Record freshness for EVERY symbol (including VIX) so staleness
                # can be detected generically instead of only for the index.
                self.live_ltp[symbol] = float(ltp)
                self.live_ltp_time[symbol] = time.time()
            if symbol == "NSE:INDIAVIX-INDEX":
                if ltp is not None:
                    self.india_vix = float(ltp)
                return
            if vwap is not None:
                self.live_vwap[symbol] = float(vwap)
            if volume is not None:
                self.live_volume[symbol] = int(volume)
            if symbol.endswith(('CE','PE')) or symbol == "NSE:NIFTY50-INDEX":
                if symbol not in self.symbol_histories:
                    self.symbol_histories[symbol] = deque(maxlen=10000)
                if ltp is not None:
                    self.symbol_histories[symbol].append((time.time(), float(ltp), float(vwap) if vwap else 0.0))

    def get_ltp(self, symbol):
        with self.lock:
            return self.live_ltp.get(symbol)

    # ---- ADDED: method to get the timestamp of last LTP update ----
    def get_ltp_time(self, symbol):
        with self.lock:
            return self.live_ltp_time.get(symbol)

    def get_symbol_data_counts(self, recency_seconds=5):
        """Return (total symbols ever seen, count of symbols updated within recency_seconds)"""
        with self.lock:
            total = len(self.live_ltp_time)
            now = time.time()
            recent = sum(1 for t in self.live_ltp_time.values() if now - t <= recency_seconds)
            return total, recent

    def get_stale_symbols(self, recency_seconds=5):
        """Return the specific symbols that have data but haven't updated recently,
        so it's obvious on the dashboard WHICH symbol stopped ticking, not just that one did."""
        with self.lock:
            now = time.time()
            return sorted(s for s, t in self.live_ltp_time.items() if now - t > recency_seconds)

    def update_chain(self, chain, fut_price, expiry_date, atm_strike, option_symbols):
        with self.lock:
            self.option_chain = chain
            self.fut_price = float(fut_price) if fut_price else 0.0
            self.expiry_date = expiry_date
            self.atm_strike = int(atm_strike) if atm_strike else None
            self.option_symbols = option_symbols

            self.strike_to_symbols = {}
            for opt in chain:
                strike = opt.get("strike_price")
                if strike is None:
                    continue
                ot = opt.get("option_type")
                sym = opt.get("symbol")
                if ot == "CE":
                    if strike not in self.strike_to_symbols:
                        self.strike_to_symbols[strike] = [None, None]
                    self.strike_to_symbols[strike][0] = sym
                elif ot == "PE":
                    if strike not in self.strike_to_symbols:
                        self.strike_to_symbols[strike] = [None, None]
                    self.strike_to_symbols[strike][1] = sym

    def reset_for_new_day(self):
        with self.lock:
            if self.active_positions:
                log_console(f"⚠️ {len(self.active_positions)} position(s) still tracked, not resetting.")
                return
            self.daily_entries = 0
            self.eod_square_off_triggered = False
            self.position_state = "FLAT"
            self.pending_signal = None
            self.paper_realized_pnl = 0.0
            self.warmup_start_time = None
            self.sold_time.clear()
            self.decision_reason = ""
            self.bars_collected = 0
            self.ltp_history.clear()
            self.straddle_history.clear()
            self.active_positions.clear()
            self.last_data_time = None
            self.straddle_data_live = False
            # Also clear live data
            self.live_ltp.clear()
            self.live_ltp_time.clear()
            self.live_vwap.clear()
            self.live_volume.clear()

# ============================================================
# WEBSOCKET MANAGER (on_open always re-subscribes)
# ============================================================
class MarketWSManager:
    def __init__(self, session_state, access_token):
        self.session = session_state
        self.access_token = access_token
        self.ws = None
        self.subscribed = set()
        self.subscribed_at = {}   # symbol -> time we first asked to subscribe to it
        self.running = False
        self.ws_thread = None

    def on_message(self, message):
        symbol = message.get("symbol")
        if symbol:
            ltp = message.get("ltp")
            vwap = message.get("avg_trade_price")
            volume = message.get("vol_traded_today")
            self.session.update_ws(symbol, ltp, vwap, volume)

    def on_error(self, msg):
        log_console(f"WS error: {msg}")

    def on_close(self, msg):
        log_console(f"WS closed: {msg}. Automatic reconnection is enabled.")

    def on_open(self):
        log_console("✅ WebSocket on_open() triggered")
        # Re-subscribe all symbols after every (re)connect
        if self.subscribed:
            self.ws.subscribe(list(self.subscribed), "SymbolUpdate")
        self.ws.keep_running()

    def _ws_connect(self):
        try:
            self.ws = data_ws.FyersDataSocket(
                access_token=self.access_token,
                log_path="",
                litemode=False,
                write_to_file=False,
                reconnect=True,
                on_connect=self.on_open,
                on_close=self.on_close,
                on_error=self.on_error,
                on_message=self.on_message
            )
            self.ws.connect()
        except Exception as e:
            log_console(f"❌ WebSocket connection thread crashed: {e}")
            log_debug(traceback.format_exc())
        finally:
            log_console("WebSocket thread finished.")

    def start(self):
        if self.running:
            return
        self.running = True
        self.ws_thread = threading.Thread(target=self._ws_connect, daemon=True)
        self.ws_thread.start()

    def subscribe(self, symbols):
        new_symbols = set(symbols) - self.subscribed
        if not new_symbols:
            return
        now = time.time()
        if self.ws:
            try:
                self.ws.subscribe(list(new_symbols), "SymbolUpdate")
            except Exception as e:
                # Don't let a mid-flight send failure disappear silently — this is
                # exactly the kind of thing that used to leave a symbol "subscribed"
                # in our bookkeeping while no data ever actually arrived for it.
                log_console(f"⚠️ Subscribe request failed for {sorted(new_symbols)}: {e}")
                log_debug(traceback.format_exc())
        else:
            log_debug(f"subscribe() called before WS was ready; relying on on_open() to send: {sorted(new_symbols)}")
        # Mark as subscribed either way: on_open() re-sends everything in self.subscribed
        # on every (re)connect, and the heartbeat monitor will force a resubscribe below
        # if a symbol never actually produces a tick.
        self.subscribed.update(new_symbols)
        for s in new_symbols:
            self.subscribed_at.setdefault(s, now)

    def resubscribe(self, symbols):
        """Force a re-send of the subscribe request for symbols already marked as
        subscribed. Used to recover a symbol whose feed went quiet, or whose original
        subscribe request never actually reached the server, without tearing down
        the whole connection (which would interrupt every other symbol too)."""
        symbols = list(symbols)
        if not self.ws or not symbols:
            return
        try:
            self.ws.subscribe(symbols, "SymbolUpdate")
        except Exception as e:
            log_console(f"⚠️ Resubscribe failed for {symbols}: {e}")
            log_debug(traceback.format_exc())

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close_connection()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2)

    def is_alive(self):
        return self.ws_thread and self.ws_thread.is_alive()

# ============================================================
# REST API (unchanged)
# ============================================================
def enforce_rate_limit(session):
    now = now_ist()
    window = (now - session.api_window_start).total_seconds()
    if window > Config.API_WINDOW:
        session.api_window_start = now
        session.option_chain_api_count = 0
    elif session.option_chain_api_count >= Config.API_RATE_LIMIT:
        wait = Config.API_WINDOW - window
        if wait > 0:
            time.sleep(wait)
        session.api_window_start = now
        session.option_chain_api_count = 0

@handle_errors
def fetch_option_chain(session, index_symbol):
    enforce_rate_limit(session)
    session.option_chain_api_count += 1
    fy = fyersModel.FyersModel(client_id=session.cid, token=session.token, is_async=False, log_path="")
    meta = fy.optionchain(data={"symbol": index_symbol, "timestamp": ""})
    if meta.get("code") != 200:
        return None, None, None, None, []

    fut_price = None
    for item in meta["data"]["optionsChain"]:
        if item.get("option_type") == "":
            fut_price = item.get("fp")
            break
    if fut_price is None:
        fut_price = meta["data"].get("fp", 0.0)

    exp_list = sorted([(e["date"], e["expiry"]) for e in meta["data"]["expiryData"]],
                      key=lambda x: datetime.strptime(x[0], "%d-%m-%Y"))
    today = now_ist().date()
    idx = 0
    for i, (d_str, _) in enumerate(exp_list):
        if datetime.strptime(d_str, "%d-%m-%Y").date() >= today:
            idx = i
            break
    curr_date, expiry_ts = exp_list[idx]

    payload = {"symbol": index_symbol, "strikecount": Config.ATM_RANGE, "timestamp": expiry_ts, "greeks": "1"}
    session.option_chain_api_count += 1
    resp = fy.optionchain(data=payload)
    if resp.get("code") != 200:
        return None, None, None, None, []

    chain = resp["data"]["optionsChain"]
    for opt in chain:
        greeks = opt.get("greeks", {})
        opt["delta"] = greeks.get("delta", 0.0) if greeks else 0.0
        opt["gamma"] = greeks.get("gamma", 0.0) if greeks else 0.0
        opt["oi"] = opt.get("oi", 0)

    option_symbols = [opt["symbol"] for opt in chain if opt.get("option_type") in ("CE","PE")]
    return chain, fut_price, curr_date, expiry_ts, option_symbols

# ============================================================
# CHAIN POLLER (unchanged)
# ============================================================
def chain_poller(session, ws_manager, index_symbol):
    while session.running:
        try:
            chain, fut_price, expiry_date, _, opt_symbols = fetch_option_chain(session, index_symbol)
            if chain is not None:
                underlying_ltp = session.get_ltp(index_symbol)
                if underlying_ltp is None:
                    underlying_ltp = fut_price
                strikes = sorted(set(opt["strike_price"] for opt in chain if opt.get("option_type") in ("CE","PE")))
                if strikes and underlying_ltp:
                    atm = min(strikes, key=lambda x: abs(x - underlying_ltp))
                else:
                    atm = None
                session.update_chain(chain, fut_price, expiry_date, atm, opt_symbols)
                all_syms = [index_symbol, "NSE:INDIAVIX-INDEX"] + opt_symbols
                ws_manager.subscribe(all_syms)
            time.sleep(Config.OPTION_CHAIN_INTERVAL)
        except Exception as e:
            log_debug(f"Chain poller error: {e}\n{traceback.format_exc()}")
            time.sleep(Config.OPTION_CHAIN_INTERVAL)

# ============================================================
# LIVE INTERPOLATION
# ============================================================
def get_interpolated_option_prices_live(session, spot):
    """
    Interpolate the CE and PE premiums at the exact current underlying price.

    These interpolated CE/PE values are used for:
      1. Signal generation
      2. Recording the reference price when a signal is generated
      3. The ±ENTRY_PRICE_RANGE_PERCENT entry confirmation check

    The actual listed ATM option is NOT used for those decisions. It is used
    only after confirmation, when selecting the real ATM contract to execute.
    """
    strike_map = session.strike_to_symbols
    if not strike_map:
        return None

    strikes = sorted(strike_map.keys())
    low_strike = None
    high_strike = None

    for s in strikes:
        if s <= spot:
            low_strike = s
        elif s > spot:
            high_strike = s
            break

    if low_strike is None or high_strike is None:
        return None

    ce_low_sym, pe_low_sym = strike_map[low_strike]
    ce_high_sym, pe_high_sym = strike_map[high_strike]

    required_symbols = [ce_low_sym, ce_high_sym, pe_low_sym, pe_high_sym]
    if any(sym is None for sym in required_symbols):
        return None

    now = time.time()
    for sym in required_symbols:
        ts = session.get_ltp_time(sym)
        if ts is None or (now - ts) > Config.OPTION_PRICE_STALE_SECONDS:
            return None

    ce_low = session.get_ltp(ce_low_sym)
    ce_high = session.get_ltp(ce_high_sym)
    pe_low = session.get_ltp(pe_low_sym)
    pe_high = session.get_ltp(pe_high_sym)

    if None in (ce_low, ce_high, pe_low, pe_high):
        return None

    width = high_strike - low_strike
    if width <= 0:
        return None

    w = (spot - low_strike) / width

    ce_interp = ce_low * (1 - w) + ce_high * w
    pe_interp = pe_low * (1 - w) + pe_high * w

    if ce_interp <= 0 or pe_interp <= 0:
        return None

    return {
        "ce": float(ce_interp),
        "pe": float(pe_interp),
        "straddle": float(ce_interp + pe_interp),
        "low_strike": low_strike,
        "high_strike": high_strike,
    }


def get_interpolated_straddle_live(session, spot):
    """Backward-compatible helper for callers that only need the straddle."""
    prices = get_interpolated_option_prices_live(session, spot)
    if prices is None:
        return None
    return prices["straddle"]


def get_current_real_atm_contract(session, spot, option_type):
    """
    Select the CURRENT real ATM contract only at execution time.
    Returns (symbol, live_or_chain_premium, strike).
    """
    chain = session.option_chain
    if not chain:
        return None, None, None

    target = "CE" if option_type == "CE" else "PE"
    best_opt = None
    best_dist = float("inf")

    for opt in chain:
        if opt.get("option_type") != target:
            continue

        strike = opt.get("strike_price")
        if strike is None:
            continue

        dist = abs(strike - spot)
        if dist < best_dist:
            best_dist = dist
            best_opt = opt

    if best_opt is None:
        return None, None, None

    symbol = best_opt.get("symbol")
    strike = best_opt.get("strike_price")
    premium = session.get_ltp(symbol) if symbol else None

    if premium is None or premium <= 0:
        premium = best_opt.get("ltp")

    if premium is None or premium <= 0:
        return symbol, None, strike

    return symbol, float(premium), strike


def get_real_atm_premium(session, spot, option_type):
    """Backward-compatible helper."""
    _, premium, _ = get_current_real_atm_contract(session, spot, option_type)
    return premium


def clear_pending_signal(session):
    session.pending_signal = None
    if not session.active_positions:
        session.position_state = "FLAT"


def is_within_entry_range(reference_price, current_price):
    if reference_price is None or current_price is None:
        return False
    if reference_price <= 0 or current_price <= 0:
        return False

    pct = Config.ENTRY_PRICE_RANGE_PERCENT / 100.0
    lower = reference_price * (1.0 - pct)
    upper = reference_price * (1.0 + pct)
    return lower <= current_price <= upper


def create_position(session, symbol, entry_premium, entry_type, entry_ltp, distance):
    pos = {
        "symbol": symbol,
        "entry_premium": entry_premium,
        "entry_type": entry_type,
        "entry_ltp": entry_ltp,
        "current_premium": entry_premium,
        "trailing_offset": distance,
    }

    if entry_type == "CE":
        pos["trail_high"] = entry_ltp
    else:
        pos["trail_low"] = entry_ltp

    session.active_positions.append(pos)
    session.daily_entries += 1
    session.position_state = "ACTIVE"
    return pos

# ============================================================
# STATE MACHINE
# ============================================================
def process_snapshot(session, fyers, execute_trades=False):
    """
    State priority:
      1. Update market history whenever a valid interpolated snapshot exists.
      2. ACTIVE  -> manage the real option position / trailing exit.
      3. WAITING -> either confirm interpolated entry or expire after timeout.
      4. FLAT    -> detect a new breakout and create a fresh pending signal.

    The same signal/wait/timeout logic is used for PAPER_MODE and LIVE mode.
    PAPER_MODE changes only how the final confirmed order is executed.
    """
    if getattr(session, '_order_lock', False):
        return

    session._order_lock = True
    try:
        with session.lock:
            now = time.time()

            # ------------------------------------------------------------
            # Always allow a pending signal to expire, even if market data
            # is currently missing/stale. This prevents a stuck WAITING state.
            # ------------------------------------------------------------
            if session.pending_signal is not None:
                pending_elapsed = now - session.pending_signal["signal_time"]
                if pending_elapsed >= Config.ENTRY_WAIT_TIMEOUT_SECONDS:
                    expired_type = session.pending_signal["entry_type"]
                    clear_pending_signal(session)
                    session.decision_reason = (
                        f"{expired_type} pending signal expired after "
                        f"{pending_elapsed:.1f}s; waiting for a new signal"
                    )
                    log_console(
                        f"⏰ Pending {expired_type} signal expired after "
                        f"{pending_elapsed:.1f}s"
                    )
                    return

            spot = session.get_ltp("NSE:NIFTY50-INDEX")
            if spot is None or spot <= 0:
                session.decision_reason = "No index LTP"
                return

            index_ts = session.get_ltp_time("NSE:NIFTY50-INDEX")
            if index_ts is None or (now - index_ts) > Config.OPTION_PRICE_STALE_SECONDS:
                session.decision_reason = "Index LTP stale; skipping this snapshot"
                return

            interp = get_interpolated_option_prices_live(session, spot)
            if interp is None:
                session.decision_reason = (
                    "Interpolated CE/PE unavailable or stale; no entry/exit decision"
                )
                return

            interp_ce = interp["ce"]
            interp_pe = interp["pe"]
            straddle = interp["straddle"]

            session.current_ltp = spot
            session.current_straddle = straddle

            # ------------------------------------------------------------
            # IMPORTANT: rolling data continues updating during WAITING.
            # ------------------------------------------------------------
            session.ltp_history.append(spot)
            session.straddle_history.append(straddle)

            if session.bars_collected < Config.REQUIRED_BARS:
                session.bars_collected += 1

            if len(session.ltp_history) >= 2:
                previous_ltp = list(session.ltp_history)[:-1]
                previous_straddle = list(session.straddle_history)[:-1]
                if previous_ltp:
                    session.rolling_max_ltp = max(previous_ltp)
                    session.rolling_min_ltp = min(previous_ltp)
                if previous_straddle:
                    session.rolling_max_straddle = max(previous_straddle)

            # ------------------------------------------------------------
            # ACTIVE POSITION
            # ------------------------------------------------------------
            if session.active_positions:
                pos = session.active_positions[0]

                offset = (
                    session.rolling_max_ltp - session.rolling_min_ltp
                ) / 2.0
                offset = max(offset, Config.MIN_TRAILING_DISTANCE)
                pos["trailing_offset"] = offset

                if pos["entry_type"] == "CE":
                    pos["trail_high"] = max(
                        pos.get("trail_high", pos["entry_ltp"]), spot
                    )
                    stop = pos["trail_high"] - offset
                    should_exit = spot <= stop
                    exit_label = "CE"
                    comparison = "≤"
                else:
                    pos["trail_low"] = min(
                        pos.get("trail_low", pos["entry_ltp"]), spot
                    )
                    stop = pos["trail_low"] + offset
                    should_exit = spot >= stop
                    exit_label = "PE"
                    comparison = "≥"

                session.decision_reason = (
                    f"Holding {exit_label} | Stop={stop:.2f} | Spot={spot:.2f}"
                )

                if not should_exit or not execute_trades:
                    current = session.get_ltp(pos["symbol"])
                    if current is not None and current > 0:
                        pos["current_premium"] = current
                    return

                held_symbol = pos["symbol"]

                if PAPER_MODE:
                    exit_premium = session.get_ltp(held_symbol)
                    if exit_premium is None or exit_premium <= 0:
                        exit_premium = pos["entry_premium"]

                    pnl = (
                        exit_premium - pos["entry_premium"]
                    ) * Config.LOT_SIZE
                    session.paper_realized_pnl += pnl
                    session.active_positions.clear()
                    session.position_state = "FLAT"
                    session.decision_reason = (
                        f"{exit_label} exited at trailing stop {comparison} {stop:.2f}"
                    )
                    log_console(
                        f"🔴 PAPER EXIT {exit_label} {held_symbol} | "
                        f"Premium {exit_premium:.2f} | P&L={pnl:+.2f}"
                    )
                    return

                order = {
                    "symbol": held_symbol,
                    "qty": Config.LOT_SIZE,
                    "type": 2,
                    "side": -1,
                    "productType": "MARGIN",
                    "validity": "DAY",
                }
                resp = fyers.place_order(data=order)
                if resp.get("s") == "ok":
                    session.active_positions.clear()
                    session.position_state = "FLAT"
                    session.decision_reason = f"{exit_label} live exit submitted"
                    log_console(
                        f"🔴 LIVE EXIT {exit_label} {held_symbol} | "
                        f"Order ID: {resp.get('id')}"
                    )
                else:
                    session.decision_reason = f"Live exit failed: {resp}"
                    log_console(f"❌ Failed to exit {held_symbol}: {resp}")
                return

            # ------------------------------------------------------------
            # WAITING FOR ENTRY CONFIRMATION
            # ------------------------------------------------------------
            if session.pending_signal is not None:
                pending = session.pending_signal
                elapsed = now - pending["signal_time"]

                selected_interp = (
                    interp_ce if pending["entry_type"] == "CE" else interp_pe
                )
                reference = pending["reference_price"]

                pct = Config.ENTRY_PRICE_RANGE_PERCENT / 100.0
                lower = reference * (1.0 - pct)
                upper = reference * (1.0 + pct)

                if not is_within_entry_range(reference, selected_interp):
                    remaining = max(
                        0.0,
                        Config.ENTRY_WAIT_TIMEOUT_SECONDS - elapsed
                    )
                    session.decision_reason = (
                        f"WAITING {pending['entry_type']} | "
                        f"Interpolated={selected_interp:.2f} | "
                        f"Range={lower:.2f}-{upper:.2f} | "
                        f"{remaining:.1f}s left"
                    )
                    return

                # Confirmation is based ONLY on the interpolated selected option.
                # The real ATM contract is selected only now, at execution time.
                entry_type = pending["entry_type"]
                symbol, entry_premium, real_atm_strike = (
                    get_current_real_atm_contract(session, spot, entry_type)
                )

                if symbol is None or entry_premium is None or entry_premium <= 0:
                    remaining = max(
                        0.0,
                        Config.ENTRY_WAIT_TIMEOUT_SECONDS - elapsed
                    )
                    session.decision_reason = (
                        f"{entry_type} interpolation confirmed, but real ATM "
                        f"contract/premium unavailable; {remaining:.1f}s left"
                    )
                    return

                distance = (
                    session.rolling_max_ltp - session.rolling_min_ltp
                ) / 2.0
                distance = max(distance, Config.MIN_TRAILING_DISTANCE)

                if not execute_trades:
                    # Auto-trading disabled: consume the confirmed signal once.
                    clear_pending_signal(session)
                    session.decision_reason = (
                        f"{entry_type} confirmed; auto-trading disabled"
                    )
                    log_console(
                        f"✅ {entry_type} confirmed by interpolated price; "
                        f"auto-trading disabled"
                    )
                    return

                if PAPER_MODE:
                    create_position(
                        session, symbol, entry_premium,
                        entry_type, spot, distance
                    )
                    clear_pending_signal(session)
                    session.position_state = "ACTIVE"
                    session.decision_reason = (
                        f"PAPER entered {symbol} after interpolated confirmation"
                    )
                    log_console(
                        f"🟢 PAPER BUY {symbol} @ {entry_premium:.2f} | "
                        f"Current real ATM={real_atm_strike} | "
                        f"Interpolated={selected_interp:.2f} within "
                        f"±{Config.ENTRY_PRICE_RANGE_PERCENT:.2f}% of "
                        f"{reference:.2f}"
                    )
                    return

                # LIVE: one submission attempt. We do not blindly resubmit on
                # an ambiguous response because duplicate market orders are dangerous.
                pending["order_submission_attempted"] = True
                order = {
                    "symbol": symbol,
                    "qty": Config.LOT_SIZE,
                    "type": 2,
                    "side": 1,
                    "productType": "MARGIN",
                    "validity": "DAY",
                }
                resp = fyers.place_order(data=order)

                if resp.get("s") == "ok":
                    create_position(
                        session, symbol, entry_premium,
                        entry_type, spot, distance
                    )
                    clear_pending_signal(session)
                    session.position_state = "ACTIVE"
                    session.decision_reason = (
                        f"LIVE entry submitted for {symbol} after interpolation confirmation"
                    )
                    log_console(
                        f"🟢 LIVE BUY {symbol} @ observed LTP {entry_premium:.2f} | "
                        f"Current real ATM={real_atm_strike} | "
                        f"Order ID: {resp.get('id')}"
                    )
                else:
                    # Consume the signal after a definite failed response.
                    # We intentionally do not repeatedly fire market orders.
                    clear_pending_signal(session)
                    session.decision_reason = f"Live entry failed: {resp}"
                    log_console(f"❌ LIVE entry failed for {symbol}: {resp}")
                return

            # ------------------------------------------------------------
            # FLAT: WARMUP AND NEW BREAKOUT DETECTION
            # ------------------------------------------------------------
            if session.is_warming_up():
                session.decision_reason = (
                    f"Warming up ({session.bars_collected}/{Config.REQUIRED_BARS})"
                )
                return

            if session.daily_entries >= Config.MAX_DAILY_ENTRIES:
                session.decision_reason = "Max daily entries reached"
                return

            buy_signal = (
                spot > session.rolling_max_ltp
                and straddle > session.rolling_max_straddle
            )
            sell_signal = (
                spot < session.rolling_min_ltp
                and straddle > session.rolling_max_straddle
            )

            if buy_signal and sell_signal:
                if (spot - session.rolling_max_ltp) >= (
                    session.rolling_min_ltp - spot
                ):
                    sell_signal = False
                else:
                    buy_signal = False

            if not buy_signal and not sell_signal:
                session.decision_reason = "IDLE (no breakout)"
                return

            entry_type = "CE" if buy_signal else "PE"
            reference_price = interp_ce if entry_type == "CE" else interp_pe
            pct = Config.ENTRY_PRICE_RANGE_PERCENT / 100.0

            session.pending_signal = {
                "entry_type": entry_type,
                "signal_time": now,
                "signal_spot": float(spot),
                "signal_interp_ce": float(interp_ce),
                "signal_interp_pe": float(interp_pe),
                "reference_price": float(reference_price),
            }
            session.position_state = "WAITING"

            lower = reference_price * (1.0 - pct)
            upper = reference_price * (1.0 + pct)

            session.decision_reason = (
                f"{entry_type} signal recorded | Reference={reference_price:.2f} | "
                f"Range={lower:.2f}-{upper:.2f} | "
                f"Timeout={Config.ENTRY_WAIT_TIMEOUT_SECONDS}s"
            )
            log_console(
                f"🔔 {entry_type} SIGNAL | Spot={spot:.2f} | "
                f"Straddle={straddle:.2f} | "
                f"Signal interpolated CE={interp_ce:.2f}, PE={interp_pe:.2f} | "
                f"Selected reference={reference_price:.2f} | "
                f"Allowed range={lower:.2f}-{upper:.2f} | "
                f"Timer={Config.ENTRY_WAIT_TIMEOUT_SECONDS}s"
            )
            return

    except Exception as e:
        log_console(f"process_snapshot error: {e}")
        log_debug(traceback.format_exc())
        session.decision_reason = f"Error: {str(e)}"
    finally:
        session._order_lock = False

# ============================================================
# SAMPLER THREAD
# ============================================================
def sampler(session, data_logger, auto_trade, fyers):
    log_debug("Sampler thread started")
    first_run = True
    while session.running:
        start = now_ist()
        try:
            if first_run:
                session.start_warmup()
                first_run = False

            process_snapshot(session, fyers, execute_trades=auto_trade)

            # Calculate total P&L
            total_pnl = session.paper_realized_pnl
            for pos in session.active_positions:
                current = pos.get("current_premium", pos["entry_premium"])
                if current and pos.get("entry_premium"):
                    total_pnl += (current - pos["entry_premium"]) * Config.LOT_SIZE

            # Update current premium for active position
            with session.lock:
                for pos in session.active_positions:
                    current_premium = session.get_ltp(pos["symbol"])
                    if current_premium is None:
                        for opt in session.option_chain:
                            if opt["symbol"] == pos["symbol"]:
                                current_premium = opt.get("ltp", 0)
                                break
                    pos["current_premium"] = current_premium if current_premium else pos["entry_premium"]

            # Log data
            data_logger.log_data(
                timestamp=timestamp_ist(),
                ltp=session.current_ltp,
                straddle=session.current_straddle,
                rolling_max_ltp=session.rolling_max_ltp,
                rolling_min_ltp=session.rolling_min_ltp,
                rolling_max_straddle=session.rolling_max_straddle,
                active_positions=session.active_positions,
                total_pnl=total_pnl,
                daily_entries=session.daily_entries,
                bars_collected=session.bars_collected,
                decision_reason=session.decision_reason
            )

            elapsed = (now_ist() - start).total_seconds()
            if elapsed < Config.SAMPLE_INTERVAL:
                time.sleep(Config.SAMPLE_INTERVAL - elapsed)

        except Exception as e:
            log_debug(f"Sampler error: {e}\n{traceback.format_exc()}")
            time.sleep(1)

# ============================================================
# DASHBOARD (now shows symbol data counts)
# ============================================================
class TerminalDashboard:
    def __init__(self, data_logger, session_state):
        self.data_logger = data_logger
        self.session_state = session_state

    def display(self):
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            mode = "PAPER" if PAPER_MODE else "LIVE"
            print("\n" + "="*90)
            print(f"📊 {mode} TRADING DASHBOARD – Straddle Breakout + Trailing Stop")
            print("="*90)

            with self.session_state.lock:
                daily_entries = self.session_state.daily_entries
                vix = self.session_state.india_vix
                realized = self.session_state.paper_realized_pnl
                decision_reason = self.session_state.decision_reason
                bars_collected = self.session_state.bars_collected
                warmup = self.session_state.is_warming_up()
                warmup_rem = self.session_state.warmup_remaining()

                ltp = self.session_state.current_ltp
                straddle = self.session_state.current_straddle
                rolling_max_ltp = self.session_state.rolling_max_ltp
                rolling_min_ltp = self.session_state.rolling_min_ltp
                rolling_max_straddle = self.session_state.rolling_max_straddle

                active_positions = self.session_state.active_positions[:]
                pending_signal = self.session_state.pending_signal.copy() if self.session_state.pending_signal else None
                last_data = self.session_state.last_data_time
                last_data_str = datetime.fromtimestamp(last_data).strftime("%H:%M:%S") if last_data else "No data yet"

                # Symbol data counts
                total_symbols, active_symbols = self.session_state.get_symbol_data_counts(recency_seconds=5)
                stale_symbols = self.session_state.get_stale_symbols(recency_seconds=5)

            window_min = Config.WINDOW_SECONDS / 60.0
            window_label = f"{int(window_min)} min" if window_min == int(window_min) else f"{Config.WINDOW_SECONDS}s"

            if warmup:
                status = f"🔥 WARMUP [{bars_collected}/{Config.REQUIRED_BARS}]"
            else:
                status = f"✅ ACTIVE [{bars_collected}/{Config.REQUIRED_BARS}]"

            pos_str = "NEUTRAL"
            if active_positions:
                pos = active_positions[0]
                entry_type = pos.get('entry_type', '?')
                offset = pos.get('trailing_offset', 0)
                entry = pos.get('entry_premium', 0)
                current = pos.get('current_premium', entry)
                pnl = ((current - entry) * Config.LOT_SIZE) if current and entry else 0

                if entry_type == 'CE':
                    trail_high = pos.get('trail_high', pos.get('entry_ltp', ltp))
                    stop = trail_high - offset
                else:
                    trail_low = pos.get('trail_low', pos.get('entry_ltp', ltp))
                    stop = trail_low + offset
                pos_str = (f"{entry_type} | Entry={entry:.2f} | Curr={current:.2f} | "
                           f"Stop={stop:.2f} | P&L={pnl:+.2f}")

            print(f"Last Update: {timestamp_ist()}   |   WS Last Data: {last_data_str}")
            print(f"📡 Symbol data: total ever seen = {total_symbols}, active (≤5s) = {active_symbols}")
            if stale_symbols:
                shown = ', '.join(stale_symbols[:5])
                more = f" (+{len(stale_symbols)-5} more)" if len(stale_symbols) > 5 else ""
                print(f"   ⚠️ Stale >5s: {shown}{more}")
            print(f"📈 Daily Entries: {daily_entries}/{Config.MAX_DAILY_ENTRIES}")
            print(f"📊 India VIX: {vix:.2f}   |   💰 Realized P&L: {realized:+.2f}")

            if warmup:
                print(f"⏳ WARMING UP... {warmup_rem} bars remaining")
            else:
                print("✅ WARMUP COMPLETE – Trading active")

            print("\n" + "-"*90)
            print("📊 SNAPSHOT DATA")
            print("-"*90)
            print(f"  LTP: {ltp:.2f}   |   Straddle: {straddle:.2f}")
            print(f"  Rolling LTP      → Max: {rolling_max_ltp:.2f}   Min: {rolling_min_ltp:.2f}")
            print(f"  Rolling Straddle → Max: {rolling_max_straddle:.2f}")

            print("\n" + "-"*90)
            print("📌 POSITION & TRAILING STOP")
            print("-"*90)
            if active_positions:
                print(f"  Symbol: {pos.get('symbol','?')}")
                print(f"  {pos_str}")
                print(f"  Offset: {offset:.2f} (dynamic, recalculated every bar)")
                if entry_type == 'CE':
                    print(f"  Trail High: {trail_high:.2f} | Stop: {trail_high:.2f} – {offset:.2f} = {stop:.2f}")
                else:
                    print(f"  Trail Low: {trail_low:.2f} | Stop: {trail_low:.2f} + {offset:.2f} = {stop:.2f}")
            else:
                if pending_signal:
                    elapsed = time.time() - pending_signal["signal_time"]
                    ref = pending_signal["reference_price"]
                    pct = Config.ENTRY_PRICE_RANGE_PERCENT / 100.0
                    print(f"  WAITING: {pending_signal['entry_type']} | Ref interp={ref:.2f} | "
                          f"Range={ref*(1-pct):.2f}-{ref*(1+pct):.2f} | "
                          f"Elapsed={elapsed:.1f}s/{Config.ENTRY_WAIT_TIMEOUT_SECONDS}s")
                else:
                    print("  No active position (FLAT)")

            print("\n" + "-"*90)
            print("🧠 STATUS")
            print("-"*90)
            print(f"  {status}")
            print(f"  Position: {pos_str}")
            print(f"  Decision: {decision_reason}")

            print("\n" + "-"*90)
            print("⚙️ CONFIGURATION")
            print("-"*90)
            print(f"  Rolling Window: {window_label}")
            print(f"  Entry rules: LTP breakout + Straddle breakout (asymmetric for sell)")
            print(f"  Trailing Stop: Anchored to extreme – offset recalculated every bar from current rolling range")
            print(f"  Min Trailing Distance: {Config.MIN_TRAILING_DISTANCE}")
            print(f"  Lot Size: {Config.LOT_SIZE}")

            print("\n" + "="*90)
        except Exception as e:
            log_debug(f"Dashboard error: {e}")

# ============================================================
# HEARTBEAT MONITOR (FIXED – now checks timestamp, not just value)
# ============================================================
def heartbeat_monitor(session, ws_manager, index_symbol):
    while session.running:
        time.sleep(Config.WS_HEARTBEAT_TIMEOUT // 2)

        # NIFTY itself is the most liquid symbol on the feed — if IT has gone stale,
        # the whole connection is almost certainly down, so do a full restart.
        last_ts = session.get_ltp_time(index_symbol)
        if last_ts is not None and (time.time() - last_ts) > Config.WS_HEARTBEAT_TIMEOUT:
            log_console(f"⚠️ NIFTY LTP stale for >{Config.WS_HEARTBEAT_TIMEOUT}s, forcing WS restart...")
            ws_manager.stop()
            time.sleep(2)
            ws_manager.start()
            # Re-subscribe all known symbols
            if ws_manager.subscribed:
                ws_manager.subscribe(list(ws_manager.subscribed))
            continue  # give the fresh connection a moment before judging individual symbols

        # Everything else (VIX + current option legs): previously nothing checked these
        # at all, so a symbol whose feed died — or whose subscribe request silently
        # never reached the server — would just sit frozen forever. Catch both cases
        # here and force a targeted resubscribe instead of restarting the whole
        # connection (which would needlessly interrupt every symbol that IS fine).
        now = time.time()
        stale = set()
        for sym in ws_manager.subscribed:
            if sym == index_symbol:
                continue
            ts = session.get_ltp_time(sym)
            if ts is not None:
                if (now - ts) > Config.WS_HEARTBEAT_TIMEOUT:
                    stale.add(sym)
            else:
                subscribed_at = ws_manager.subscribed_at.get(sym)
                if subscribed_at is not None and (now - subscribed_at) > Config.WS_HEARTBEAT_TIMEOUT:
                    stale.add(sym)  # never produced a single tick since we subscribed

        if stale:
            log_console(f"⚠️ {len(stale)} symbol(s) stale/never-arrived, resubscribing: {sorted(stale)}")
            ws_manager.resubscribe(stale)

# ============================================================
# CREDENTIALS & MAIN
# ============================================================
def load_credentials(session):
    try:
        with open('fyers_credentials.json', 'r') as f:
            creds = json.load(f)
            session.cid = creds.get('client_id', '')
            session.token = creds.get('access_token', '')
        log_console(f"Loaded credentials: {session.cid}")
    except Exception as e:
        log_console(f"Failed to load credentials file: {e} – falling back to env vars")
        session.cid = os.getenv("FYERS_CID", "")
        session.token = os.getenv("FYERS_TOKEN", "")
        log_console(f"Using env vars: {session.cid}")

def main():
    log_console("Starting Straddle + Price Breakout Strategy (Corrected Trailing Stop)...")
    session = SessionState()
    load_credentials(session)
    data_logger = DataLogger()
    dashboard = TerminalDashboard(data_logger, session)
    index_symbol = "NSE:NIFTY50-INDEX"
    vix_symbol = "NSE:INDIAVIX-INDEX"

    poller_thread = None
    sampler_thread = None
    heartbeat_thread = None
    ws_manager = None
    fyers = None

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*50)
        mode = "PAPER" if PAPER_MODE else "LIVE"
        print(f"📈 FYERS ALGO TRADING ({mode} MODE)")
        print("1. Start Trading")
        print("2. Settings")
        print("3. Exit")
        choice = input("Select option: ").strip()
        if choice == "1":
            if not (session.cid and session.token):
                print("❌ Set credentials first")
                input("Press Enter...")
                continue

            session.reset_for_new_day()

            auto_trade = input("Enable Auto Trade? (y/n): ").lower() == 'y'
            log_console(f"Auto trade: {'ON' if auto_trade else 'OFF'}")

            ws_manager = MarketWSManager(session, session.token)
            ws_manager.start()
            ws_manager.subscribe([index_symbol, vix_symbol])
            session.running = True

            fyers = fyersModel.FyersModel(client_id=session.cid, token=session.token, is_async=False, log_path="")
            poller_thread = threading.Thread(target=chain_poller, args=(session, ws_manager, index_symbol), daemon=True)
            sampler_thread = threading.Thread(target=sampler, args=(session, data_logger, auto_trade, fyers), daemon=True)
            # Updated heartbeat thread with index_symbol argument
            heartbeat_thread = threading.Thread(target=heartbeat_monitor, args=(session, ws_manager, index_symbol), daemon=True)

            poller_thread.start()
            sampler_thread.start()
            heartbeat_thread.start()

            print("\n✅ Trading started. Press Ctrl+C to stop.\n")
            try:
                while True:
                    dashboard.display()
                    time.sleep(1)
            except KeyboardInterrupt:
                log_console("\n🛑 Stopping trading...")
                session.running = False
                ws_manager.stop()

                if sampler_thread is not None:
                    sampler_thread.join(timeout=5)
                if poller_thread is not None:
                    poller_thread.join(timeout=Config.OPTION_CHAIN_INTERVAL + 5)
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=2)

                if session.active_positions:
                    log_console(f"Attempting to close {len(session.active_positions)} open position(s)...")
                    process_snapshot(session, fyers, execute_trades=True)

                with session.lock:
                    if session.active_positions:
                        log_console(f"⚠️ {len(session.active_positions)} position(s) could not be closed. Check broker manually.")
                    else:
                        session.eod_square_off_triggered = False
                        session.daily_entries = 0
                        session.paper_realized_pnl = 0.0
                        session.warmup_start_time = None
                        session.sold_time.clear()
                        log_console("✅ Trading stopped and state reset. You can start again.")
        elif choice == "2":
            print("\n⚙️ SETTINGS")
            new_cid = input(f"Client ID [{session.cid}]: ").strip()
            if new_cid:
                session.cid = new_cid
            new_token = input("Access token: ").strip()
            if new_token:
                session.token = new_token
            with open('fyers_credentials.json', 'w') as f:
                json.dump({"client_id": session.cid, "access_token": session.token}, f)
            print("✅ Saved")
            input("Press Enter...")
        elif choice == "3":
            if session.running:
                session.running = False
                if ws_manager:
                    ws_manager.stop()
            if session.active_positions:
                log_console(f"⚠️ Exiting with {len(session.active_positions)} position(s) still tracked.")
            break

if __name__ == "__main__":
    main()