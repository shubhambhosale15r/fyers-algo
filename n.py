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
PAPER_MODE = True

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    SAMPLE_INTERVAL = 1
    OPTION_CHAIN_INTERVAL = 30

    ATM_RANGE = 2
    STRIKE_SPAN = 1

    LOT_SIZE = 195
    MAX_DAILY_ENTRIES = 100
    API_RATE_LIMIT = 200
    API_WINDOW = 60
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    WS_HEARTBEAT_TIMEOUT = 10

    # ---- R-BASED STRATEGY PARAMETERS ----
    R_ENTRY_THRESHOLD = 0.786
    R_TRAIL_OFFSET    = 0.168
    R_TARGET_EXIT     = 1.0     # NEW: exit held position when its own R >= this

    # [C1] prev_R older than this is treated as "no previous" for signal purposes
    R_PREV_MAX_AGE = 5

    # [C2] force-exit if the held option's own feed is silent this long (live mode)
    HOLD_STALE_FORCE_EXIT_SECONDS = 30

    # [C4] don't re-enter the same symbol this soon after an exit
    REENTRY_COOLDOWN_SECONDS = 0 #30

    OPTION_PRICE_STALE_SECONDS = 5

    TICK_CSV = "trading_data.csv"
    TRADE_CSV = "trades.csv"

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
# TICK LOGGER  (one row per snapshot)
# ============================================================
class TickLogger:
    COLUMNS = [
        'timestamp', 'index_ltp', 'vix', 'atm_strike',
        'atm_ce_sym', 'atm_ce_ltp', 'atm_ce_high', 'atm_ce_low',
        'atm_ce_R', 'atm_ce_prevR',
        'atm_pe_sym', 'atm_pe_ltp', 'atm_pe_high', 'atm_pe_low',
        'atm_pe_R', 'atm_pe_prevR',
        'neigh_m1_ce_R', 'neigh_m1_pe_R',
        'neigh_p1_ce_R', 'neigh_p1_pe_R',
        'held_symbol', 'held_type', 'held_strike',
        'held_entry_premium', 'held_current_premium',
        'held_entry_R', 'held_current_R', 'held_highest_R', 'held_trail_sl_R',
        'held_entry_spot', 'held_current_spot', 'held_unrealized_pnl',
        'realized_pnl', 'total_pnl', 'daily_entries',
        'decision_reason'
    ]

    def __init__(self, filename=None):
        self.filename = filename or Config.TICK_CSV
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                csv.writer(f).writerow(self.COLUMNS)

    @staticmethod
    def _fmt(v, digits=4):
        if v is None:
            return ''
        try:
            return round(float(v), digits)
        except (TypeError, ValueError):
            return ''

    def log(self, **kw):
        row = [
            kw.get('timestamp', ''),
            self._fmt(kw.get('index_ltp'), 2),
            self._fmt(kw.get('vix'), 2),
            kw.get('atm_strike') or '',
            kw.get('atm_ce_sym', '') or '',
            self._fmt(kw.get('atm_ce_ltp'), 2),
            self._fmt(kw.get('atm_ce_high'), 2),
            self._fmt(kw.get('atm_ce_low'), 2),
            self._fmt(kw.get('atm_ce_R')),
            self._fmt(kw.get('atm_ce_prevR')),
            kw.get('atm_pe_sym', '') or '',
            self._fmt(kw.get('atm_pe_ltp'), 2),
            self._fmt(kw.get('atm_pe_high'), 2),
            self._fmt(kw.get('atm_pe_low'), 2),
            self._fmt(kw.get('atm_pe_R')),
            self._fmt(kw.get('atm_pe_prevR')),
            self._fmt(kw.get('neigh_m1_ce_R')),
            self._fmt(kw.get('neigh_m1_pe_R')),
            self._fmt(kw.get('neigh_p1_ce_R')),
            self._fmt(kw.get('neigh_p1_pe_R')),
            kw.get('held_symbol', '') or '',
            kw.get('held_type', '') or '',
            kw.get('held_strike') or '',
            self._fmt(kw.get('held_entry_premium'), 2),
            self._fmt(kw.get('held_current_premium'), 2),
            self._fmt(kw.get('held_entry_R')),
            self._fmt(kw.get('held_current_R')),
            self._fmt(kw.get('held_highest_R')),
            self._fmt(kw.get('held_trail_sl_R')),
            self._fmt(kw.get('held_entry_spot'), 2),
            self._fmt(kw.get('held_current_spot'), 2),
            self._fmt(kw.get('held_unrealized_pnl'), 2),
            self._fmt(kw.get('realized_pnl'), 2),
            self._fmt(kw.get('total_pnl'), 2),
            kw.get('daily_entries', 0),
            kw.get('decision_reason', '')
        ]
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow(row)

# ============================================================
# TRADE LOGGER  (one row per ENTRY / EXIT / FAILURE event)
# ============================================================
class TradeLogger:
    COLUMNS = [
        'timestamp', 'event', 'symbol', 'option_type', 'strike',
        'premium', 'R', 'spot', 'highest_R', 'trail_sl_R',
        'pnl', 'cumulative_pnl', 'daily_entry_num',
        'paper_mode', 'order_id', 'notes'
    ]

    def __init__(self, filename=None, max_memory=30):
        self.filename = filename or Config.TRADE_CSV
        self.recent = deque(maxlen=max_memory)
        self._lock = threading.Lock()          # [M4] protect recent list
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                csv.writer(f).writerow(self.COLUMNS)

    @staticmethod
    def _fmt(v, digits=4):
        if v is None:
            return ''
        try:
            return round(float(v), digits)
        except (TypeError, ValueError):
            return ''

    def _write(self, row):
        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        with self._lock:                       # [M4]
            self.recent.append(dict(zip(self.COLUMNS, row)))

    def snapshot_recent(self):                 # [M4] read snapshot safely
        with self._lock:
            return list(self.recent)

    def log_entry(self, pos, cumulative_pnl, order_id='', notes=''):
        row = [
            timestamp_ist(), 'ENTRY',
            pos.get('symbol', ''),
            pos.get('entry_type', ''),
            pos.get('strike') or '',
            self._fmt(pos.get('entry_premium'), 2),
            self._fmt(pos.get('entry_R')),
            self._fmt(pos.get('entry_spot'), 2),
            self._fmt(pos.get('highest_R')),
            self._fmt(pos.get('trailing_sl_R')),
            self._fmt(0.0, 2),
            self._fmt(cumulative_pnl, 2),
            pos.get('daily_entry_num', ''),
            'PAPER' if PAPER_MODE else 'LIVE',
            order_id or '',
            notes or 'ENTRY'
        ]
        self._write(row)

    def log_exit(self, pos, exit_premium, exit_R, exit_spot,
                 exit_reason, pnl, cumulative_pnl,
                 order_id='', notes=''):
        row = [
            timestamp_ist(), 'EXIT',
            pos.get('symbol', ''),
            pos.get('entry_type', ''),
            pos.get('strike') or '',
            self._fmt(exit_premium, 2),
            self._fmt(exit_R),
            self._fmt(exit_spot, 2),
            self._fmt(pos.get('highest_R')),
            self._fmt(pos.get('trailing_sl_R')),
            self._fmt(pnl, 2),
            self._fmt(cumulative_pnl, 2),
            pos.get('daily_entry_num', ''),
            'PAPER' if PAPER_MODE else 'LIVE',
            order_id or '',
            f"{exit_reason} | entry_R={self._fmt(pos.get('entry_R'))} | {notes}"
        ]
        self._write(row)

    # [B3] failures now land in trades.csv too
    def log_failure(self, event, symbol='', option_type='', strike='',
                    premium=None, R=None, spot=None,
                    cumulative_pnl=None, order_id='', notes='',
                    position=None):
        row = [
            timestamp_ist(), event,
            symbol or '',
            option_type or '',
            strike or '',
            self._fmt(premium, 2),
            self._fmt(R),
            self._fmt(spot, 2),
            self._fmt(position.get('highest_R')) if position else '',
            self._fmt(position.get('trailing_sl_R')) if position else '',
            self._fmt(0.0, 2),
            self._fmt(cumulative_pnl, 2) if cumulative_pnl is not None else '',
            position.get('daily_entry_num', '') if position else '',
            'PAPER' if PAPER_MODE else 'LIVE',
            order_id or '',
            notes or event
        ]
        self._write(row)

# ============================================================
# SESSION STATE
# ============================================================
class SessionState:
    def __init__(self):
        self.lock = threading.RLock()
        self.option_chain_api_count = 0
        self.api_window_start = now_ist()
        self.cid = ""
        self.token = ""
        self.daily_entries = 0
        self.eod_square_off_triggered = False

        # [C7] symbol_histories removed — was dead weight

        self.live_ltp = {}
        self.live_ltp_time = {}
        self.live_high = {}
        self.live_low = {}
        self.live_vwap = {}
        self.live_volume = {}

        self.option_chain = []
        self.fut_price = None
        self.expiry_date = None
        self.atm_strike = None
        self.option_symbols = []
        self.strike_to_symbols = {}
        self.symbol_to_strike = {}

        self.active_positions = []
        self.india_vix = 15.0
        self.paper_realized_pnl = 0.0

        self.decision_reason = ""
        self.running = False
        self._order_lock = False

        self.current_ltp = 0.0
        self.current_ce_R = None
        self.current_pe_R = None
        self.current_ce_prev_R = None
        self.current_pe_prev_R = None

        self.last_R_snapshot = {}
        self.last_R_snapshot_time = None

        # [C1] prev_R now maps symbol -> (R_value, timestamp)
        self.prev_R = {}

        # [C4] symbol -> last exit time (for cooldown)
        self.recent_exits = {}

        self.last_data_time = None

        # [C6] explicitly declare both loggers
        self.tick_logger = None
        self.trade_logger = None

    # -------------------- WS ingest --------------------
    def update_ws(self, symbol, ltp, vwap, volume, high=None, low=None):
        with self.lock:
            self.last_data_time = time.time()
            if ltp is not None:
                self.live_ltp[symbol] = float(ltp)
                self.live_ltp_time[symbol] = time.time()
            if high is not None:
                try:
                    self.live_high[symbol] = float(high)
                except (TypeError, ValueError):
                    pass
            if low is not None:
                try:
                    self.live_low[symbol] = float(low)
                except (TypeError, ValueError):
                    pass
            if symbol == "NSE:INDIAVIX-INDEX":
                if ltp is not None:
                    self.india_vix = float(ltp)
                return
            if vwap is not None:
                self.live_vwap[symbol] = float(vwap)
            if volume is not None:
                self.live_volume[symbol] = int(volume)

    # -------------------- accessors --------------------
    def get_ltp(self, symbol):
        with self.lock:
            return self.live_ltp.get(symbol)

    def get_ltp_time(self, symbol):
        with self.lock:
            return self.live_ltp_time.get(symbol)

    def get_high(self, symbol):
        with self.lock:
            return self.live_high.get(symbol)

    def get_low(self, symbol):
        with self.lock:
            return self.live_low.get(symbol)

    def get_symbol_data_counts(self, recency_seconds=5):
        with self.lock:
            total = len(self.live_ltp_time)
            now = time.time()
            recent = sum(1 for t in self.live_ltp_time.values() if now - t <= recency_seconds)
            return total, recent

    # [C9] only report staleness for symbols we actually care about
    def get_watched_symbols(self):
        with self.lock:
            watched = {"NSE:NIFTY50-INDEX", "NSE:INDIAVIX-INDEX"}
            for k in self.get_tracked_strikes():
                pair = self.strike_to_symbols.get(k)
                if pair:
                    for s in pair:
                        if s:
                            watched.add(s)
            return watched

    def get_stale_symbols(self, recency_seconds=5):
        with self.lock:
            now = time.time()
            watched = self.get_watched_symbols()
            return sorted(
                s for s in watched
                if s in self.live_ltp_time and (now - self.live_ltp_time[s]) > recency_seconds
            )

    # [C1] fresh prev_R (returns None if older than R_PREV_MAX_AGE)
    def get_fresh_prev_R(self, symbol):
        entry = self.prev_R.get(symbol)
        if entry is None:
            return None
        R, ts = entry
        if (time.time() - ts) > Config.R_PREV_MAX_AGE:
            return None
        return R

    # -------------------- chain --------------------
    def update_chain(self, chain, fut_price, expiry_date, atm_strike, option_symbols):
        with self.lock:
            self.option_chain = chain
            self.fut_price = float(fut_price) if fut_price else 0.0
            self.expiry_date = expiry_date
            self.atm_strike = int(atm_strike) if atm_strike else None
            self.option_symbols = option_symbols

            self.strike_to_symbols = {}
            self.symbol_to_strike = {}
            for opt in chain:
                strike = opt.get("strike_price")
                if strike is None:
                    continue
                ot = opt.get("option_type")
                sym = opt.get("symbol")
                if not sym:
                    continue
                self.symbol_to_strike[sym] = (strike, ot)
                if ot == "CE":
                    if strike not in self.strike_to_symbols:
                        self.strike_to_symbols[strike] = [None, None]
                    self.strike_to_symbols[strike][0] = sym
                elif ot == "PE":
                    if strike not in self.strike_to_symbols:
                        self.strike_to_symbols[strike] = [None, None]
                    self.strike_to_symbols[strike][1] = sym

    def get_tracked_strikes(self):
        with self.lock:
            if self.atm_strike is None or not self.strike_to_symbols:
                return []
            all_strikes = sorted(self.strike_to_symbols.keys())
            if self.atm_strike not in all_strikes:
                return []
            idx = all_strikes.index(self.atm_strike)
            lo = max(0, idx - Config.STRIKE_SPAN)
            hi = min(len(all_strikes), idx + Config.STRIKE_SPAN + 1)
            return all_strikes[lo:hi]

    def update_all_R_values(self):
        results = {}
        for strike in self.get_tracked_strikes():
            pair = self.strike_to_symbols.get(strike)
            if not pair:
                continue
            for ot, sym in (("CE", pair[0]), ("PE", pair[1])):
                if not sym:
                    continue
                R = compute_R(self, sym)
                if R is None:
                    continue
                # [C1] read fresh prev_R BEFORE overwriting, then store (R, ts)
                prev_R = self.get_fresh_prev_R(sym)
                self.prev_R[sym] = (R, time.time())
                results[(strike, ot)] = {
                    "symbol": sym,
                    "strike": strike,
                    "R": R,
                    "prev_R": prev_R,
                    "ltp": self.get_ltp(sym),
                    "high": self.get_high(sym),
                    "low": self.get_low(sym),
                }
        return results

    # -------------------- reset --------------------
    def reset_for_new_day(self):
        with self.lock:
            if self.active_positions:
                log_console(f"⚠️ {len(self.active_positions)} position(s) still tracked, not resetting.")
                return
            self.daily_entries = 0
            self.eod_square_off_triggered = False
            self.paper_realized_pnl = 0.0
            self.decision_reason = ""
            self.active_positions.clear()
            self.last_data_time = None
            self.live_ltp.clear()
            self.live_ltp_time.clear()
            self.live_high.clear()
            self.live_low.clear()
            self.live_vwap.clear()
            self.live_volume.clear()
            self.current_ce_R = None
            self.current_pe_R = None
            self.current_ce_prev_R = None
            self.current_pe_prev_R = None
            self.prev_R.clear()
            self.recent_exits.clear()
            self.last_R_snapshot = {}
            self.last_R_snapshot_time = None

# ============================================================
# WEBSOCKET MANAGER
# ============================================================
class MarketWSManager:
    def __init__(self, session_state, access_token):
        self.session = session_state
        self.access_token = access_token
        self.ws = None
        self.subscribed = set()
        self.subscribed_at = {}
        self.running = False
        self.ws_thread = None

    def on_message(self, message):
        symbol = message.get("symbol")
        if symbol:
            self.session.update_ws(
                symbol,
                message.get("ltp"),
                message.get("avg_trade_price"),
                message.get("vol_traded_today"),
                high=message.get("high_price"),
                low=message.get("low_price")
            )

    def on_error(self, msg):
        log_console(f"WS error: {msg}")

    def on_close(self, msg):
        log_console(f"WS closed: {msg}. Automatic reconnection is enabled.")

    def on_open(self):
        log_console("✅ WebSocket on_open() triggered")
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
                log_console(f"⚠️ Subscribe request failed for {sorted(new_symbols)}: {e}")
                log_debug(traceback.format_exc())
        else:
            log_debug(f"subscribe() called before WS was ready; on_open(): {sorted(new_symbols)}")
        self.subscribed.update(new_symbols)
        for s in new_symbols:
            self.subscribed_at.setdefault(s, now)

    def resubscribe(self, symbols):
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
# REST API
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

    payload = {"symbol": index_symbol, "strikecount": Config.ATM_RANGE,
               "timestamp": expiry_ts, "greeks": "1"}
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
# CHAIN POLLER  (uses futures price for ATM selection)  [C5]
# ============================================================
def chain_poller(session, ws_manager, index_symbol):
    while session.running:
        try:
            chain, fut_price, expiry_date, _, opt_symbols = fetch_option_chain(session, index_symbol)
            if chain is not None:
                # [C5] Prefer futures price — NIFTY options strike on futures,
                # not spot index. Fall back to index LTP if futures unavailable.
                if fut_price and fut_price > 0:
                    underlying_ref = float(fut_price)
                    ref_src = "futures"
                else:
                    underlying_ref = session.get_ltp(index_symbol)
                    ref_src = "spot"

                strikes = sorted(set(opt["strike_price"] for opt in chain
                                     if opt.get("option_type") in ("CE","PE")))
                if strikes and underlying_ref and underlying_ref > 0:
                    atm = min(strikes, key=lambda x: abs(x - underlying_ref))
                else:
                    atm = None

                session.update_chain(chain, fut_price, expiry_date, atm, opt_symbols)
                log_debug(f"Chain updated | ATM={atm} (ref={underlying_ref}, src={ref_src})")

                all_syms = [index_symbol, "NSE:INDIAVIX-INDEX"] + opt_symbols
                ws_manager.subscribe(all_syms)
            time.sleep(Config.OPTION_CHAIN_INTERVAL)
        except Exception as e:
            log_debug(f"Chain poller error: {e}\n{traceback.format_exc()}")
            time.sleep(Config.OPTION_CHAIN_INTERVAL)

# ============================================================
# R CALCULATION
# ============================================================
def compute_R(session, symbol):
    """R = (ltp - day_low) / (day_high - day_low) for the given option symbol."""
    ltp = session.get_ltp(symbol)
    high = session.get_high(symbol)
    low = session.get_low(symbol)
    ts = session.get_ltp_time(symbol)
    if ltp is None or high is None or low is None:
        return None
    if ts is None or (time.time() - ts) > Config.OPTION_PRICE_STALE_SECONDS:
        return None
    rng = high - low
    if rng <= 0:
        return 0.0
    return (ltp - low) / rng

# ============================================================
# POSITION CREATION
# ============================================================
def create_position(session, symbol, entry_premium, entry_type, entry_R, spot,
                    order_id=''):
    strike = None
    info = session.symbol_to_strike.get(symbol)
    if info:
        strike = info[0]

    pos = {
        "symbol": symbol,
        "entry_premium": float(entry_premium),
        "entry_type": entry_type,
        "strike": strike,
        "entry_R": float(entry_R),
        "highest_R": float(entry_R),
        "current_R": float(entry_R),
        "trailing_sl_R": float(entry_R) - Config.R_TRAIL_OFFSET,
        "current_premium": float(entry_premium),
        "entry_spot": float(spot),
        "daily_entry_num": session.daily_entries + 1,
        "order_id": order_id or '',
    }
    session.active_positions.append(pos)
    session.daily_entries += 1

    if session.trade_logger:
        session.trade_logger.log_entry(pos, session.paper_realized_pnl,
                                       order_id=order_id)

    log_console(
        f"🎯 ENTERED {entry_type} {symbol} @ {entry_premium:.2f} | "
        f"Entry R={entry_R:.4f} | Initial Trail SL R={pos['trailing_sl_R']:.4f}"
    )
    return pos

# ============================================================
# STATE MACHINE
# ============================================================
def process_snapshot(session, fyers, execute_trades=False):
    if getattr(session, '_order_lock', False):
        return
    session._order_lock = True
    try:
        with session.lock:
            spot = session.get_ltp("NSE:NIFTY50-INDEX")
            if spot is None or spot <= 0:
                session.decision_reason = "No index LTP"
                return
            index_ts = session.get_ltp_time("NSE:NIFTY50-INDEX")
            if index_ts is None or (time.time() - index_ts) > Config.OPTION_PRICE_STALE_SECONDS:
                session.decision_reason = "Index LTP stale; skipping"
                return

            session.current_ltp = float(spot)

            # [C8] reset current R fields; only set below if we have fresh data
            session.current_ce_R = None
            session.current_pe_R = None
            session.current_ce_prev_R = None
            session.current_pe_prev_R = None

            all_R = session.update_all_R_values()
            session.last_R_snapshot = all_R
            session.last_R_snapshot_time = time.time()

            atm = session.atm_strike
            atm_ce = all_R.get((atm, "CE"))
            atm_pe = all_R.get((atm, "PE"))

            if atm_ce is not None:
                session.current_ce_R = atm_ce["R"]
                session.current_ce_prev_R = atm_ce["prev_R"]
            if atm_pe is not None:
                session.current_pe_R = atm_pe["R"]
                session.current_pe_prev_R = atm_pe["prev_R"]

            # ---------------- ACTIVE POSITION ----------------
            if session.active_positions:
                pos = session.active_positions[0]
                sym = pos["symbol"]

                # [C2] safety: if the held option's feed itself is dead, force close
                held_ltp_ts = session.get_ltp_time(sym)
                if held_ltp_ts is not None and \
                   (time.time() - held_ltp_ts) > Config.HOLD_STALE_FORCE_EXIT_SECONDS:
                    session.decision_reason = (
                        f"Held {sym} stale for "
                        f"{time.time()-held_ltp_ts:.0f}s → FORCE EXIT"
                    )
                    log_console(
                        f"🚨 {sym} feed stale for "
                        f"{time.time()-held_ltp_ts:.0f}s → forcing exit"
                    )
                    if execute_trades:
                        _exit_position(session, fyers, pos,
                                       reason="HOLD_FEED_STALE",
                                       extra=f"stale={time.time()-held_ltp_ts:.0f}s",
                                       current_R=None, exit_spot=spot)
                    return

                current_R = compute_R(session, sym)
                if current_R is None:
                    session.decision_reason = f"Holding {pos['entry_type']} | R unavailable/stale"
                    return

                cur_prem = session.get_ltp(sym)
                if cur_prem is not None and cur_prem > 0:
                    pos["current_premium"] = float(cur_prem)

                pos["current_R"] = float(current_R)
                highest_R = max(pos.get("highest_R", pos["entry_R"]), current_R)
                pos["highest_R"] = float(highest_R)
                trail_sl_R = highest_R - Config.R_TRAIL_OFFSET
                pos["trailing_sl_R"] = float(trail_sl_R)

                # ============================================================
                # NEW: EXIT CHECK 1 — hard target (held option's own R >= 1)
                # Runs BEFORE the trailing-SL check so whichever triggers first
                # in the R sequence wins.
                # ============================================================
                if current_R >= Config.R_TARGET_EXIT:
                    session.decision_reason = (
                        f"Holding {pos['entry_type']} | EXIT=R_TARGET | "
                        f"R={current_R:.4f} >= {Config.R_TARGET_EXIT}"
                    )
                    if not execute_trades:
                        return
                    _exit_position(session, fyers, pos,
                                   reason="R_TARGET",
                                   extra=(f"R={current_R:.4f} target={Config.R_TARGET_EXIT} "
                                          f"HighR={highest_R:.4f}"),
                                   current_R=current_R,
                                   exit_spot=spot)
                    return

                # ============================================================
                # EXISTING: EXIT CHECK 2 — trailing SL
                # ============================================================
                if current_R <= trail_sl_R:
                    session.decision_reason = (
                        f"Holding {pos['entry_type']} | EXIT=TrailingR | "
                        f"R={current_R:.4f} <= SL_R={trail_sl_R:.4f}"
                    )
                    if not execute_trades:
                        return
                    _exit_position(session, fyers, pos,
                                   reason="TRAILING_R",
                                   extra=(f"R={current_R:.4f} SL_R={trail_sl_R:.4f} "
                                          f"HighR={highest_R:.4f}"),
                                   current_R=current_R,
                                   exit_spot=spot)
                    return
                else:
                    session.decision_reason = (
                        f"Holding {pos['entry_type']} {sym} | "
                        f"R={current_R:.4f} | HighR={highest_R:.4f} | "
                        f"TrailSL_R={trail_sl_R:.4f}"
                    )
                    return

            # ---------------- FLAT: check entry ----------------
            if session.daily_entries >= Config.MAX_DAILY_ENTRIES:
                session.decision_reason = "Max daily entries reached"
                return

            if atm_ce is None or atm_pe is None:
                session.decision_reason = "ATM CE/PE data unavailable or stale"
                return

            ce_trigger = (
                atm_ce["prev_R"] is not None
                and atm_ce["prev_R"] < Config.R_ENTRY_THRESHOLD
                and atm_ce["R"] > Config.R_ENTRY_THRESHOLD
            )
            pe_trigger = (
                atm_pe["prev_R"] is not None
                and atm_pe["prev_R"] < Config.R_ENTRY_THRESHOLD
                and atm_pe["R"] > Config.R_ENTRY_THRESHOLD
            )

            if not ce_trigger and not pe_trigger:
                session.decision_reason = (
                    f"IDLE | ATM={atm} | "
                    f"CE R={atm_ce['R']:.4f} (prev="
                    f"{atm_ce['prev_R'] if atm_ce['prev_R'] is not None else 'n/a'}) | "
                    f"PE R={atm_pe['R']:.4f} (prev="
                    f"{atm_pe['prev_R'] if atm_pe['prev_R'] is not None else 'n/a'}) | "
                    f"thr={Config.R_ENTRY_THRESHOLD}"
                )
                return

            if ce_trigger and pe_trigger:
                entry_type = "CE" if atm_ce["R"] >= atm_pe["R"] else "PE"
            elif ce_trigger:
                entry_type = "CE"
            else:
                entry_type = "PE"

            info = atm_ce if entry_type == "CE" else atm_pe
            symbol = info["symbol"]
            entry_premium = info["ltp"]
            entry_R = info["R"]

            # [C4] cooldown: skip if we just exited this symbol
            last_exit = session.recent_exits.get(symbol)
            if last_exit is not None and \
               (time.time() - last_exit) < Config.REENTRY_COOLDOWN_SECONDS:
                remaining = Config.REENTRY_COOLDOWN_SECONDS - (time.time() - last_exit)
                session.decision_reason = (
                    f"{entry_type} cross but {symbol} in cooldown "
                    f"({remaining:.0f}s left)"
                )
                return

            log_console(
                f"🔔 R CROSS {entry_type} | ATM={info['strike']} | "
                f"Symbol={symbol} | LTP={info['ltp']:.2f} | "
                f"High={info['high']:.2f} | Low={info['low']:.2f} | "
                f"R {info['prev_R']:.4f} → {info['R']:.4f} "
                f"(crossed above {Config.R_ENTRY_THRESHOLD})"
            )

            if not execute_trades:
                session.decision_reason = (
                    f"{entry_type} R-cross {info['prev_R']:.4f}→{info['R']:.4f} "
                    f"but auto-trade OFF"
                )
                return

            if PAPER_MODE:
                create_position(session, symbol, entry_premium, entry_type, entry_R, spot)
                session.decision_reason = (
                    f"PAPER entered {entry_type} {symbol} | R={entry_R:.4f}"
                )
                return

            # LIVE entry
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
                order_id = resp.get("id", "")
                create_position(session, symbol, entry_premium, entry_type,
                                entry_R, spot, order_id=order_id)
                session.decision_reason = (
                    f"LIVE entered {entry_type} {symbol} | R={entry_R:.4f}"
                )
                log_console(
                    f"🟢 LIVE BUY {symbol} @ {entry_premium:.2f} | "
                    f"R={entry_R:.4f} | Order ID: {order_id}"
                )
            else:
                # [B3] log failure to trades.csv
                session.decision_reason = f"Live entry failed: {resp}"
                log_console(f"❌ LIVE entry failed for {symbol}: {resp}")
                if session.trade_logger:
                    session.trade_logger.log_failure(
                        event="ENTRY_FAILED",
                        symbol=symbol, option_type=entry_type,
                        strike=info["strike"], premium=entry_premium,
                        R=entry_R, spot=spot,
                        cumulative_pnl=session.paper_realized_pnl,
                        notes=str(resp)
                    )
            return

    except Exception as e:
        log_console(f"process_snapshot error: {e}")
        log_debug(traceback.format_exc())
        session.decision_reason = f"Error: {str(e)}"
    finally:
        session._order_lock = False

def _exit_position(session, fyers, pos, reason, extra="",
                   current_R=None, exit_spot=None):
    """Exit the given position (paper or live), log the trade."""
    held_symbol = pos["symbol"]
    entry_type = pos["entry_type"]

    if current_R is None:
        current_R = compute_R(session, held_symbol)
    if exit_spot is None:
        exit_spot = session.get_ltp("NSE:NIFTY50-INDEX")

    exit_premium = session.get_ltp(held_symbol)
    if exit_premium is None or exit_premium <= 0:
        exit_premium = pos["entry_premium"]

    if PAPER_MODE:
        pnl = (exit_premium - pos["entry_premium"]) * Config.LOT_SIZE
        session.paper_realized_pnl += pnl
        session.active_positions.clear()
        session.decision_reason = f"{entry_type} exited ({reason}) | {extra}"
        # [C4] record exit for cooldown
        session.recent_exits[held_symbol] = time.time()

        if session.trade_logger:
            session.trade_logger.log_exit(
                pos=pos, exit_premium=exit_premium,
                exit_R=current_R, exit_spot=exit_spot,
                exit_reason=reason, pnl=pnl,
                cumulative_pnl=session.paper_realized_pnl,
                notes=extra
            )

        log_console(
            f"🔴 PAPER EXIT {entry_type} {held_symbol} | Reason={reason} | "
            f"Premium {exit_premium:.2f} | P&L={pnl:+.2f} | "
            f"Cumulative={session.paper_realized_pnl:+.2f} | {extra}"
        )
        return True

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
        order_id = resp.get("id", "")
        pnl = (exit_premium - pos["entry_premium"]) * Config.LOT_SIZE
        session.paper_realized_pnl += pnl
        session.active_positions.clear()
        session.decision_reason = f"{entry_type} live exit ({reason}) | {extra}"
        session.recent_exits[held_symbol] = time.time()

        if session.trade_logger:
            session.trade_logger.log_exit(
                pos=pos, exit_premium=exit_premium,
                exit_R=current_R, exit_spot=exit_spot,
                exit_reason=reason, pnl=pnl,
                cumulative_pnl=session.paper_realized_pnl,
                order_id=order_id, notes=extra
            )

        log_console(
            f"🔴 LIVE EXIT {entry_type} {held_symbol} | Reason={reason} | "
            f"Order ID: {order_id} | P&L={pnl:+.2f} | "
            f"Cumulative={session.paper_realized_pnl:+.2f} | {extra}"
        )
        return True

    # [B3] log exit failure too
    session.decision_reason = f"Live exit failed: {resp}"
    log_console(f"❌ Live exit failed for {held_symbol}: {resp}")
    if session.trade_logger:
        session.trade_logger.log_failure(
            event="EXIT_FAILED",
            symbol=held_symbol, option_type=entry_type,
            strike=pos.get("strike"), premium=exit_premium,
            R=current_R, spot=exit_spot,
            cumulative_pnl=session.paper_realized_pnl,
            position=pos,
            notes=str(resp)
        )
    return False


# [B1][B2] bounded, lock-aware force-close
def force_close_all_positions(session, fyers, reason="EOD_FORCE_CLOSE"):
    # [B2] wait a bounded time for the sampler to release the lock
    for _ in range(20):
        if not getattr(session, '_order_lock', False):
            break
        time.sleep(0.25)
    if getattr(session, '_order_lock', False):
        log_console("⚠️ Sampler still busy during force-close; proceeding cautiously.")

    session._order_lock = True
    try:
        for attempt in range(3):
            if not session.active_positions:
                break
            with session.lock:
                pos = session.active_positions[0]
                current_R = compute_R(session, pos["symbol"])
                exit_spot = session.get_ltp("NSE:NIFTY50-INDEX")
                ok = _exit_position(session, fyers, pos, reason=reason,
                                    extra=f"forced attempt {attempt+1} | R={current_R}",
                                    current_R=current_R, exit_spot=exit_spot)
            if not ok:
                log_console(f"⚠️ Force-close attempt {attempt+1} failed; retrying…")
                time.sleep(1)
        if session.active_positions:
            log_console(f"❌ Could not force-close {len(session.active_positions)} position(s).")
    finally:
        session._order_lock = False

# ============================================================
# SAMPLER THREAD
# ============================================================
def sampler(session, tick_logger, auto_trade, fyers):
    log_debug("Sampler thread started")
    while session.running:
        start = now_ist()
        try:
            process_snapshot(session, fyers, execute_trades=auto_trade)

            with session.lock:
                spot = session.current_ltp
                atm = session.atm_strike
                vix = session.india_vix
                snap = dict(session.last_R_snapshot)
                active = [dict(p) for p in session.active_positions]
                realized = session.paper_realized_pnl
                daily_entries = session.daily_entries
                reason = session.decision_reason

            atm_ce_sym = atm_pe_sym = ''
            atm_ce_ltp = atm_pe_ltp = None
            atm_ce_high = atm_pe_high = None
            atm_ce_low = atm_pe_low = None
            atm_ce_R = atm_pe_R = None
            atm_ce_prevR = atm_pe_prevR = None

            ce_entry = snap.get((atm, "CE")) if atm else None
            pe_entry = snap.get((atm, "PE")) if atm else None

            if ce_entry:
                atm_ce_sym = ce_entry["symbol"]
                atm_ce_ltp = ce_entry["ltp"]
                atm_ce_high = ce_entry["high"]
                atm_ce_low = ce_entry["low"]
                atm_ce_R = ce_entry["R"]
                atm_ce_prevR = ce_entry["prev_R"]

            if pe_entry:
                atm_pe_sym = pe_entry["symbol"]
                atm_pe_ltp = pe_entry["ltp"]
                atm_pe_high = pe_entry["high"]
                atm_pe_low = pe_entry["low"]
                atm_pe_R = pe_entry["R"]
                atm_pe_prevR = pe_entry["prev_R"]

            neigh_m1_ce_R = neigh_m1_pe_R = None
            neigh_p1_ce_R = neigh_p1_pe_R = None
            for (k, ot), data in snap.items():
                if k is None or atm is None:
                    continue
                if k < atm:
                    if ot == "CE":
                        neigh_m1_ce_R = data["R"]
                    else:
                        neigh_m1_pe_R = data["R"]
                elif k > atm:
                    if ot == "CE":
                        neigh_p1_ce_R = data["R"]
                    else:
                        neigh_p1_pe_R = data["R"]

            held = {}
            total_pnl = realized
            if active:
                pos = active[0]
                held = {
                    'held_symbol': pos.get('symbol'),
                    'held_type': pos.get('entry_type'),
                    'held_strike': pos.get('strike'),
                    'held_entry_premium': pos.get('entry_premium'),
                    'held_current_premium': pos.get('current_premium'),
                    'held_entry_R': pos.get('entry_R'),
                    'held_current_R': pos.get('current_R'),
                    'held_highest_R': pos.get('highest_R'),
                    'held_trail_sl_R': pos.get('trailing_sl_R'),
                    'held_entry_spot': pos.get('entry_spot'),
                    'held_current_spot': spot,
                }
                entry = pos.get('entry_premium') or 0
                current = pos.get('current_premium') or entry
                if entry and current:
                    held['held_unrealized_pnl'] = (current - entry) * Config.LOT_SIZE
                    total_pnl += held['held_unrealized_pnl']

            tick_logger.log(
                timestamp=timestamp_ist(),
                index_ltp=spot,
                vix=vix,
                atm_strike=atm,
                atm_ce_sym=atm_ce_sym, atm_ce_ltp=atm_ce_ltp,
                atm_ce_high=atm_ce_high, atm_ce_low=atm_ce_low,
                atm_ce_R=atm_ce_R, atm_ce_prevR=atm_ce_prevR,
                atm_pe_sym=atm_pe_sym, atm_pe_ltp=atm_pe_ltp,
                atm_pe_high=atm_pe_high, atm_pe_low=atm_pe_low,
                atm_pe_R=atm_pe_R, atm_pe_prevR=atm_pe_prevR,
                neigh_m1_ce_R=neigh_m1_ce_R,
                neigh_m1_pe_R=neigh_m1_pe_R,
                neigh_p1_ce_R=neigh_p1_ce_R,
                neigh_p1_pe_R=neigh_p1_pe_R,
                realized_pnl=realized,
                total_pnl=total_pnl,
                daily_entries=daily_entries,
                decision_reason=reason,
                **held
            )

            elapsed = (now_ist() - start).total_seconds()
            if elapsed < Config.SAMPLE_INTERVAL:
                time.sleep(Config.SAMPLE_INTERVAL - elapsed)

        except Exception as e:
            log_debug(f"Sampler error: {e}\n{traceback.format_exc()}")
            time.sleep(1)

# ============================================================
# DASHBOARD
# ============================================================
class TerminalDashboard:
    def __init__(self, tick_logger, trade_logger, session_state):
        self.tick_logger = tick_logger
        self.trade_logger = trade_logger
        self.session_state = session_state

    def _hr(self, ch="─", w=100):
        return ch * w

    def display(self):
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            mode = "PAPER" if PAPER_MODE else "LIVE"

            ss = self.session_state
            with ss.lock:
                daily_entries = ss.daily_entries
                vix = ss.india_vix
                realized = ss.paper_realized_pnl
                decision_reason = ss.decision_reason
                ltp = ss.current_ltp
                atm = ss.atm_strike
                ce_R = ss.current_ce_R
                pe_R = ss.current_pe_R
                ce_prev_R = ss.current_ce_prev_R
                pe_prev_R = ss.current_pe_prev_R
                active_positions = [dict(p) for p in ss.active_positions]
                last_data = ss.last_data_time
                total_symbols, active_symbols = ss.get_symbol_data_counts(recency_seconds=5)
                stale_symbols = ss.get_stale_symbols(recency_seconds=5)
                tracked = ss.get_tracked_strikes()
                last_snap = dict(ss.last_R_snapshot)

            recent_trades = self.trade_logger.snapshot_recent() if self.trade_logger else []
            last_data_str = datetime.fromtimestamp(last_data).strftime("%H:%M:%S") if last_data else "—"

            W = 100
            # ------------------- header -------------------
            print()
            print(self._hr("═", W))
            print(f"  📊  {mode} TRADING DASHBOARD   ·   ATM R-Strategy")
            print(self._hr("═", W))
            print(f"  Time: {timestamp_ist()}   |   WS last data: {last_data_str}   |   "
                  f"Symbols active: {active_symbols}/{total_symbols}")

            if stale_symbols:
                shown = ', '.join(stale_symbols[:4])
                more = f" (+{len(stale_symbols)-4} more)" if len(stale_symbols) > 4 else ""
                print(f"  ⚠️  Watched & stale >5s: {shown}{more}")

            # ------------------- market -------------------
            print()
            print(self._hr())
            print("  🌐  MARKET")
            print(self._hr())
            print(f"  Index LTP:   {ltp:>10.2f}       India VIX:  {vix:>6.2f}")
            print(f"  ATM Strike:  {str(atm) if atm is not None else '—':>10}       "
                  f"Tracked:    {tracked}")

            # ------------------- R table -------------------
            print()
            print(self._hr())
            print(f"  📈  R VALUES    (R = (ltp − dayLow) / (dayHigh − dayLow))    "
                  f"Entry: prev_R < {Config.R_ENTRY_THRESHOLD} < current_R")
            print(self._hr())
            hdr = (f"  {'Strike':<8} | "
                   f"{'CE LTP':>9} {'CE High':>9} {'CE Low':>9} {'CE R':>8} {'CE prev':>9} | "
                   f"{'PE LTP':>9} {'PE High':>9} {'PE Low':>9} {'PE R':>8} {'PE prev':>9}")
            print(hdr)
            print("  " + "─" * (W - 2))

            # [M3] use NaN so column widths stay aligned when values are missing
            def col(d, key, fmt):
                v = d.get(key)
                if v is None:
                    v = float('nan')
                return format(v, fmt)

            for k in tracked:
                ce_d = last_snap.get((k, "CE"), {})
                pe_d = last_snap.get((k, "PE"), {})
                star = "★" if k == atm else " "
                print(
                    f" {star}{k:<7} | "
                    f"{col(ce_d,'ltp','9.2f')} {col(ce_d,'high','9.2f')} {col(ce_d,'low','9.2f')} "
                    f"{col(ce_d,'R','8.4f')} {col(ce_d,'prev_R','9.4f')} | "
                    f"{col(pe_d,'ltp','9.2f')} {col(pe_d,'high','9.2f')} {col(pe_d,'low','9.2f')} "
                    f"{col(pe_d,'R','8.4f')} {col(pe_d,'prev_R','9.4f')}"
                )
            print("  (★ = current ATM;  prev = R from the tick immediately before this one;  nan = no fresh value)")

            # ------------------- signal watch -------------------
            print()
            print(self._hr())
            print("  🔔  SIGNAL WATCH  (ATM only)")
            print(self._hr())
            ce_state = "—"
            pe_state = "—"
            if ce_R is not None and ce_prev_R is not None:
                if ce_prev_R < Config.R_ENTRY_THRESHOLD and ce_R > Config.R_ENTRY_THRESHOLD:
                    ce_state = "✅ CROSSED UP → CE BUY"
                elif ce_R > Config.R_ENTRY_THRESHOLD:
                    ce_state = "🟡 above thr (no cross this tick)"
                else:
                    ce_state = "⏳ waiting"
            if pe_R is not None and pe_prev_R is not None:
                if pe_prev_R < Config.R_ENTRY_THRESHOLD and pe_R > Config.R_ENTRY_THRESHOLD:
                    pe_state = "✅ CROSSED UP → PE BUY"
                elif pe_R > Config.R_ENTRY_THRESHOLD:
                    pe_state = "🟡 above thr (no cross this tick)"
                else:
                    pe_state = "⏳ waiting"

            ce_prev_s = f"{ce_prev_R:.4f}" if ce_prev_R is not None else "—"
            pe_prev_s = f"{pe_prev_R:.4f}" if pe_prev_R is not None else "—"
            ce_R_s = f"{ce_R:.4f}" if ce_R is not None else "—"
            pe_R_s = f"{pe_R:.4f}" if pe_R is not None else "—"
            print(f"  CE:  prev={ce_prev_s:>8}  →  cur={ce_R_s:>8}      {ce_state}")
            print(f"  PE:  prev={pe_prev_s:>8}  →  cur={pe_R_s:>8}      {pe_state}")

            # ------------------- position -------------------
            print()
            print(self._hr())
            print("  📌  POSITION")
            print(self._hr())
            if active_positions:
                pos = active_positions[0]
                entry = pos.get("entry_premium", 0) or 0
                cur = pos.get("current_premium", entry) or entry
                pnl = (cur - entry) * Config.LOT_SIZE if entry else 0
                print(f"  Symbol:      {pos.get('symbol')}")
                print(f"  Type:        {pos.get('entry_type')}       Strike: {pos.get('strike')}")
                print(f"  Entry Prem:  {entry:>8.2f}       Current:   {cur:>8.2f}       "
                      f"Unrealized P&L: {pnl:>+9.2f}")
                print(f"  Entry Spot:  {pos.get('entry_spot', 0):>8.2f}       "
                      f"Current Spot: {ltp:>8.2f}")
                print(f"  Entry R:     {pos.get('entry_R', 0):>8.4f}       "
                      f"Current R: {pos.get('current_R', 0):>8.4f}       "
                      f"Highest R: {pos.get('highest_R', 0):>8.4f}")
                print(f"  Trailing SL R (Highest R − {Config.R_TRAIL_OFFSET}):  "
                      f"{pos.get('trailing_sl_R', 0):.4f}")
                print(f"  Daily Entry #: {pos.get('daily_entry_num')}")
            else:
                print("  No active position (FLAT)")

            # ------------------- P&L -------------------
            print()
            print(self._hr())
            print("  💰  P&L")
            print(self._hr())
            total_unreal = 0.0
            for pos in active_positions:
                entry = pos.get('entry_premium') or 0
                cur = pos.get('current_premium') or entry
                if entry and cur:
                    total_unreal += (cur - entry) * Config.LOT_SIZE
            total_pnl = realized + total_unreal
            print(f"  Realized P&L:    {realized:>+12.2f}")
            print(f"  Unrealized P&L:  {total_unreal:>+12.2f}")
            print(f"  Total P&L:       {total_pnl:>+12.2f}")
            print(f"  Daily Entries:   {daily_entries}/{Config.MAX_DAILY_ENTRIES}")

            # ------------------- recent trades -------------------
            print()
            print(self._hr())
            print(f"  📜  RECENT TRADES (last {len(recent_trades)})")
            print(self._hr())
            if not recent_trades:
                print("  (no trades yet)")
            else:
                print(f"  {'Time':<9} {'Evt':<13} {'Type':<5} {'Strike':>7} "
                      f"{'Premium':>9} {'R':>8} {'PnL':>10} {'CumPnL':>11}  Notes")
                print("  " + "─" * (W - 2))
                for t in recent_trades:
                    ts = t.get('timestamp', '')
                    t_short = ts.split(' ')[-1] if ' ' in ts else ts
                    evt = t.get('event', '')
                    ot = t.get('option_type', '') or ''
                    strike = t.get('strike', '') or ''
                    prem = t.get('premium', '')
                    R = t.get('R', '')
                    pnl = t.get('pnl', '')
                    cum = t.get('cumulative_pnl', '')
                    notes = t.get('notes', '') or ''
                    if len(notes) > 34:
                        notes = notes[:31] + "..."
                    def f(v, fmt):
                        try:
                            return format(float(v), fmt)
                        except (TypeError, ValueError):
                            return format(float('nan'), fmt)
                    print(f"  {t_short:<9} {evt:<13} {ot:<5} {str(strike):>7} "
                          f"{f(prem, '9.2f')} {f(R, '8.4f')} {f(pnl, '10.2f')} "
                          f"{f(cum, '11.2f')}  {notes}")

            # ------------------- decision -------------------
            print()
            print(self._hr())
            print("  🧠  DECISION")
            print(self._hr())
            print(f"  {decision_reason}")

            # ------------------- config -------------------
            print()
            print(self._hr())
            print("  ⚙️   CONFIG")
            print(self._hr())
            print(f"  Entry:           prev_R < {Config.R_ENTRY_THRESHOLD}  AND  current_R > {Config.R_ENTRY_THRESHOLD}  (ATM only)")
            print(f"  Exit (first of): R >= {Config.R_TARGET_EXIT}   OR   R <= (highest_R − {Config.R_TRAIL_OFFSET})")
            print(f"  Tracked strikes: ATM ± {Config.STRIKE_SPAN}     |     Lot size: {Config.LOT_SIZE}")
            print(f"  Cooldowns:       prev_R max age {Config.R_PREV_MAX_AGE}s   |   "
                  f"re-entry {Config.REENTRY_COOLDOWN_SECONDS}s   |   "
                  f"hold-feed force-exit {Config.HOLD_STALE_FORCE_EXIT_SECONDS}s")
            print(f"  Logs:            ticks → {Config.TICK_CSV}   |   trades → {Config.TRADE_CSV}")
            print(self._hr("═", W))
            print()
        except Exception as e:
            log_debug(f"Dashboard error: {e}\n{traceback.format_exc()}")

# ============================================================
# HEARTBEAT MONITOR
# ============================================================
def heartbeat_monitor(session, ws_manager, index_symbol):
    while session.running:
        time.sleep(Config.WS_HEARTBEAT_TIMEOUT // 2)

        last_ts = session.get_ltp_time(index_symbol)
        if last_ts is not None and (time.time() - last_ts) > Config.WS_HEARTBEAT_TIMEOUT:
            log_console(f"⚠️ NIFTY LTP stale for >{Config.WS_HEARTBEAT_TIMEOUT}s, forcing WS restart...")
            ws_manager.stop()
            time.sleep(2)
            ws_manager.start()
            if ws_manager.subscribed:
                ws_manager.subscribe(list(ws_manager.subscribed))
            continue

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
                    stale.add(sym)

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
    log_console(
        f"Starting ATM R-Strategy | Entry: prev_R < {Config.R_ENTRY_THRESHOLD} "
        f"AND current_R > {Config.R_ENTRY_THRESHOLD} | "
        f"Exit (first of): R >= {Config.R_TARGET_EXIT} OR R <= (highest_R − {Config.R_TRAIL_OFFSET}) | "
        f"Tracking ATM ± {Config.STRIKE_SPAN}"
    )
    session = SessionState()
    load_credentials(session)

    tick_logger = TickLogger()
    trade_logger = TradeLogger()
    session.tick_logger = tick_logger
    session.trade_logger = trade_logger

    dashboard = TerminalDashboard(tick_logger, trade_logger, session)
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

            fyers = fyersModel.FyersModel(client_id=session.cid, token=session.token,
                                          is_async=False, log_path="")
            poller_thread = threading.Thread(
                target=chain_poller, args=(session, ws_manager, index_symbol), daemon=True)
            sampler_thread = threading.Thread(
                target=sampler, args=(session, tick_logger, auto_trade, fyers), daemon=True)
            heartbeat_thread = threading.Thread(
                target=heartbeat_monitor, args=(session, ws_manager, index_symbol), daemon=True)

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
                    log_console(f"⚠️ Forcibly closing {len(session.active_positions)} open position(s)...")
                    try:
                        force_close_all_positions(session, fyers, reason="EOD_FORCE_CLOSE")
                    except Exception as e:
                        log_console(f"Force-close failed: {e}")
                        log_debug(traceback.format_exc())

                with session.lock:
                    if session.active_positions:
                        log_console(f"⚠️ {len(session.active_positions)} position(s) could not be closed. Check broker manually.")
                    else:
                        session.eod_square_off_triggered = False
                        session.daily_entries = 0
                        session.paper_realized_pnl = 0.0
                        log_console("✅ Trading stopped and state reset.")
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