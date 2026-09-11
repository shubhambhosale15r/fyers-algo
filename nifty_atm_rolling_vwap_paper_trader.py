import csv
import os
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple

import pytz
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

GRAPH_AVAILABLE = False
try:
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    GRAPH_AVAILABLE = True
except Exception:
    GRAPH_AVAILABLE = False


# ============================================================
# ANSI COLORS
# ============================================================

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def col(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


# ============================================================
# CONFIG
# ============================================================

class Config:
    CLIENT_ID = 'J0ENYFX9ME-200'
    ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcW8zX1lsTm1fWGFvTEdVU2lZdWF1YnA4ODRBWjlTYW5Idm9lRnkzVlBGNXE5WTNlWG1RNDZvNXp0ZjNoX0N4bGhqdlQyT3hsenBNRHBiRjdVSWVfNFN2TjlVY1o3N1VYbFVhWVZnTlhWdG15eEQzQT0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI0NDIxYTAyNDU1ZWQyOTg3NzZhYzdiMzRmYzAzODJmYjk2MTE4Y2M1NzE3Y2NlMWE0ZWJkYmU0YyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFBMzE0OTUiLCJhcHBUeXBlIjoyMDAsImV4cCI6MTc4OTE3MzAwMCwiaWF0IjoxNzg5MDk5OTkyLCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3ODkwOTk5OTIsInN1YiI6ImFjY2Vzc190b2tlbiJ9.Pgp_UWNAJca3mqDbFsAy67wCWStfdn9DvPApQj0ltpE"

    INDEX_SYMBOL = "NSE:NIFTY50-INDEX"
    PAPER_MODE = True

    OPTION_CHAIN_INTERVAL_SEC = 30
    VWAP_FAST_SEC = 60
    RANGE_LOOKBACK_SEC = 300
    REQUIRE_FULL_RANGE_WARMUP = True
    MAX_OPEN_POSITIONS = 1

    USE_ASK_FOR_ENTRY = True
    USE_BID_FOR_EXIT = True

    LOT_SIZE = 65

    DAILY_MAX_LOSS_PCT = 0.0
    DAILY_TARGET_PCT = 0.0
    PAPER_CAPITAL = 100000.0

    TRAILING_SL_MIN_RANGE = 0.0

    TIMEZONE = "Asia/Kolkata"
    MARKET_START = dt_time(9, 15)
    LAST_ENTRY_TIME = dt_time(15, 0)
    FORCE_EXIT_TIME = dt_time(15, 5)
    MARKET_END = dt_time(15, 30)

    PRINT_EVERY_TICK = False
    DASHBOARD_INTERVAL_SEC = 1
    BANNER_HOLD_SEC = 5.0

    LOG_DIR = "vwap_logs"

    WS_LOG_PATH = ""
    WS_LITEMODE = False
    WS_WRITE_TO_FILE = False
    WS_RECONNECT = True

    ENABLE_LIVE_GRAPH = True

    FRAME_WIDTH = 128          # nominal dashboard width
    NON_TTY_REFRESH_SEC = 30   # refresh period when piped/redirected


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Tick:
    ltt: int
    ltp: float
    ltq: int
    vol_traded_today: int
    bid: float
    ask: float
    received_at: float

    @property
    def timestamp(self) -> float:
        return float(self.ltt)

    @property
    def pv(self) -> float:
        return self.ltp * self.ltq


@dataclass
class ContractInfo:
    symbol: str
    strike: int
    option_type: str
    offset_name: str


@dataclass
class Position:
    side: str
    symbol: str
    strike: int
    option_type: str
    entry_time: str
    entry_price: float
    entry_ltp: float
    entry_bid: float
    entry_ask: float
    entry_vwap_1m: float
    entry_range_high: Optional[float]
    entry_range_low: Optional[float]
    entry_range: Optional[float]
    quantity: int
    highest_vwap_since_entry: float
    trailing_sl: Optional[float] = None


# ============================================================
# HELPERS
# ============================================================

class TimeHelper:
    TZ = pytz.timezone(Config.TIMEZONE)

    @classmethod
    def now(cls) -> datetime:
        return datetime.now(cls.TZ)

    @classmethod
    def now_str(cls) -> str:
        return cls.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def epoch_to_str(cls, epoch: int) -> str:
        try:
            return datetime.fromtimestamp(int(epoch), cls.TZ).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception:
            return ""

    @classmethod
    def current_time_only(cls) -> dt_time:
        return cls.now().time().replace(microsecond=0)


def safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def is_valid_price(value: float) -> bool:
    return value is not None and value > 0


def fmt_num(v, width: int = 10, decimals: int = 2, dash: str = "—") -> str:
    if v is None:
        return f"{dash:>{width}}"
    try:
        return f"{v:>{width}.{decimals}f}"
    except Exception:
        return f"{dash:>{width}}"


# ============================================================
# CONSOLE — single owner of stdout
# ============================================================

class Console:
    """
    All terminal output funnels through this class. Every write is
    serialized by one lock, so the dashboard frame and any status
    messages can never interleave.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.is_tty = sys.stdout.isatty()
        self._last_stable_sig: Optional[Tuple] = None
        self._last_emit_ts: float = 0.0

    # ---- size ----
    def width(self) -> int:
        try:
            return max(80, shutil.get_terminal_size((Config.FRAME_WIDTH, 40)).columns)
        except Exception:
            return Config.FRAME_WIDTH

    def height(self) -> int:
        try:
            return max(15, shutil.get_terminal_size((Config.FRAME_WIDTH, 40)).lines)
        except Exception:
            return 40

    # ---- static output (before dashboard starts, and during shutdown) ----
    def write_static(self, text: str):
        with self.lock:
            sys.stdout.write(text)
            if not text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

    # ---- dashboard frame ----
    def render_frame(self, frame: str, stable_sig: Tuple, force: bool = False):
        """
        Redraw the dashboard in place. `stable_sig` is a tuple of the
        fields that matter — when stdout is not a TTY we only re-emit
        the frame when the signature changes (or every NON_TTY_REFRESH_SEC).
        """
        with self.lock:
            # Trim to fit terminal
            cols = self.width() - 1
            rows = self.height() - 1
            raw_lines = frame.split("\n")
            trimmed = []
            for ln in raw_lines:
                if len(ln) > cols:
                    ln = ln[:cols]
                trimmed.append(ln)
            if len(trimmed) > rows:
                trimmed = trimmed[:rows - 1] + [col("… (truncated) …", C.GRAY)]
            frame = "\n".join(trimmed)

            if not self.is_tty:
                now = time.time()
                same = (stable_sig == self._last_stable_sig)
                recent = (now - self._last_emit_ts) < Config.NON_TTY_REFRESH_SEC
                if same and recent and not force:
                    return
                self._last_stable_sig = stable_sig
                self._last_emit_ts = now
                sys.stdout.write("\n" + frame + "\n")
                sys.stdout.flush()
                return

            # TTY: atomic rewind + redraw + clear-below
            sys.stdout.write("\033[H" + frame + "\033[J")
            sys.stdout.flush()

    def clear(self):
        with self.lock:
            if self.is_tty:
                sys.stdout.write("\033[2J\033[H")
            else:
                os.system("cls" if os.name == "nt" else "clear")
            sys.stdout.flush()


# ============================================================
# CSV LOGGER
# ============================================================

class CSVLogger:
    def __init__(self):
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        date_str = TimeHelper.now().strftime("%Y%m%d")

        self.tick_file = os.path.join(Config.LOG_DIR, f"tick_data_{date_str}.csv")
        self.trade_file = os.path.join(Config.LOG_DIR, f"paper_trades_{date_str}.csv")
        self.event_file = os.path.join(Config.LOG_DIR, f"events_{date_str}.csv")

        self.lock = threading.Lock()

        self.tick_fields = [
            "log_time", "ltt", "ltt_ist", "symbol", "strike", "option_type",
            "offset_name", "ltp", "ltq", "vol_traded_today", "bid", "ask",
            "rolling_qty_1m", "vwap_1m",
            "range_high", "range_low", "range",
            "breakout", "armed", "warmup_ready",
            "is_atm", "position",
        ]
        self.trade_fields = [
            "event_time", "action", "side", "symbol", "strike", "option_type",
            "quantity",
            "entry_time", "entry_price", "entry_ltp", "entry_bid", "entry_ask",
            "entry_vwap_1m", "entry_range_high", "entry_range_low", "entry_range",
            "exit_time", "exit_price", "exit_ltp", "exit_bid", "exit_ask",
            "exit_vwap_1m", "exit_range_high", "exit_range_low", "exit_range",
            "exit_trailing_sl", "exit_highest_vwap",
            "gross_pnl_points", "gross_pnl_rupees",
            "estimated_charges", "net_pnl_rupees", "reason",
        ]
        self.event_fields = [
            "event_time", "event", "message", "symbol", "strike",
            "option_type", "details",
        ]

        self._ensure_header(self.tick_file, self.tick_fields)
        self._ensure_header(self.trade_file, self.trade_fields)
        self._ensure_header(self.event_file, self.event_fields)

    @staticmethod
    def _ensure_header(path: str, fields: List[str]):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fields)

    def _write(self, path, fields, row):
        with self.lock:
            with open(path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields,
                               extrasaction="ignore").writerow(row)

    def write_tick(self, row):  self._write(self.tick_file, self.tick_fields, row)
    def write_trade(self, row): self._write(self.trade_file, self.trade_fields, row)
    def write_event(self, row): self._write(self.event_file, self.event_fields, row)


# ============================================================
# ROLLING VWAP
# ============================================================

class RollingVWAP:
    def __init__(self):
        self.ticks = deque()
        self.sum_pv = 0.0
        self.sum_qty = 0
        self.last_signature: Optional[Tuple] = None

    def reset(self):
        self.ticks.clear()
        self.sum_pv = 0.0
        self.sum_qty = 0
        self.last_signature = None

    def add_tick(self, tick: Tick) -> bool:
        sig = (tick.ltt, round(tick.ltp, 8), tick.ltq, tick.vol_traded_today)
        if sig == self.last_signature:
            return False
        self.last_signature = sig
        if tick.ltt <= 0 or tick.ltp <= 0 or tick.ltq <= 0:
            return True
        self.ticks.append(tick)
        self.sum_pv += tick.pv
        self.sum_qty += tick.ltq
        return True

    def remove_before(self, cutoff_epoch: float):
        while self.ticks and self.ticks[0].timestamp < cutoff_epoch:
            old = self.ticks.popleft()
            self.sum_pv -= old.pv
            self.sum_qty -= old.ltq
        if abs(self.sum_pv) < 1e-10:
            self.sum_pv = 0.0
        if self.sum_qty < 0:
            self.sum_qty = 0

    def value(self) -> Optional[float]:
        if self.sum_qty <= 0:
            return None
        return self.sum_pv / self.sum_qty

    def quantity(self) -> int:
        return self.sum_qty


# ============================================================
# SYMBOL STATE
# ============================================================

class SymbolVWAPState:
    def __init__(self):
        self.engine_1m = RollingVWAP()
        self.vwap_history = deque()
        self.previous_range_high: Optional[float] = None
        self.previous_range_low: Optional[float] = None
        self.breakout_armed: bool = True
        self.breakout_this_tick: bool = False
        self.last_tick: Optional[Tick] = None
        self.last_update_wallclock: float = 0.0

    def reset(self):
        self.engine_1m.reset()
        self.vwap_history.clear()
        self.previous_range_high = None
        self.previous_range_low = None
        self.breakout_armed = True
        self.breakout_this_tick = False
        self.last_tick = None
        self.last_update_wallclock = 0.0

    def qty_1m(self) -> int:
        return self.engine_1m.quantity()

    def _prune_history(self, cutoff_epoch: float):
        while self.vwap_history and self.vwap_history[0][0] < cutoff_epoch:
            self.vwap_history.popleft()

    def _range_from_history(self):
        if not self.vwap_history:
            return None, None
        vals = [v for _, v in self.vwap_history]
        return max(vals), min(vals)

    def update(self, tick: Tick) -> bool:
        if tick.ltq <= 0 or tick.ltp <= 0 or tick.ltt <= 0:
            return False
        if not self.engine_1m.add_tick(tick):
            return False

        self.engine_1m.remove_before(tick.ltt - Config.VWAP_FAST_SEC)
        v1 = self.engine_1m.value()
        if v1 is None:
            return False

        self._prune_history(tick.ltt - Config.RANGE_LOOKBACK_SEC)

        warm = (
            bool(self.vwap_history)
            and (tick.ltt - self.vwap_history[0][0]) >= Config.RANGE_LOOKBACK_SEC
        )
        if warm and self.vwap_history:
            prev_high, prev_low = self._range_from_history()
        else:
            prev_high, prev_low = None, None

        self.previous_range_high = prev_high
        self.previous_range_low = prev_low

        self.breakout_this_tick = False
        if warm and prev_high is not None:
            if v1 > prev_high:
                if self.breakout_armed:
                    self.breakout_this_tick = True
                    self.breakout_armed = False
            else:
                self.breakout_armed = True

        self.vwap_history.append((tick.ltt, v1))
        self.last_tick = tick
        self.last_update_wallclock = time.time()
        return True

    def current_vwap_1m(self) -> Optional[float]:
        return self.engine_1m.value()

    def warmup_ready(self) -> bool:
        if not self.vwap_history:
            return False
        span = self.vwap_history[-1][0] - self.vwap_history[0][0]
        return span >= Config.RANGE_LOOKBACK_SEC

    def metrics(self) -> dict:
        v1 = self.engine_1m.value()
        rh = self.previous_range_high
        rl = self.previous_range_low
        rng = (rh - rl) if (rh is not None and rl is not None) else None
        return {
            "vwap_1m": v1,
            "range_high": rh,
            "range_low": rl,
            "range": rng,
            "breakout": bool(self.breakout_this_tick),
            "armed": bool(self.breakout_armed),
            "warmup_ready": self.warmup_ready(),
        }


# ============================================================
# PAPER TRADER
# ============================================================

class PaperTrader:
    def __init__(self, logger: CSVLogger):
        self.logger = logger
        self.position: Optional[Position] = None
        self.realized_pnl = 0.0
        self.trade_count = 0
        self.lock = threading.RLock()
        self.banner_until: float = 0.0
        self.banner_lines: List[str] = []

    def _estimated_charges(self, entry_price: float, exit_price: float) -> float:
        return 0.0

    def can_enter(self) -> bool:
        with self.lock:
            if self.position is not None: return False
            if Config.MAX_OPEN_POSITIONS <= 0: return False
            if not self._within_daily_limits(): return False
            now_t = TimeHelper.current_time_only()
            return Config.MARKET_START <= now_t < Config.LAST_ENTRY_TIME

    def _within_daily_limits(self) -> bool:
        capital = Config.PAPER_CAPITAL
        if capital <= 0: return True
        if Config.DAILY_MAX_LOSS_PCT > 0:
            max_loss = capital * Config.DAILY_MAX_LOSS_PCT / 100.0
            if self.realized_pnl <= -max_loss: return False
        if Config.DAILY_TARGET_PCT > 0:
            target = capital * Config.DAILY_TARGET_PCT / 100.0
            if self.realized_pnl >= target: return False
        return True

    def entry_execution_price(self, tick: Tick) -> float:
        if Config.USE_ASK_FOR_ENTRY and is_valid_price(tick.ask):
            return tick.ask
        return tick.ltp

    def exit_execution_price(self, tick: Tick) -> float:
        if Config.USE_BID_FOR_EXIT and is_valid_price(tick.bid):
            return tick.bid
        return tick.ltp

    def _banner(self, title: str, color: str, rows: List[Tuple[str, str]]):
        w = 88
        lines = [col("═" * w, color),
                 col(f"  {title}", C.BOLD + color),
                 col("─" * w, color)]
        for k, v in rows:
            lines.append(f"  {col(f'{k:<12}', C.CYAN)}: {v}")
        lines.append(col("═" * w, color))
        self.banner_lines = lines
        self.banner_until = time.time() + Config.BANNER_HOLD_SEC

    def enter(self, contract: ContractInfo, tick: Tick,
              metrics: dict, reason: str) -> bool:
        with self.lock:
            if not self.can_enter(): return False
            price = self.entry_execution_price(tick)
            if price <= 0: return False
            v1 = metrics.get("vwap_1m")
            if v1 is None: return False
            rh = metrics.get("range_high"); rl = metrics.get("range_low")
            rng = metrics.get("range")
            now_str = TimeHelper.now_str()

            self.position = Position(
                side=contract.option_type, symbol=contract.symbol,
                strike=contract.strike, option_type=contract.option_type,
                entry_time=now_str, entry_price=price,
                entry_ltp=tick.ltp, entry_bid=tick.bid, entry_ask=tick.ask,
                entry_vwap_1m=v1,
                entry_range_high=rh, entry_range_low=rl, entry_range=rng,
                quantity=Config.LOT_SIZE,
                highest_vwap_since_entry=v1, trailing_sl=None,
            )

            self.logger.write_event({
                "event_time": now_str, "event": "PAPER_ENTRY",
                "message": reason, "symbol": contract.symbol,
                "strike": contract.strike,
                "option_type": contract.option_type,
                "details": (f"price={price},ltp={tick.ltp},bid={tick.bid},"
                            f"ask={tick.ask},vwap1={v1},range_high={rh},"
                            f"range_low={rl},range={rng}"),
            })

            self._banner("★★★  PAPER ENTRY  ★★★", C.GREEN, [
                ("TIME", now_str),
                ("ACTION", col(f"BUY {contract.option_type}", C.BOLD + C.GREEN)),
                ("SYMBOL", contract.symbol),
                ("STRIKE", str(contract.strike)),
                ("ENTRY PRICE", col(f"{price:.2f}  (ask)", C.YELLOW)),
                ("LTP", f"{tick.ltp:.2f}"),
                ("1M VWAP", f"{v1:.4f}"),
                ("RANGE HIGH", f"{rh}" if rh is not None else "—"),
                ("RANGE LOW", f"{rl}" if rl is not None else "—"),
                ("RANGE", f"{rng}" if rng is not None else "—"),
                ("QUANTITY", str(Config.LOT_SIZE)),
                ("REASON", reason),
            ])
            return True

    def update_trailing_and_check_exit(self, tick: Tick, metrics: dict) -> Optional[dict]:
        with self.lock:
            pos = self.position
            if pos is None: return None
            v1 = metrics.get("vwap_1m")
            rng = metrics.get("range")
            if v1 is None: return None
            if v1 > pos.highest_vwap_since_entry:
                pos.highest_vwap_since_entry = v1
            if rng is not None and rng > Config.TRAILING_SL_MIN_RANGE:
                candidate = pos.highest_vwap_since_entry - rng
                if pos.trailing_sl is None or candidate > pos.trailing_sl:
                    pos.trailing_sl = candidate
            if pos.trailing_sl is not None and v1 <= pos.trailing_sl:
                reason = (f"TRAILING_SL_HIT vwap1={v1:.4f} <= sl={pos.trailing_sl:.4f} "
                          f"high={pos.highest_vwap_since_entry:.4f} "
                          f"range={rng if rng is not None else 'NA'}")
                return self.exit(tick, metrics, reason)
            return None

    def exit(self, tick: Tick, metrics: dict, reason: str) -> Optional[dict]:
        with self.lock:
            if self.position is None: return None
            pos = self.position
            exit_price = self.exit_execution_price(tick)
            if exit_price <= 0: return None

            pnl_points = exit_price - pos.entry_price
            gross_pnl = pnl_points * pos.quantity
            charges = self._estimated_charges(pos.entry_price, exit_price)
            net_pnl = gross_pnl - charges
            self.realized_pnl += net_pnl
            self.trade_count += 1
            now_str = TimeHelper.now_str()

            row = {
                "event_time": now_str, "action": "EXIT", "side": pos.side,
                "symbol": pos.symbol, "strike": pos.strike,
                "option_type": pos.option_type, "quantity": pos.quantity,
                "entry_time": pos.entry_time, "entry_price": pos.entry_price,
                "entry_ltp": pos.entry_ltp, "entry_bid": pos.entry_bid,
                "entry_ask": pos.entry_ask,
                "entry_vwap_1m": pos.entry_vwap_1m,
                "entry_range_high": pos.entry_range_high,
                "entry_range_low": pos.entry_range_low,
                "entry_range": pos.entry_range,
                "exit_time": now_str, "exit_price": exit_price,
                "exit_ltp": tick.ltp, "exit_bid": tick.bid, "exit_ask": tick.ask,
                "exit_vwap_1m": metrics.get("vwap_1m"),
                "exit_range_high": metrics.get("range_high"),
                "exit_range_low": metrics.get("range_low"),
                "exit_range": metrics.get("range"),
                "exit_trailing_sl": pos.trailing_sl,
                "exit_highest_vwap": pos.highest_vwap_since_entry,
                "gross_pnl_points": pnl_points,
                "gross_pnl_rupees": gross_pnl,
                "estimated_charges": charges,
                "net_pnl_rupees": net_pnl, "reason": reason,
            }

            self.logger.write_trade(row)
            self.logger.write_event({
                "event_time": now_str, "event": "PAPER_EXIT", "message": reason,
                "symbol": pos.symbol, "strike": pos.strike,
                "option_type": pos.option_type,
                "details": (f"entry={pos.entry_price},exit={exit_price},"
                            f"sl={pos.trailing_sl},high={pos.highest_vwap_since_entry},"
                            f"gross={gross_pnl},charges={charges},net={net_pnl},"
                            f"daily={self.realized_pnl}"),
            })

            pnl_color = C.GREEN if net_pnl >= 0 else C.RED
            self._banner("★★★  PAPER EXIT  ★★★", C.YELLOW, [
                ("TIME", now_str),
                ("ACTION", col(f"SELL {pos.option_type}", C.BOLD + C.YELLOW)),
                ("SYMBOL", pos.symbol),
                ("STRIKE", str(pos.strike)),
                ("ENTRY PRICE", f"{pos.entry_price:.2f}  (ask)"),
                ("EXIT PRICE", f"{exit_price:.2f}  (bid)"),
                ("POINTS P&L", col(f"{pnl_points:+.2f}", pnl_color)),
                ("GROSS P&L", col(f"{gross_pnl:+.2f}", pnl_color)),
                ("CHARGES", f"{charges:.2f}"),
                ("NET P&L", col(f"{net_pnl:+.2f}", C.BOLD + pnl_color)),
                ("DAILY NET", col(f"{self.realized_pnl:+.2f}", pnl_color)),
                ("HIGH VWAP", f"{pos.highest_vwap_since_entry:.4f}"),
                ("TRAILING SL", f"{pos.trailing_sl}"),
                ("REASON", reason),
            ])

            self.position = None
            return row

    def force_exit_if_needed(self, tick_lookup, metrics_lookup):
        if self.position is None: return
        now_t = TimeHelper.current_time_only()
        if now_t < Config.FORCE_EXIT_TIME: return
        pos = self.position
        tick = tick_lookup.get(pos.symbol)
        metrics = metrics_lookup.get(pos.symbol)
        if tick is None: return
        if metrics is None:
            metrics = {"vwap_1m": None, "range_high": None,
                       "range_low": None, "range": None}
        self.exit(tick, metrics, "FORCED_MARKET_CLOSE")


# ============================================================
# LIVE VWAP GRAPH
# ============================================================

class LiveVWAPGraph:
    MAX_POINTS = 900

    def __init__(self):
        self.lock = threading.RLock()
        self.data: Dict[str, dict] = {}
        self.atm_ce_symbol: Optional[str] = None
        self.atm_pe_symbol: Optional[str] = None
        self.running = True
        self.fig = None
        self._lines_ready = False

    def set_atm(self, ce_symbol, pe_symbol):
        with self.lock:
            self.atm_ce_symbol = ce_symbol
            self.atm_pe_symbol = pe_symbol

    def push(self, symbol, ts, v1, rh, rl, breakout):
        if v1 is None: return
        with self.lock:
            d = self.data.get(symbol)
            if d is None:
                d = {"times": deque(maxlen=self.MAX_POINTS),
                     "v1": deque(maxlen=self.MAX_POINTS),
                     "rh": deque(maxlen=self.MAX_POINTS),
                     "rl": deque(maxlen=self.MAX_POINTS),
                     "breakouts": deque(maxlen=200)}
                self.data[symbol] = d
            d["times"].append(ts); d["v1"].append(v1)
            d["rh"].append(rh if rh is not None else v1)
            d["rl"].append(rl if rl is not None else v1)
            if breakout:
                d["breakouts"].append((ts, v1))

    def _setup(self):
        self.fig, (self.ax_ce, self.ax_pe) = plt.subplots(2, 1, figsize=(14, 8))
        self.fig.suptitle("ATM Rolling VWAP Breakout | 1m VWAP + 5m range | ▲ = breakout",
                          fontsize=11)
        for ax, title in ((self.ax_ce, "ATM CE"), (self.ax_pe, "ATM PE")):
            ax.set_title(title, fontsize=10, loc="left")
            ax.grid(True, alpha=0.3); ax.set_ylabel("1m VWAP")
            ax.tick_params(axis="x", rotation=30)
        self.line_ce_v1, = self.ax_ce.plot([], [], "b-", lw=1.3, label="1m VWAP")
        self.line_ce_rh, = self.ax_ce.plot([], [], "--", color="green", lw=1.0, label="range high")
        self.line_ce_rl, = self.ax_ce.plot([], [], "--", color="red", lw=1.0, label="range low")
        self.line_pe_v1, = self.ax_pe.plot([], [], "b-", lw=1.3, label="1m VWAP")
        self.line_pe_rh, = self.ax_pe.plot([], [], "--", color="green", lw=1.0, label="range high")
        self.line_pe_rl, = self.ax_pe.plot([], [], "--", color="red", lw=1.0, label="range low")
        self.ax_ce.legend(loc="upper left", fontsize=8)
        self.ax_pe.legend(loc="upper left", fontsize=8)
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._lines_ready = True

    def _update(self, frame):
        if not self._lines_ready: return
        with self.lock:
            ce_sym = self.atm_ce_symbol; pe_sym = self.atm_pe_symbol
            ce_data = dict(self.data.get(ce_sym, {})) if ce_sym else {}
            pe_data = dict(self.data.get(pe_sym, {})) if pe_sym else {}
        self._draw(self.ax_ce, self.line_ce_v1, self.line_ce_rh, self.line_ce_rl, ce_data, ce_sym)
        self._draw(self.ax_pe, self.line_pe_v1, self.line_pe_rh, self.line_pe_rl, pe_data, pe_sym)
        self.fig.canvas.draw_idle()

    def _draw(self, ax, line_v1, line_rh, line_rl, data, symbol):
        if not data or not data.get("times"):
            line_v1.set_data([], []); line_rh.set_data([], []); line_rl.set_data([], [])
            ax.set_title("(waiting for ATM symbol...)", fontsize=10, loc="left")
            return
        times = list(data["times"]); v1s = list(data["v1"])
        rhs = list(data["rh"]); rls = list(data["rl"])
        line_v1.set_data(times, v1s)
        line_rh.set_data(times, rhs)
        line_rl.set_data(times, rls)
        ax.set_xlim(times[0], times[-1])
        lo = min(min(v1s), min(rls)); hi = max(max(v1s), max(rhs))
        pad = (hi - lo) * 0.1 if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        for coll in list(ax.collections):
            try: coll.remove()
            except Exception: pass
        bks = list(data.get("breakouts", []))
        if bks:
            ax.scatter([c[0] for c in bks], [c[1] for c in bks],
                       marker="^", s=90, color="green", zorder=5)
        ax.set_title(symbol or "", fontsize=9, loc="left")
        ax.relim(visible_only=True)

    def show(self):
        if not (GRAPH_AVAILABLE and Config.ENABLE_LIVE_GRAPH):
            while self.running:
                time.sleep(1)
            return
        self._setup()
        self._anim = FuncAnimation(self.fig, self._update, interval=1000,
                                   cache_frame_data=False)
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "running", False))
        plt.show()


# ============================================================
# MAIN STRATEGY
# ============================================================

class VWAPPaperStrategy:
    def __init__(self, console: Console, graph: Optional[LiveVWAPGraph] = None):
        self.tz = pytz.timezone(Config.TIMEZONE)
        self.console = console

        self.rest = fyersModel.FyersModel(
            client_id=Config.CLIENT_ID, token=Config.ACCESS_TOKEN,
            is_async=False, log_path="",
        )
        self.ws = data_ws.FyersDataSocket(
            access_token=Config.ACCESS_TOKEN, log_path=Config.WS_LOG_PATH,
            litemode=Config.WS_LITEMODE, write_to_file=Config.WS_WRITE_TO_FILE,
            reconnect=Config.WS_RECONNECT,
            on_connect=self.on_open, on_close=self.on_close,
            on_error=self.on_error, on_message=self.on_message,
        )

        self.logger = CSVLogger()
        self.paper = PaperTrader(self.logger)
        self.graph = graph

        self.lock = threading.RLock()

        self.contracts: Dict[str, ContractInfo] = {}
        self.states: Dict[str, SymbolVWAPState] = {}
        self.latest_ticks: Dict[str, Tick] = {}
        self.atm_strike: Optional[int] = None
        self.subscribed_symbols = set()
        self.ws_connected = False
        self.running = True

        # Queued status messages, rendered inside the dashboard frame.
        self.status_lines: List[Tuple[float, str]] = []

        self.chain_thread = threading.Thread(
            target=self.option_chain_loop, name="option-chain-loop", daemon=True)
        self.dashboard_thread = threading.Thread(
            target=self.dashboard_loop, name="dashboard-loop", daemon=True)

    def push_status(self, line: str, hold_sec: float = 8.0):
        with self.lock:
            self.status_lines.append((time.time() + hold_sec, line))

    # --------------------------------------------------------
    # WS callbacks — must never print
    # --------------------------------------------------------
    def on_open(self):
        self.push_status(col("[WS] Connected.", C.GREEN))
        self.ws_connected = True
        symbols = self.get_desired_symbols()
        if symbols:
            try:
                self.ws.subscribe(symbols=list(symbols), data_type="SymbolUpdate")
                self.subscribed_symbols = set(symbols)
                self.push_status(col(f"[WS] Initial subscription: {len(symbols)} symbols", C.CYAN))
                self.logger.write_event({
                    "event_time": TimeHelper.now_str(), "event": "SUBSCRIBE",
                    "message": "Initial subscription", "symbol": "",
                    "strike": "", "option_type": "",
                    "details": ",".join(sorted(symbols)),
                })
            except Exception as e:
                self.push_status(col(f"[WS] Initial subscribe error: {e}", C.RED))
        self.ws.keep_running()

    def on_close(self, message):
        self.push_status(col(f"[WS] Connection closed: {message}", C.RED))
        self.ws_connected = False

    def on_error(self, message):
        self.push_status(col(f"[WS] Error: {message}", C.RED))

    # --------------------------------------------------------
    # Option chain
    # --------------------------------------------------------
    def fetch_option_chain(self):
        payload = {"symbol": Config.INDEX_SYMBOL, "strikecount": 3,
                   "timestamp": "", "greeks": "1"}
        response = self.rest.optionchain(data=payload)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected optionchain response type: {type(response)}")
        if response.get("s") != "ok":
            raise RuntimeError(f"Optionchain failed: {response}")
        data = response.get("data", {})
        if not data:
            raise RuntimeError("Optionchain data is empty.")
        return data

    def parse_atm_contracts(self, chain_data):
        chain = chain_data.get("optionsChain", [])
        if not chain:
            raise RuntimeError("optionsChain is empty.")

        underlying_ltp = None
        for item in chain:
            if item.get("option_type", "") == "" and safe_int(item.get("strike_price"), -1) == -1:
                underlying_ltp = safe_float(item.get("ltp"), 0.0); break
        if underlying_ltp is None or underlying_ltp <= 0:
            for item in chain:
                if not item.get("option_type"):
                    c = safe_float(item.get("ltp"), 0.0)
                    if c > 0:
                        underlying_ltp = c; break
        if underlying_ltp is None or underlying_ltp <= 0:
            raise RuntimeError("Could not determine NIFTY underlying LTP.")

        by_strike = {}
        for item in chain:
            ot = str(item.get("option_type", "")).upper()
            k = safe_int(item.get("strike_price"), -1)
            s = str(item.get("symbol", ""))
            if ot not in ("CE", "PE") or k <= 0 or not s:
                continue
            by_strike.setdefault(k, {})[ot] = s

        strikes = sorted(by_strike.keys())
        if len(strikes) < 3:
            raise RuntimeError(f"Not enough option strikes: {strikes}")

        atm = min(strikes, key=lambda s: abs(s - underlying_ltp))
        lower = [s for s in strikes if s < atm]
        upper = [s for s in strikes if s > atm]
        if not lower or not upper:
            raise RuntimeError(f"Unable to find strikes around ATM {atm}")
        atm_m1, atm_p1 = max(lower), min(upper)

        selected: Dict[str, ContractInfo] = {}
        for name, strike in [("ATM-1", atm_m1), ("ATM", atm), ("ATM+1", atm_p1)]:
            sd = by_strike.get(strike, {})
            for ot in ("CE", "PE"):
                sym = sd.get(ot)
                if not sym:
                    raise RuntimeError(f"Missing {ot} for strike {strike}")
                selected[sym] = ContractInfo(symbol=sym, strike=strike,
                                             option_type=ot, offset_name=name)

        self.push_status(col(
            f"[CHAIN] NIFTY LTP={underlying_ltp:.2f} | ATM={atm} | "
            f"ATM-1={atm_m1} | ATM+1={atm_p1}", C.MAGENTA))
        return atm, selected

    def get_desired_symbols(self):
        with self.lock:
            syms = set(self.contracts.keys())
            if self.paper.position is not None:
                syms.add(self.paper.position.symbol)
            return sorted(syms)

    def update_subscriptions(self, new_contracts):
        with self.lock:
            old = set(self.contracts.keys())
            new = set(new_contracts.keys())
            pos_sym = self.paper.position.symbol if self.paper.position else None
            to_unsub = old - new
            if pos_sym:
                to_unsub.discard(pos_sym)
            to_sub = new - self.subscribed_symbols
            try:
                if self.ws_connected and to_unsub:
                    self.ws.unsubscribe(symbols=sorted(to_unsub), data_type="SymbolUpdate")
                    self.subscribed_symbols -= to_unsub
                if self.ws_connected and to_sub:
                    self.ws.subscribe(symbols=sorted(to_sub), data_type="SymbolUpdate")
                    self.subscribed_symbols |= to_sub
            except Exception as e:
                self.push_status(col(f"[WS] Subscription update error: {e}", C.RED))

            prev = set(self.contracts.keys())
            for s in new:
                if s not in prev:
                    self.states[s] = SymbolVWAPState()
            for s in list(self.states.keys()):
                if s not in new and s != pos_sym:
                    del self.states[s]
            self.contracts = dict(new_contracts)
            for s in new:
                self.states.setdefault(s, SymbolVWAPState())

            if self.graph is not None:
                ce = pe = None
                for sym, info in self.contracts.items():
                    if info.offset_name == "ATM":
                        if info.option_type == "CE": ce = sym
                        elif info.option_type == "PE": pe = sym
                self.graph.set_atm(ce, pe)

    def refresh_option_chain(self):
        try:
            atm, new_contracts = self.parse_atm_contracts(self.fetch_option_chain())
            with self.lock:
                old = self.atm_strike
                self.atm_strike = atm
            if old != atm:
                self.push_status(col(f"[ATM] Changed: {old} -> {atm}", C.MAGENTA))
            self.update_subscriptions(new_contracts)
        except Exception as e:
            self.push_status(col(f"[CHAIN] Refresh error: {e}", C.RED))

    def option_chain_loop(self):
        while self.running:
            try:
                now_t = TimeHelper.current_time_only()
                if Config.MARKET_START <= now_t <= Config.MARKET_END:
                    self.refresh_option_chain()
                time.sleep(Config.OPTION_CHAIN_INTERVAL_SEC)
            except Exception as e:
                self.push_status(col(f"[CHAIN LOOP] {e}", C.RED))
                time.sleep(5)

    # --------------------------------------------------------
    # Tick handler
    # --------------------------------------------------------
    def on_message(self, message):
        try:
            if not isinstance(message, dict) or message.get("type") != "sf":
                return
            symbol = str(message.get("symbol", ""))
            if not symbol:
                return

            with self.lock:
                contract = self.contracts.get(symbol)
                if contract is None and self.paper.position is not None:
                    if symbol == self.paper.position.symbol:
                        contract = ContractInfo(
                            symbol=symbol, strike=self.paper.position.strike,
                            option_type=self.paper.position.option_type,
                            offset_name="OPEN_POSITION")
                if contract is None:
                    return

            ltt = safe_int(message.get("last_traded_time"), 0)
            ltp = safe_float(message.get("ltp"), 0.0)
            ltq = safe_int(message.get("last_traded_qty"), 0)
            vol = safe_int(message.get("vol_traded_today"), 0)
            bid = safe_float(message.get("bid_price"), 0.0)
            ask = safe_float(message.get("ask_price"), 0.0)
            if ltt <= 0 or ltp <= 0 or ltq <= 0:
                return

            ltt_dt = datetime.fromtimestamp(ltt, self.tz)
            if not (Config.MARKET_START <= ltt_dt.time() <= Config.MARKET_END):
                return

            tick = Tick(ltt=ltt, ltp=ltp, ltq=ltq, vol_traded_today=vol,
                        bid=bid, ask=ask, received_at=time.time())

            with self.lock:
                state = self.states.setdefault(symbol, SymbolVWAPState())
                if not state.update(tick):
                    return
                metrics = state.metrics()
                self.latest_ticks[symbol] = tick
                pos_name = self.paper.position.side if self.paper.position else "NONE"
                is_atm = (contract.offset_name == "ATM")

                self.logger.write_tick({
                    "log_time": TimeHelper.now_str(), "ltt": ltt,
                    "ltt_ist": TimeHelper.epoch_to_str(ltt), "symbol": symbol,
                    "strike": contract.strike, "option_type": contract.option_type,
                    "offset_name": contract.offset_name, "ltp": tick.ltp,
                    "ltq": tick.ltq, "vol_traded_today": tick.vol_traded_today,
                    "bid": tick.bid, "ask": tick.ask,
                    "rolling_qty_1m": state.qty_1m(),
                    "vwap_1m": metrics["vwap_1m"],
                    "range_high": metrics["range_high"],
                    "range_low": metrics["range_low"],
                    "range": metrics["range"],
                    "breakout": metrics["breakout"],
                    "armed": metrics["armed"],
                    "warmup_ready": metrics["warmup_ready"],
                    "is_atm": is_atm, "position": pos_name,
                })

                if Config.PRINT_EVERY_TICK:
                    self.push_status(
                        f"[TICK] {TimeHelper.epoch_to_str(tick.ltt)} "
                        f"{contract.offset_name} {contract.option_type} "
                        f"LTP={tick.ltp:.2f} 1m={metrics['vwap_1m']}",
                        hold_sec=2.0)

                if self.graph is not None:
                    self.graph.push(symbol=symbol, ts=ltt_dt,
                                    v1=metrics["vwap_1m"],
                                    rh=metrics["range_high"],
                                    rl=metrics["range_low"],
                                    breakout=metrics["breakout"])

                if (self.paper.position is not None
                        and contract.symbol == self.paper.position.symbol):
                    self.paper.update_trailing_and_check_exit(tick, metrics)
                    return

                if is_atm and metrics["breakout"] and metrics["warmup_ready"]:
                    self.handle_breakout_entry(contract, tick, metrics)

        except Exception as e:
            self.push_status(col(f"[WS MESSAGE ERROR] {e}", C.RED))

    def handle_breakout_entry(self, contract, tick, metrics):
        if self.paper.position is not None or not self.paper.can_enter():
            return
        reason = (f"ATM {contract.option_type} 1m VWAP broke above previous "
                  f"{Config.RANGE_LOOKBACK_SEC}s range high "
                  f"(vwap1={metrics['vwap_1m']:.4f} > high={metrics['range_high']:.4f})")
        self.push_status(col(
            f"[BREAKOUT] {contract.option_type} | {contract.symbol} | "
            f"1m={metrics['vwap_1m']:.4f} | high={metrics['range_high']:.4f}",
            C.BOLD + C.GREEN), hold_sec=8.0)
        self.paper.enter(contract, tick, metrics, reason)

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------
    def _active_status_lines(self):
        now = time.time()
        self.status_lines = [(t, s) for t, s in self.status_lines if t > now]
        return [s for _, s in self.status_lines]

    def build_frame(self) -> Tuple[str, Tuple]:
        with self.lock:
            W = Config.FRAME_WIDTH
            now = TimeHelper.now()
            atm = self.atm_strike
            pos = self.paper.position
            pnl = self.paper.realized_pnl
            trade_count = self.paper.trade_count
            ws_state = self.ws_connected

            if pos is None:
                pos_text = col("NONE", C.DIM)
            else:
                sl_txt = f"{pos.trailing_sl:.4f}" if pos.trailing_sl else "—"
                pos_text = col(
                    f"LONG {pos.side} {pos.strike} @ {pos.entry_price:.2f}  "
                    f"(sl={sl_txt}, high={pos.highest_vwap_since_entry:.4f})",
                    C.BOLD + C.YELLOW)

            pnl_color = C.GREEN if pnl >= 0 else C.RED
            status = col("CONNECTED" if ws_state else "DISCONNECTED",
                         C.GREEN if ws_state else C.RED)

            lines: List[str] = []
            lines.append(col("═" * W, C.CYAN))
            lines.append(col("  NIFTY ATM OPTION ROLLING VWAP BREAKOUT PAPER TRADER",
                             C.BOLD + C.WHITE))
            lines.append(col("═" * W, C.CYAN))
            lines.append(
                f"  {col('TIME', C.GRAY)}: {now.strftime('%Y-%m-%d %H:%M:%S')}   "
                f"{col('ATM', C.GRAY)}: {atm if atm is not None else '—'}   "
                f"{col('WS', C.GRAY)}: {status}   "
                f"{col('TRADES', C.GRAY)}: {trade_count}   "
                f"{col('P&L', C.GRAY)}: {col(f'{pnl:+.2f}', pnl_color)}")
            lines.append(f"  {col('POSITION', C.GRAY)}: {pos_text}")

            for s in self._active_status_lines():
                lines.append(f"  {s}")

            if time.time() < self.paper.banner_until and self.paper.banner_lines:
                lines.append("")
                lines.extend(self.paper.banner_lines)
                lines.append("")

            header = (f"  {'CONTRACT':<9}{'STRIKE':>8}{'LTP':>11}{'1m VWAP':>11}"
                      f"{'RANGE HI':>11}{'RANGE LO':>11}{'RANGE':>10}"
                      f"{'ARMED':>8}{'READY':>8}{'BRK':>6}")
            lines.append(col("─" * W, C.GRAY))
            lines.append(col(header, C.BOLD + C.WHITE))
            lines.append(col("─" * W, C.GRAY))

            sorted_contracts = sorted(self.contracts.items(),
                                      key=lambda x: (x[1].strike, x[1].option_type))
            for symbol, contract in sorted_contracts:
                state = self.states.get(symbol)
                tick = self.latest_ticks.get(symbol)
                if state is None:
                    continue
                m = state.metrics()
                ltp = tick.ltp if tick else None
                label = f"{contract.offset_name} {contract.option_type}"

                v1 = m["vwap_1m"]; rh = m["range_high"]
                vwap_txt = fmt_num(v1, 11, 4)
                if v1 is not None and rh is not None and v1 > rh:
                    vwap_txt = col(vwap_txt, C.GREEN)

                armed_txt = (col("Y".rjust(8), C.GREEN) if m["armed"]
                             else col("N".rjust(8), C.GRAY))
                ready_txt = (col("Y".rjust(8), C.GREEN) if m["warmup_ready"]
                             else col("N".rjust(8), C.GRAY))
                brk_txt = (col("▲".rjust(6), C.BOLD + C.GREEN) if m["breakout"]
                           else " " * 6)

                line = (f"  {label:<9}{contract.strike:>8}"
                        f"{fmt_num(ltp, 11, 2)}"
                        f"{vwap_txt}"
                        f"{fmt_num(m['range_high'], 11, 4)}"
                        f"{fmt_num(m['range_low'], 11, 4)}"
                        f"{fmt_num(m['range'], 10, 4)}"
                        f"{armed_txt}{ready_txt}{brk_txt}")
                if contract.offset_name == "ATM":
                    line = C.BOLD + line + C.RESET
                lines.append(line)

            lines.append(col("─" * W, C.GRAY))
            lines.append(
                f"  Legend:  {col('ARMED', C.GREEN)}=waiting for breakout  "
                f"{col('READY', C.GREEN)}=range warmed up  "
                f"{col('▲', C.GREEN)}=breakout this tick  "
                f"{col('—', C.GRAY)}=no data yet")
            lines.append(col("═" * W, C.CYAN))

            stable_sig = (ws_state, atm, trade_count,
                          round(pnl, 2),
                          pos.symbol if pos else None,
                          tuple(sorted((s, m["vwap_1m"] is not None)
                                       for s, m in
                                       ((sym, self.states[sym].metrics())
                                        for sym in self.states))),
                          )
            return "\n".join(lines), stable_sig

    def dashboard_loop(self):
        # First frame: clear startup text, then render.
        self.console.clear()
        while self.running:
            try:
                time.sleep(Config.DASHBOARD_INTERVAL_SEC)
                if not self.running:
                    break
                frame, sig = self.build_frame()
                self.console.render_frame(frame, sig)
                self.paper.force_exit_if_needed(
                    self.latest_ticks, self.collect_metrics_lookup())
            except Exception as e:
                self.push_status(col(f"[DASHBOARD ERROR] {e}", C.RED))

    def collect_metrics_lookup(self):
        result = {}
        with self.lock:
            for sym, state in self.states.items():
                m = state.metrics()
                if m["vwap_1m"] is not None:
                    result[sym] = m
        return result

    # --------------------------------------------------------
    # Startup / shutdown
    # --------------------------------------------------------
    def start(self):
        self.console.clear()

        W = 88
        banner = "\n".join([
            col("═" * W, C.CYAN),
            col("  NIFTY ATM OPTION ROLLING VWAP BREAKOUT PAPER TRADER",
                C.BOLD + C.WHITE),
            col("═" * W, C.CYAN),
            f"  {col('Paper mode', C.CYAN):<32}: {Config.PAPER_MODE}",
            f"  {col('Index', C.CYAN):<32}: {Config.INDEX_SYMBOL}",
            f"  {col('1m VWAP window', C.CYAN):<32}: {Config.VWAP_FAST_SEC}s",
            f"  {col('Range lookback', C.CYAN):<32}: {Config.RANGE_LOOKBACK_SEC}s",
            f"  {col('Option chain interval', C.CYAN):<32}: {Config.OPTION_CHAIN_INTERVAL_SEC}s",
            f"  {col('Lot size', C.CYAN):<32}: {Config.LOT_SIZE}",
            f"  {col('Logs', C.CYAN):<32}: {Config.LOG_DIR}",
            col("─" * W, C.CYAN),
            f"  {col('ENTRY', C.GREEN)}: buy ATM CE/PE when its 1m VWAP breaks "
            f"above previous 5m range high of 1m VWAP",
            f"  {col('EXIT ', C.YELLOW)}: trailing SL = highest 1m VWAP − 5m range; "
            f"exit when 1m VWAP <= trailing SL",
            col("═" * W, C.CYAN),
            "",
        ])
        self.console.write_static(banner)

        if "YOUR_" in Config.CLIENT_ID or "YOUR_" in Config.ACCESS_TOKEN:
            raise RuntimeError("Set Fyers client ID and access token first.")

        self.push_status(col("[INIT] Fetching option chain...", C.CYAN))
        self.refresh_option_chain()

        self.chain_thread.start()
        self.dashboard_thread.start()

        self.push_status(col("[WS] Connecting...", C.CYAN))
        self.ws.connect()

    def shutdown(self):
        self.running = False
        # Let the dashboard stop cleanly before we write anything else.
        if self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=Config.DASHBOARD_INTERVAL_SEC + 1.0)

        try:
            if self.paper.position is not None:
                pos = self.paper.position
                tick = self.latest_ticks.get(pos.symbol)
                if tick:
                    state = self.states.get(pos.symbol)
                    metrics = state.metrics() if state else {
                        "vwap_1m": None, "range_high": None,
                        "range_low": None, "range": None}
                    self.paper.exit(tick, metrics, "PROGRAM_SHUTDOWN")
        except Exception:
            pass

        try:
            self.ws.close()
        except Exception:
            pass

        self.console.write_static("\n" + col("[SHUTDOWN] Done.", C.GREEN))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    console = Console()
    graph = LiveVWAPGraph() if GRAPH_AVAILABLE else None
    if graph is not None:
        graph.running = True

    strategy = VWAPPaperStrategy(console=console, graph=graph)

    t = threading.Thread(target=strategy.start, name="strategy-main", daemon=True)
    t.start()

    try:
        if graph is not None and Config.ENABLE_LIVE_GRAPH:
            graph.show()
        else:
            while strategy.running:
                time.sleep(1)
    except KeyboardInterrupt:
        console.write_static("\n" + col("[MAIN] Ctrl+C received.", C.YELLOW))
    except Exception as e:
        console.write_static("\n" + col(f"[FATAL] {e}", C.RED))
        traceback.print_exc()
    finally:
        if graph is not None:
            graph.running = False
        strategy.shutdown()