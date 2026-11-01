#!/usr/bin/env python3
"""
NIFTY ATM Premium-Crash Bot — FYERS API v3 (Windows + Linux)

Strategy:
  1. Fetch index open from Quotes API.
  2. Round it to the nearest configurable strike step.
  3. Fetch the selected expiry option chain and resolve the exact ATM CE/PE.
  4. Fetch each option's own session open using Quotes API.
  5. Stream both option LTPS over FYERS Data WebSocket.
  6. CE and PE signals are independent. Each may trigger once per session.
  7. At a 50% drop from that option's session open, buy an OTM hedge at
     approximately 25% of the signal LTP and sell the selected ATM option
     using POST /api/v3/multi-order/sync (SDK: place_basket_orders).
  8. After broker positions confirm matching fills, place a BUY-side GTT OCO
     on the short: SL above LTP, target below LTP.
  9. When the short is flat, cancel any still-pending OCO and close the hedge.
 10. State is persisted across process restarts and calendar days.

IMPORTANT:
  - Live orders are disabled by default. Set LIVE_TRADING=true only after
    testing with the exact account, lot size, expiry and symbols.
  - Credentials are read from FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN.
  - This bot does not guarantee fills or atomic execution. The multi-order
    endpoint returns per-leg results; the bot reconciles actual broker state.
  - F&O GTT protection is limited by the contract expiry. Overnight carry
    does not mean protection beyond the option's expiry.
  - Verify current NIFTY lot size before enabling live orders.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Use the native file-locking module for the current operating system.
# Linux/macOS: fcntl. Windows: msvcrt.
if os.name == "nt":
    import msvcrt
else:
    import fcntl

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws


# =============================================================================
# CONFIGURATION
# =============================================================================
# CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "").strip()
# ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN", "").strip()

CLIENT_ID    = ""
ACCESS_TOKEN = ""


NIFTY_SYMBOL = os.getenv("NIFTY_SYMBOL", "NSE:NIFTY50-INDEX").strip()
STRIKE_STEP = int(os.getenv("STRIKE_STEP", "50"))
STRIKE_COUNT = 50                       # FYERS maximum: 50 on either side
PRODUCT_TYPE = "MARGIN"                # FYERS derivative carry-forward product
LOT_SIZE = int(os.getenv("LOT_SIZE", "65"))  # VERIFY current exchange lot size
LOTS = int(os.getenv("LOTS", "1"))
ORDER_QTY = LOT_SIZE * LOTS

DROP_PCT = float(os.getenv("DROP_PCT", "0.50"))
HEDGE_RATIO = float(os.getenv("HEDGE_RATIO", "0.25"))
TARGET_RS = float(os.getenv("TARGET_RS", "1.0"))
TICK_SIZE = float(os.getenv("OPTION_TICK_SIZE", "0.05"))

# LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() in {"1", "true", "yes", "y"}
LIVE_TRADING = "true"
RECONCILE_INTERVAL_SEC = int(os.getenv("RECONCILE_INTERVAL_SEC", "10"))
# Protective OCO GTTs are NOT cancelled at end of day. They remain active
# overnight until triggered, explicitly cancelled, or contract expiry.
GTT_EXIT_SETTLE_SEC = 10
DASHBOARD_INTERVAL_SEC = float(os.getenv("DASHBOARD_INTERVAL_SEC", "1"))
# Poll index LTP during an active session; option LTPs are streamed by WebSocket.
INDEX_QUOTE_REFRESH_SEC = float(os.getenv("INDEX_QUOTE_REFRESH_SEC", "2"))
# When no active session baseline exists (weekend / pre-open / setup retry),
# fetch a display-only option-chain snapshot and refresh its quotes separately.
PREVIEW_CHAIN_REFRESH_SEC = float(os.getenv("PREVIEW_CHAIN_REFRESH_SEC", "60"))
PREVIEW_QUOTE_REFRESH_SEC = float(os.getenv("PREVIEW_QUOTE_REFRESH_SEC", "5"))
ORDER_WAIT_SEC = int(os.getenv("ORDER_WAIT_SEC", "20"))
WS_RECONNECT_RETRIES = 50
MAX_HEDGE_SEARCH = 50

DATA_DIR = Path(os.getenv("BOT_DATA_DIR", str(Path.home() / "nifty_premium_bot")))
STATE_FILE = Path(os.getenv("BOT_STATE_FILE", str(DATA_DIR / "state.json")))
LOCK_FILE = Path(os.getenv("BOT_LOCK_FILE", str(DATA_DIR / "bot.lock")))
LOG_FILE = Path(os.getenv("BOT_LOG_FILE", str(DATA_DIR / "bot.log")))

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

# =============================================================================
# LOGGING / SHARED STATE
# =============================================================================
DATA_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

EVENTS = deque(maxlen=12)
EVENT_LOCK = threading.Lock()
STATE_LOCK = threading.RLock()
API_LOCK = threading.RLock()
PRICE_LOCK = threading.RLock()

CURRENT_PRICES: Dict[str, float] = {}
LAST_TICK_AT: Dict[str, float] = {}
WS_STATUS: Dict[str, Any] = {"connected": False, "message": "not started", "last_event": 0.0}
BROKER_SNAPSHOT: Dict[str, Any] = {
    "ok": False,
    "positions": {},
    "orders": [],
    "gtt": [],
    "updated_at": 0.0,
    "error": "not synced",
}


def now_ist() -> datetime:
    return datetime.now(IST)


def now_utc() -> datetime:
    return datetime.now(UTC)


def today_key() -> str:
    return now_ist().strftime("%Y-%m-%d")


def stamp() -> str:
    return now_ist().strftime("%Y-%m-%d %H:%M:%S IST")


def emit(level: str, message: str) -> None:
    line = f"[{now_ist().strftime('%H:%M:%S')}] [{level}] {message}"
    with EVENT_LOCK:
        EVENTS.appendleft(line)
    getattr(logging, "warning" if level in {"WARN", "ERROR"} else "info")(message)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def strict_qty_int(value: Any, label: str) -> int:
    """Parse broker quantity without truncating decimal or malformed values."""
    if value in (None, ""):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} quantity returned by broker: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Non-integer {label} quantity returned by broker: {value!r}")
    return int(numeric)


def market_session_open() -> bool:
    now = now_ist()
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)


def round_to_tick(price: float, direction: str = "nearest") -> float:
    if price <= 0:
        raise ValueError(f"Invalid price: {price}")
    units = price / TICK_SIZE
    if direction == "up":
        units = math.ceil(units - 1e-10)
    elif direction == "down":
        units = math.floor(units + 1e-10)
    else:
        units = round(units)
    return round(units * TICK_SIZE, 2)


def nearest_atm(index_open: float) -> int:
    # Half-way values round upward; Python's built-in round uses bankers' rounding.
    return int(math.floor(index_open / STRIKE_STEP + 0.5) * STRIKE_STEP)


def trade_key(session_date: str, leg: str) -> str:
    return f"{session_date}:{leg}"


def order_tag(session_date: str, leg: str, role: str) -> str:
    # FYERS orderTag: 1–30 alphanumeric characters.
    return f"NPC{session_date.replace('-', '')}{leg}{role}"


# =============================================================================
# STATE PERSISTENCE — atomically replaced; survives restarts if BOT_DATA_DIR
# is on persistent EBS storage.
# =============================================================================
def empty_state() -> Dict[str, Any]:
    return {"schema": 2, "sessions": {}, "trades": {}}


def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return empty_state()
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("State root is not an object")
        data.setdefault("schema", 2)
        data.setdefault("sessions", {})
        data.setdefault("trades", {})
        return data
    except Exception as exc:
        emit("ERROR", f"Could not read state file {STATE_FILE}: {exc}; refusing to overwrite it")
        raise


def save_state() -> None:
    with STATE_LOCK:
        tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(STATE, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)


STATE = empty_state()


def acquire_single_instance_lock():
    """Acquire a non-blocking process lock on both Windows and Linux."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.touch(exist_ok=True)
    handle = LOCK_FILE.open("r+b")

    try:
        # msvcrt.locking locks a byte range, so ensure byte zero exists first.
        if os.name == "nt":
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            "Another bot process already holds the lock, or the lock file "
            f"cannot be locked: {LOCK_FILE} ({exc})"
        ) from exc

    # Keep this handle open for the entire bot lifetime. Closing it releases
    # the operating-system lock. The file itself is deliberately not deleted.
    return handle


def release_single_instance_lock(handle) -> None:
    """Release the process lock using the matching platform API."""
    if handle is None:
        return
    try:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def clear_terminal() -> None:
    """Clear an interactive terminal on Windows and POSIX systems."""
    if not sys.stdout.isatty():
        return

    if os.name == "nt":
        # Enable ANSI/VT sequences when supported (Windows 10/11 terminals).
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            stdout_handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint()
            if kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode)):
                enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                if kernel32.SetConsoleMode(stdout_handle, mode.value | enable_vt):
                    sys.stdout.write("\033[2J\033[H")
                    return
        except Exception:
            pass

        # Fallback for older Windows consoles without VT support.
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")


# =============================================================================
# FYERS CLIENT / API HELPERS
# =============================================================================
if not CLIENT_ID or not ACCESS_TOKEN:
    print("ERROR: Set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN environment variables.", file=sys.stderr)
    sys.exit(2)

fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    is_async=False,
    log_path="",
)


def api_call(method_name: str, *args, **kwargs) -> Dict[str, Any]:
    method = getattr(fyers, method_name)
    with API_LOCK:
        response = method(*args, **kwargs)
    if not isinstance(response, dict):
        raise RuntimeError(f"FYERS {method_name} returned non-dict response: {response!r}")
    return response


def response_ok(resp: Dict[str, Any]) -> bool:
    return resp.get("s") == "ok"


def get_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Quotes API supports up to 50 symbols per request."""
    result: Dict[str, Dict[str, Any]] = {}
    symbols = list(dict.fromkeys(s for s in symbols if s))
    for start in range(0, len(symbols), 50):
        batch = symbols[start:start + 50]
        resp = api_call("quotes", {"symbols": ",".join(batch)})
        if not response_ok(resp):
            raise RuntimeError(f"Quotes API failed: {resp}")
        for row in resp.get("d", []):
            if row.get("s") != "ok":
                emit("WARN", f"Quote failed for {row.get('n')}: {row}")
                continue
            symbol = row.get("n")
            values = row.get("v", {})
            if symbol:
                result[symbol] = values
    return result


def quote_ltp(values: Dict[str, Any]) -> float:
    """Read last traded price from a FYERS Quotes API value object."""
    return safe_float(values.get("lp", values.get("ltp", 0.0)))


def fetch_index_quote() -> Dict[str, float]:
    """Fetch NIFTY's session open and current LTP in one Quotes API request."""
    values = get_quotes([NIFTY_SYMBOL]).get(NIFTY_SYMBOL)
    if not values:
        raise RuntimeError(f"No quote row returned for {NIFTY_SYMBOL}")
    open_price = safe_float(values.get("open_price"))
    ltp = quote_ltp(values)
    if open_price <= 0:
        raise RuntimeError(f"Invalid NIFTY open_price in Quotes response: {values}")
    return {"open": open_price, "ltp": ltp}


def fetch_index_open() -> float:
    # Kept as a small helper for any caller that only needs the opening value.
    return fetch_index_quote()["open"]


def fetch_option_chain(timestamp: str = "", strikecount: int = STRIKE_COUNT) -> Dict[str, Any]:
    payload = {
        "symbol": NIFTY_SYMBOL,
        "strikecount": min(max(1, int(strikecount)), 50),
        "timestamp": str(timestamp),
        "greeks": "1",
    }
    resp = api_call("optionchain", data=payload)
    if not response_ok(resp):
        raise RuntimeError(f"Option Chain API failed: {resp}")
    return resp.get("data", {})


def choose_nearest_expiry(expiry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    today = now_ist().date()
    candidates = []
    for item in expiry_data:
        try:
            expiry_date = datetime.strptime(item["date"], "%d-%m-%Y").date()
            expiry_epoch = str(item["expiry"])
        except (KeyError, TypeError, ValueError):
            continue
        if expiry_date >= today:
            candidates.append((expiry_date, item))
    if not candidates:
        raise RuntimeError(f"No current/future expiry available: {expiry_data}")
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def find_exact_atm(chain: List[Dict[str, Any]], atm: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ce = next((x for x in chain if safe_float(x.get("strike_price"), -999) == atm and x.get("option_type") == "CE"), None)
    pe = next((x for x in chain if safe_float(x.get("strike_price"), -999) == atm and x.get("option_type") == "PE"), None)
    if not ce or not pe:
        found = sorted({safe_float(x.get("strike_price")) for x in chain if x.get("option_type") in {"CE", "PE"}})
        raise RuntimeError(
            f"Opening ATM {atm} not in returned chain. Returned range: "
            f"{found[0] if found else 'n/a'} to {found[-1] if found else 'n/a'}. "
            "The underlying may have moved beyond the chain's 50-strike window."
        )
    if not ce.get("symbol") or not pe.get("symbol"):
        raise RuntimeError(f"Missing option symbol in ATM chain rows: CE={ce}, PE={pe}")
    return ce, pe


def build_daily_session() -> Dict[str, Any]:
    """Build opening baseline once; option opens come from Quotes API, not chain LTP."""
    index_quote = fetch_index_quote()
    index_open = index_quote["open"]
    index_ltp = index_quote["ltp"]
    atm = nearest_atm(index_open)

    initial = fetch_option_chain(timestamp="", strikecount=1)
    expiry = choose_nearest_expiry(initial.get("expiryData", []))
    chain_data = fetch_option_chain(timestamp=str(expiry["expiry"]), strikecount=STRIKE_COUNT)
    ce, pe = find_exact_atm(chain_data.get("optionsChain", []), atm)
    ce_symbol, pe_symbol = ce["symbol"], pe["symbol"]

    option_quotes = get_quotes([ce_symbol, pe_symbol])
    ce_q = option_quotes.get(ce_symbol, {})
    pe_q = option_quotes.get(pe_symbol, {})
    ce_open = safe_float(ce_q.get("open_price"))
    pe_open = safe_float(pe_q.get("open_price"))
    ce_ltp = quote_ltp(ce_q)
    pe_ltp = quote_ltp(pe_q)
    if ce_open <= 0 or pe_open <= 0:
        raise RuntimeError(
            f"Option session open not available yet. CE open={ce_open}, PE open={pe_open}. "
            "Will retry; the bot will not substitute option-chain LTP."
        )

    session_date = today_key()
    session = {
        "date": session_date,
        "created_at": stamp(),
        "index_symbol": NIFTY_SYMBOL,
        "index_open": index_open,
        "atm": atm,
        "expiry_date": expiry.get("date"),
        "expiry_timestamp": str(expiry["expiry"]),
        "expiry_flag": expiry.get("expiry_flag", ""),
        "ce_symbol": ce_symbol,
        "pe_symbol": pe_symbol,
        "ce_open": ce_open,
        "pe_open": pe_open,
        "signals": {"CE": "WAITING", "PE": "WAITING"},
    }

    # Seed the dashboard immediately; subsequent CE/PE updates arrive by
    # WebSocket and the index LTP is refreshed periodically through Quotes API.
    with PRICE_LOCK:
        if index_ltp > 0:
            CURRENT_PRICES[NIFTY_SYMBOL] = index_ltp
        if ce_ltp > 0:
            CURRENT_PRICES[ce_symbol] = ce_ltp
        if pe_ltp > 0:
            CURRENT_PRICES[pe_symbol] = pe_ltp
    return session


def build_dashboard_preview() -> Dict[str, Any]:
    """
    Fetch the latest available NIFTY/ATM CE/PE quote snapshot for display only.

    This is deliberately NOT saved as a trading session and must NEVER trigger
    orders. It lets the terminal show the last available broker prices on
    weekends, before 09:15, after 15:30, or while today's session setup retries.
    During a closed market, Quotes API values generally represent the last
    available trading session, not live prices.
    """
    index_quote = fetch_index_quote()
    index_open = safe_float(index_quote.get("open"))
    index_ltp = safe_float(index_quote.get("ltp"))
    atm = nearest_atm(index_open)

    initial = fetch_option_chain(timestamp="", strikecount=1)
    expiry = choose_nearest_expiry(initial.get("expiryData", []))
    chain_data = fetch_option_chain(timestamp=str(expiry["expiry"]), strikecount=STRIKE_COUNT)
    ce, pe = find_exact_atm(chain_data.get("optionsChain", []), atm)
    ce_symbol, pe_symbol = ce["symbol"], pe["symbol"]

    option_quotes = get_quotes([ce_symbol, pe_symbol])
    ce_q = option_quotes.get(ce_symbol, {})
    pe_q = option_quotes.get(pe_symbol, {})

    preview = {
        "fetched_date": today_key(),
        "fetched_at": stamp(),
        "index_symbol": NIFTY_SYMBOL,
        "index_open": index_open,
        "index_ltp": index_ltp,
        "atm": atm,
        "expiry_date": expiry.get("date"),
        "ce_symbol": ce_symbol,
        "pe_symbol": pe_symbol,
        "ce_open": safe_float(ce_q.get("open_price")),
        "pe_open": safe_float(pe_q.get("open_price")),
        "ce_ltp": quote_ltp(ce_q),
        "pe_ltp": quote_ltp(pe_q),
    }

    # Seed values so the dashboard has prices immediately, even if the market
    # is closed and therefore no WebSocket ticks are arriving.
    with PRICE_LOCK:
        if index_ltp > 0:
            CURRENT_PRICES[NIFTY_SYMBOL] = index_ltp
        if preview["ce_ltp"] > 0:
            CURRENT_PRICES[ce_symbol] = preview["ce_ltp"]
        if preview["pe_ltp"] > 0:
            CURRENT_PRICES[pe_symbol] = preview["pe_ltp"]
    return preview


# =============================================================================
# BROKER SNAPSHOT / RECONCILIATION DATA
# =============================================================================
def list_from_response(resp: Dict[str, Any], *possible_keys: str) -> List[Dict[str, Any]]:
    for key in possible_keys:
        value = resp.get(key)
        if isinstance(value, list):
            return value
    return []


def build_broker_snapshot() -> Dict[str, Any]:
    positions_resp = api_call("positions")
    orders_resp = api_call("orderbook")
    gtt_resp = api_call("gtt_orderbook")

    for name, resp in (("positions", positions_resp), ("orderbook", orders_resp), ("gtt_orderbook", gtt_resp)):
        if not response_ok(resp):
            raise RuntimeError(f"FYERS {name} failed: {resp}")

    position_map: Dict[str, Dict[str, Any]] = {}
    for p in positions_resp.get("netPositions", []):
        symbol = p.get("symbol")
        if not symbol:
            continue
        product = p.get("productType", "")
        if product and product != PRODUCT_TYPE:
            continue
        if symbol not in position_map:
            position_map[symbol] = {"netQty": 0, "rows": []}
        position_map[symbol]["netQty"] += strict_qty_int(p.get("netQty"), f"position {symbol}")
        position_map[symbol]["rows"].append(p)

    orders = list_from_response(orders_resp, "orderBook", "data")
    gtt_orders = list_from_response(gtt_resp, "orderBook", "data")
    return {
        "ok": True,
        "positions": position_map,
        "orders": orders,
        "gtt": gtt_orders,
        "updated_at": time.time(),
        "error": "",
    }


def position_qty(snapshot: Dict[str, Any], symbol: str) -> int:
    value = snapshot.get("positions", {}).get(symbol, {}).get("netQty", 0)
    return strict_qty_int(value, f"position {symbol}")


def require_lot_quantity(qty: int, label: str, allow_zero: bool = True) -> int:
    """Reject non-integer or non-lot quantities; never truncate/round quantities."""
    value = strict_qty_int(qty, label)
    if value < 0:
        raise ValueError(f"{label} must be a non-negative integer quantity; got {qty!r}")
    if not allow_zero and value == 0:
        raise ValueError(f"{label} must be greater than zero")
    if LOT_SIZE <= 0 or value % LOT_SIZE != 0:
        raise ValueError(
            f"{label}={value} is not a multiple of LOT_SIZE={LOT_SIZE}. "
            "Automated order adjustment is blocked; inspect the broker position manually."
        )
    return value


def quantity_to_lots(qty: int, label: str) -> int:
    value = require_lot_quantity(abs(safe_int(qty)), label)
    return value // LOT_SIZE


def order_side(order: Dict[str, Any]) -> int:
    return safe_int(order.get("side", order.get("tran_side", 0)))


def pending_short_buy_orders(snapshot: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
    """Open BUY orders on the short symbol, normally the child order of its GTT."""
    matches = []
    for order in snapshot.get("orders", []):
        if order.get("symbol") != symbol or order_side(order) != 1:
            continue
        product = order.get("productType", order.get("product_type", ""))
        if product and product != PRODUCT_TYPE:
            continue
        if order_status(order) in {4, 6}:
            matches.append(order)
    return matches


def order_status(order: Dict[str, Any]) -> int:
    return safe_int(order.get("status", order.get("order_status", 0)))


def gtt_status(order: Dict[str, Any]) -> int:
    return safe_int(order.get("ord_status", order.get("status", 0)))


def order_filled_qty(order: Dict[str, Any]) -> int:
    return strict_qty_int(order.get("filledQty", order.get("filled_qty", 0)), f"filledQty order {order.get('id', '')}")


def order_qty(order: Dict[str, Any]) -> int:
    return strict_qty_int(order.get("qty", order.get("quantity", 0)), f"qty order {order.get('id', '')}")


def find_order(snapshot: Dict[str, Any], order_id: str = "", tag: str = "") -> Optional[Dict[str, Any]]:
    wanted_tag = str(tag or "")
    if wanted_tag.startswith("1:") or wanted_tag.startswith("2:"):
        wanted_tag = wanted_tag[2:]
    for order in snapshot.get("orders", []):
        oid = str(order.get("id", ""))
        otag = str(order.get("orderTag", order.get("order_tag", order.get("tag", ""))))
        normalized_tag = otag[2:] if otag.startswith(("1:", "2:")) else otag
        if order_id and oid == str(order_id):
            return order
        if wanted_tag and normalized_tag == wanted_tag:
            return order
    return None


def find_gtt(snapshot: Dict[str, Any], order_id: str = "", symbol: str = "") -> Optional[Dict[str, Any]]:
    for order in snapshot.get("gtt", []):
        oid = str(order.get("id", ""))
        if order_id and oid == str(order_id):
            return order
        if symbol and order.get("symbol") == symbol and safe_int(order.get("gtt_oco_ind")) == 2:
            if gtt_status(order) in {4, 6}:
                return order
    return None


def symbol_has_pending_order(snapshot: Dict[str, Any], symbol: str) -> bool:
    return any(
        o.get("symbol") == symbol and order_status(o) in {4, 6}
        for o in snapshot.get("orders", [])
    )


def symbol_has_pending_gtt(snapshot: Dict[str, Any], symbol: str) -> bool:
    return any(
        o.get("symbol") == symbol and gtt_status(o) in {4, 6}
        for o in snapshot.get("gtt", [])
    )


# =============================================================================
# STRATEGY / ORDER HELPERS
# =============================================================================
def find_hedge_contract(session: Dict[str, Any], leg: str, selected_ltp: float) -> Tuple[str, int, float]:
    """Choose same-side OTM contract whose live LTP is closest to selected LTP / 4."""
    target = selected_ltp * HEDGE_RATIO
    chain_data = fetch_option_chain(
        timestamp=session["expiry_timestamp"],
        strikecount=STRIKE_COUNT,
    )
    chain = chain_data.get("optionsChain", [])
    atm = safe_int(session["atm"])
    option_type = leg
    candidates = []
    for row in chain:
        if row.get("option_type") != option_type or not row.get("symbol"):
            continue
        strike = safe_int(row.get("strike_price"))
        if leg == "CE" and strike > atm:
            candidates.append((strike, row["symbol"]))
        elif leg == "PE" and strike < atm:
            candidates.append((strike, row["symbol"]))
    candidates = candidates[:MAX_HEDGE_SEARCH]
    if not candidates:
        raise RuntimeError(f"No OTM {leg} hedge contracts found in selected expiry chain")

    quotes = get_quotes([symbol for _, symbol in candidates])
    best: Optional[Tuple[str, int, float, float]] = None
    for strike, symbol in candidates:
        ltp = safe_float(quotes.get(symbol, {}).get("lp"))
        if ltp <= 0:
            continue
        diff = abs(ltp - target)
        candidate = (diff, symbol, strike, ltp)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError(f"No valid hedge LTPs found for {leg}; target={target:.2f}")
    return best[1], best[2], best[3]


def entry_order_payload(symbol: str, side: int, qty: int, tag: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "qty": qty,
        "type": 2,                 # Market / MPP
        "side": side,              # 1 buy, -1 sell
        "productType": PRODUCT_TYPE,
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "orderTag": tag,
    }


def parse_multi_order_response(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = resp.get("data", [])
    parsed = []
    if isinstance(rows, list):
        for row in rows:
            body = row.get("body", row) if isinstance(row, dict) else {}
            parsed.append({
                "statusCode": row.get("statusCode") if isinstance(row, dict) else None,
                "code": body.get("code"),
                "s": body.get("s"),
                "id": body.get("id"),
                "message": body.get("message", body.get("messageText", "")),
            })
    return parsed


def submit_entry_basket(trade: Dict[str, Any]) -> None:
    require_lot_quantity(ORDER_QTY, "entry quantity", allow_zero=False)
    session_date = trade["date"]
    leg = trade["leg"]
    hedge_tag = order_tag(session_date, leg, "H")
    short_tag = order_tag(session_date, leg, "S")
    payload = [
        entry_order_payload(trade["hedge_symbol"], 1, ORDER_QTY, hedge_tag),
        entry_order_payload(trade["short_symbol"], -1, ORDER_QTY, short_tag),
    ]

    with STATE_LOCK:
        trade["status"] = "ENTRY_SUBMITTING"
        trade["entry_submitted_at"] = time.time()
        trade["hedge_order_tag"] = hedge_tag
        trade["short_order_tag"] = short_tag
        save_state()

    emit("ORDER", f"Submitting SYNC multi-order: BUY hedge {trade['hedge_symbol']} then SELL {trade['short_symbol']} qty={ORDER_QTY}")
    try:
        # FYERS SDK method maps to POST /api/v3/multi-order/sync.
        resp = api_call("place_basket_orders", data=payload)
    except Exception as exc:
        with STATE_LOCK:
            trade["status"] = "ENTRY_UNKNOWN"
            trade["last_error"] = str(exc)
            save_state()
        emit("ERROR", f"Multi-order request outcome unknown; will reconcile tags, NOT resubmit: {exc}")
        return

    parsed = parse_multi_order_response(resp)
    if not parsed:
        with STATE_LOCK:
            trade["status"] = "ENTRY_UNKNOWN"
            trade["entry_response"] = resp
            save_state()
        emit("ERROR", f"Multi-order returned no per-leg result; will reconcile, not resubmit: {resp}")
        return

    with STATE_LOCK:
        if len(parsed) > 0:
            trade["hedge_order_id"] = parsed[0].get("id")
        if len(parsed) > 1:
            trade["short_order_id"] = parsed[1].get("id")
        trade["entry_response"] = resp
        trade["entry_leg_results"] = parsed
        trade["status"] = "ENTRY_PENDING"
        save_state()

    for i, row in enumerate(parsed):
        emit("ORDER", f"Multi-order leg {i + 1}: code={row.get('code')} id={row.get('id')} statusCode={row.get('statusCode')} message={row.get('message')}")


def place_market_order(symbol: str, side: int, qty: int, tag: str) -> Dict[str, Any]:
    qty = require_lot_quantity(qty, f"order quantity for {symbol}", allow_zero=False)
    if side not in {-1, 1}:
        raise ValueError(f"Invalid order side {side}; expected -1 (SELL) or 1 (BUY)")
    payload = entry_order_payload(symbol, side, qty, tag)
    return api_call("place_order", data=payload)


def submit_gtt_oco(trade: Dict[str, Any], qty: int, current_ltp: float) -> None:
    qty = require_lot_quantity(qty, "GTT OCO quantity", allow_zero=False)
    signal_ltp = safe_float(trade["signal_ltp"])
    open_px = safe_float(trade["option_open"])
    distance = open_px * DROP_PCT
    sl_trigger = round_to_tick(signal_ltp + distance / 2.0, "up")
    target_trigger = round_to_tick(signal_ltp - TARGET_RS, "down")

    if target_trigger <= 0:
        raise RuntimeError(f"Calculated target is non-positive: {target_trigger}")
    if not (sl_trigger > current_ltp and target_trigger < current_ltp):
        raise OcoLevelCrossed(
            f"OCO levels no longer bracket current LTP. current={current_ltp:.2f}, "
            f"SL={sl_trigger:.2f}, target={target_trigger:.2f}. "
            "Short will be closed/reviewed rather than placing an invalid OCO."
        )

    # Both legs BUY to close the short. BUY limit prices are set one tick above
    # their triggers to improve the chance of execution after triggering.
    sl_limit = round_to_tick(sl_trigger + TICK_SIZE, "up")
    target_limit = round_to_tick(target_trigger + TICK_SIZE, "up")
    payload = {
        "side": 1,
        "symbol": trade["short_symbol"],
        "productType": PRODUCT_TYPE,
        "orderInfo": {
            "leg1": {"price": sl_limit, "triggerPrice": sl_trigger, "qty": qty},
            "leg2": {"price": target_limit, "triggerPrice": target_trigger, "qty": qty},
        },
    }

    with STATE_LOCK:
        trade["sl_trigger"] = sl_trigger
        trade["target_trigger"] = target_trigger
        trade["oco_qty"] = qty
        trade["status"] = "OCO_SUBMITTING"
        trade["oco_submitted_at"] = time.time()
        save_state()

    emit("ORDER", f"Placing BUY OCO {trade['short_symbol']} qty={qty}: SL trigger={sl_trigger:.2f}, target trigger={target_trigger:.2f}")
    try:
        resp = api_call("place_gtt_order", data=payload)
    except Exception as exc:
        with STATE_LOCK:
            trade["status"] = "OCO_UNKNOWN"
            trade["last_error"] = str(exc)
            save_state()
        emit("ERROR", f"GTT OCO result unknown; broker reconciliation required; will not duplicate or send a second exit: {exc}")
        raise OcoSubmissionUnknown(str(exc)) from exc

    code = safe_int(resp.get("code"))
    with STATE_LOCK:
        trade["gtt_id"] = resp.get("id")
        trade["gtt_response"] = resp
        trade["status"] = "OCO_PENDING" if code in {1101, 201} else "OCO_REJECTED"
        save_state()
    if code in {1101, 201}:
        emit("ORDER", f"GTT OCO submitted: id={resp.get('id')} code={code}; verifying GTT order book")
    else:
        emit("ERROR", f"GTT OCO rejected: {resp}")
        raise RuntimeError(f"GTT OCO rejected: {resp}")


# =============================================================================
# BOT
# =============================================================================
class OcoLevelCrossed(RuntimeError):
    pass


class OcoSubmissionUnknown(RuntimeError):
    pass


class PremiumCrashBot:
    def __init__(self):
        self.monitored_sessions = set()
        self.session_locks: Dict[str, threading.Lock] = {}
        self.running = True
        self.last_broker_sync = 0.0
        self.last_dashboard = 0.0
        self.last_setup_attempt = 0.0
        self.last_index_quote = 0.0
        self.preview_snapshot: Optional[Dict[str, Any]] = None
        self.last_preview_setup = 0.0
        self.last_preview_quote = 0.0
        self.preview_error = ""
        self.last_error = ""

    def get_session(self, date: str) -> Optional[Dict[str, Any]]:
        with STATE_LOCK:
            return STATE.get("sessions", {}).get(date)

    def create_session_if_needed(self) -> Optional[Dict[str, Any]]:
        date = today_key()
        existing = self.get_session(date)
        if existing:
            return existing
        if not market_session_open():
            return None
        # Opening option quotes can take a few seconds to become available.
        # Throttle retries instead of hammering the Quotes/Option Chain APIs.
        if time.time() - self.last_setup_attempt < 15:
            return None
        self.last_setup_attempt = time.time()
        try:
            session = build_daily_session()
        except Exception as exc:
            self.last_error = str(exc)
            emit("WARN", f"Daily setup not ready; retrying: {exc}")
            return None
        with STATE_LOCK:
            STATE.setdefault("sessions", {})[date] = session
            save_state()
        emit("DATA", (
            f"Session ready: {NIFTY_SYMBOL} open={session['index_open']:.2f}, ATM={session['atm']}, "
            f"expiry={session['expiry_date']}, CE open={session['ce_open']:.2f}, PE open={session['pe_open']:.2f}"
        ))
        return session

    def refresh_dashboard_preview(self) -> None:
        """Refresh a non-trading quote snapshot when today's baseline is absent."""
        if self.get_session(today_key()):
            return
        now = time.time()
        if now - self.last_preview_setup < PREVIEW_CHAIN_REFRESH_SEC:
            return
        self.last_preview_setup = now
        try:
            preview = build_dashboard_preview()
            self.preview_snapshot = preview
            self.preview_error = ""
            emit(
                "DATA",
                "Display-only quote snapshot: "
                f"NIFTY open={preview['index_open']:.2f} LTP={preview['index_ltp']:.2f}, "
                f"ATM={preview['atm']}, CE open={preview['ce_open']:.2f}, "
                f"PE open={preview['pe_open']:.2f}. No trading baseline created.",
            )
        except Exception as exc:
            self.preview_error = str(exc)
            message = f"Display quote snapshot unavailable: {exc}"
            if message != self.last_error:
                emit("WARN", message)
            self.last_error = message

    def refresh_preview_quotes(self) -> None:
        """Refresh LTPs for the display-only snapshot via the Quotes API."""
        preview = self.preview_snapshot
        if not preview or self.get_session(today_key()):
            return
        now = time.time()
        if now - self.last_preview_quote < PREVIEW_QUOTE_REFRESH_SEC:
            return
        self.last_preview_quote = now
        symbols = [NIFTY_SYMBOL, preview.get("ce_symbol"), preview.get("pe_symbol")]
        try:
            quotes = get_quotes(symbols)
            with PRICE_LOCK:
                for symbol in symbols:
                    if not symbol:
                        continue
                    values = quotes.get(symbol, {})
                    ltp = quote_ltp(values)
                    if ltp > 0:
                        CURRENT_PRICES[symbol] = ltp
        except Exception as exc:
            message = f"Display quote LTP refresh failed: {exc}"
            if message != self.last_error:
                emit("WARN", message)
            self.last_error = message

    def prime_session_quotes(self, session: Dict[str, Any]) -> None:
        """Populate the dashboard immediately after startup/restart."""
        symbols = [NIFTY_SYMBOL, session.get("ce_symbol"), session.get("pe_symbol")]
        try:
            quotes = get_quotes(symbols)
            with PRICE_LOCK:
                for symbol in symbols:
                    values = quotes.get(symbol, {}) if symbol else {}
                    ltp = quote_ltp(values)
                    if ltp > 0:
                        CURRENT_PRICES[symbol] = ltp
        except Exception as exc:
            emit("WARN", f"Initial quote refresh failed; waiting for live updates: {exc}")

    def refresh_index_ltp(self) -> None:
        """Refresh only the underlying index LTP at a controlled API interval."""
        now = time.time()
        if now - self.last_index_quote < INDEX_QUOTE_REFRESH_SEC:
            return
        self.last_index_quote = now
        try:
            values = get_quotes([NIFTY_SYMBOL]).get(NIFTY_SYMBOL, {})
            ltp = quote_ltp(values)
            if ltp > 0:
                with PRICE_LOCK:
                    CURRENT_PRICES[NIFTY_SYMBOL] = ltp
        except Exception as exc:
            # Avoid flooding the terminal with the same transient quote error.
            message = f"Index Quotes API refresh failed: {exc}"
            if message != self.last_error:
                emit("WARN", message)
            self.last_error = message

    def start_websocket(self, session: Dict[str, Any]) -> None:
        date = session["date"]
        if date in self.monitored_sessions:
            return
        self.prime_session_quotes(session)
        self.monitored_sessions.add(date)
        thread = threading.Thread(target=self._ws_worker, args=(session,), daemon=True, name=f"fyers-ws-{date}")
        thread.start()

    def _ws_worker(self, session: Dict[str, Any]) -> None:
        ce_symbol = session["ce_symbol"]
        pe_symbol = session["pe_symbol"]

        def on_message(message):
            try:
                if not isinstance(message, dict):
                    return
                if message.get("type") != "sf":
                    return
                symbol = message.get("symbol")
                ltp = safe_float(message.get("ltp"))
                if not symbol or ltp <= 0:
                    return
                now = time.time()
                with PRICE_LOCK:
                    CURRENT_PRICES[symbol] = ltp
                    LAST_TICK_AT[symbol] = now
                WS_STATUS["last_event"] = now
                if symbol == ce_symbol:
                    self.check_trigger(session, "CE", ltp)
                elif symbol == pe_symbol:
                    self.check_trigger(session, "PE", ltp)
            except Exception as exc:
                emit("WARN", f"WebSocket message parse/processing error: {exc}")

        def on_connect():
            WS_STATUS["connected"] = True
            WS_STATUS["message"] = f"connected ({session['date']})"
            emit("DATA", f"WebSocket connected; subscribing {ce_symbol} and {pe_symbol}")
            try:
                socket.subscribe(symbols=[ce_symbol, pe_symbol], data_type="SymbolUpdate")
            except Exception as exc:
                emit("ERROR", f"WebSocket subscribe failed: {exc}")

        def on_close(*args):
            WS_STATUS["connected"] = False
            WS_STATUS["message"] = "closed/reconnecting"
            emit("WARN", f"WebSocket closed: {args}")

        def on_error(error):
            WS_STATUS["message"] = f"error: {error}"
            emit("ERROR", f"WebSocket error: {error}")

        socket = data_ws.FyersDataSocket(
            access_token=f"{CLIENT_ID}:{ACCESS_TOKEN}",
            log_path="",
            litemode=True,
            write_to_file=False,
            reconnect=True,
            reconnect_retry=WS_RECONNECT_RETRIES,
            on_connect=on_connect,
            on_close=on_close,
            on_error=on_error,
            on_message=on_message,
        )
        try:
            socket.keep_running()
        except Exception as exc:
            WS_STATUS["connected"] = False
            WS_STATUS["message"] = str(exc)
            emit("ERROR", f"WebSocket worker stopped: {exc}")

    def check_trigger(self, session: Dict[str, Any], leg: str, ltp: float) -> None:
        date = session["date"]
        lock = self.session_locks.setdefault(trade_key(date, leg), threading.Lock())
        if not lock.acquire(blocking=False):
            return
        try:
            trigger_open = safe_float(session[f"{leg.lower()}_open"])
            trigger = trigger_open * (1.0 - DROP_PCT)
            with STATE_LOCK:
                signal_state = session.setdefault("signals", {}).get(leg, "WAITING")
                if signal_state != "WAITING":
                    return
                if ltp > trigger:
                    return
                session["signals"][leg] = "TRIGGERED"
                session["signals"][f"{leg}_trigger_ltp"] = ltp
                session["signals"][f"{leg}_trigger_time"] = stamp()
                save_state()
            emit("SIGNAL", f"{leg} trigger: LTP={ltp:.2f} <= 50% level={trigger:.2f}; open={trigger_open:.2f}")
            thread = threading.Thread(
                target=self.execute_signal,
                args=(session.copy(), leg, ltp, trigger_open),
                daemon=True,
                name=f"entry-{date}-{leg}",
            )
            thread.start()
        finally:
            lock.release()

    def execute_signal(self, session: Dict[str, Any], leg: str, signal_ltp: float, option_open: float) -> None:
        date = session["date"]
        key = trade_key(date, leg)
        short_symbol = session[f"{leg.lower()}_symbol"]
        trade = {
            "key": key,
            "date": date,
            "leg": leg,
            "status": "PREPARING",
            "short_symbol": short_symbol,
            "short_strike": safe_int(session["atm"]),
            "option_open": option_open,
            "signal_ltp": signal_ltp,
            "signal_time": stamp(),
            "qty_requested": ORDER_QTY,
            "expiry_date": session["expiry_date"],
            "expiry_timestamp": session["expiry_timestamp"],
        }
        with STATE_LOCK:
            STATE.setdefault("trades", {})[key] = trade
            save_state()

        if not LIVE_TRADING:
            try:
                hedge_symbol, hedge_strike, hedge_ltp = find_hedge_contract(session, leg, signal_ltp)
                trade.update({
                    "hedge_symbol": hedge_symbol,
                    "hedge_strike": hedge_strike,
                    "hedge_ltp_at_selection": hedge_ltp,
                    "status": "DRY_RUN",
                })
                with STATE_LOCK:
                    save_state()
                emit("DRYRUN", (
                    f"Would BUY hedge {hedge_symbol} LTP={hedge_ltp:.2f} and SELL {short_symbol}; "
                    f"hedge target={signal_ltp * HEDGE_RATIO:.2f}; no orders sent"
                ))
            except Exception as exc:
                trade["status"] = "DRY_RUN_ERROR"
                trade["last_error"] = str(exc)
                with STATE_LOCK:
                    save_state()
                emit("ERROR", f"Dry-run hedge selection failed for {leg}: {exc}")
            return

        try:
            hedge_symbol, hedge_strike, hedge_ltp = find_hedge_contract(session, leg, signal_ltp)
            trade.update({
                "hedge_symbol": hedge_symbol,
                "hedge_strike": hedge_strike,
                "hedge_ltp_at_selection": hedge_ltp,
            })
            self.sync_broker(force=True)
            snapshot = BROKER_SNAPSHOT
            if not snapshot.get("ok"):
                raise RuntimeError("Broker snapshot unavailable; refusing entry")
            if position_qty(snapshot, short_symbol) != 0 or symbol_has_pending_order(snapshot, short_symbol) or symbol_has_pending_gtt(snapshot, short_symbol):
                raise RuntimeError(f"Short contract already has a position or pending order: {short_symbol}")
            if position_qty(snapshot, hedge_symbol) != 0 or symbol_has_pending_order(snapshot, hedge_symbol) or symbol_has_pending_gtt(snapshot, hedge_symbol):
                raise RuntimeError(f"Hedge contract already has a position or pending order: {hedge_symbol}")
            submit_entry_basket(trade)
        except Exception as exc:
            with STATE_LOCK:
                trade["status"] = "ENTRY_BLOCKED"
                trade["last_error"] = str(exc)
                save_state()
            emit("ERROR", f"{leg} entry aborted: {exc}")

    def sync_broker(self, force: bool = False) -> None:
        if not force and time.time() - self.last_broker_sync < RECONCILE_INTERVAL_SEC:
            return
        try:
            snap = build_broker_snapshot()
            BROKER_SNAPSHOT.update(snap)
            self.last_broker_sync = time.time()
        except Exception as exc:
            BROKER_SNAPSHOT["ok"] = False
            BROKER_SNAPSHOT["error"] = str(exc)
            self.last_error = str(exc)
            emit("ERROR", f"Broker reconciliation API failed: {exc}")

    def reconcile_all_trades(self) -> None:
        if not BROKER_SNAPSHOT.get("ok"):
            return
        with STATE_LOCK:
            trades = list(STATE.get("trades", {}).values())
        for trade in trades:
            try:
                self.reconcile_trade(trade)
            except Exception as exc:
                emit("ERROR", f"Trade reconciliation failed for {trade.get('key')}: {exc}")
                with STATE_LOCK:
                    trade["last_error"] = str(exc)
                    save_state()

    def reconcile_trade(self, trade: Dict[str, Any]) -> None:
        status = trade.get("status")
        if status == "PREPARING":
            self.manual_review(
                trade,
                "Process stopped during trade preparation before order submission could be confirmed; inspect broker before any manual retry",
            )
            return
        if status in {"CLOSED", "DRY_RUN", "DRY_RUN_ERROR", "ENTRY_BLOCKED", "ENTRY_FAILED", "MANUAL_REVIEW", "OCO_REJECTED"}:
            return

        snapshot = BROKER_SNAPSHOT
        if not snapshot.get("ok"):
            return
        if not trade.get("short_symbol") or not trade.get("hedge_symbol"):
            return

        if status in {"ENTRY_SUBMITTING", "ENTRY_UNKNOWN", "ENTRY_PENDING", "RECOVERY_PENDING", "OCO_SUBMITTING", "OCO_UNKNOWN", "OCO_PENDING"}:
            self.reconcile_entry_and_oco(trade, snapshot)
            return

        active_states = {
            "OCO_ACTIVE", "OCO_TRIGGERED", "OCO_CANCELLED_1505",
            "HEDGE_REBALANCE_SUBMITTING", "HEDGE_REBALANCE_PENDING",
            "SHORT_EXIT_SUBMITTING", "SHORT_EXIT_PENDING",
            "HEDGE_EXIT_SUBMITTING", "HEDGE_EXIT_PENDING",
        }
        if status in active_states:
            self.reconcile_active_spread(trade, snapshot)

    def reconcile_active_spread(self, trade: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        """Keep the hedge equal to the remaining short in whole lots.

        The GTT exits only the short option. When its resulting BUY order is
        working/part-filled, broker positions determine how many hedge lots
        must be sold. No hedge order is sent from a GTT trigger event alone.
        """
        short_symbol = trade["short_symbol"]
        hedge_symbol = trade["hedge_symbol"]
        short_net = position_qty(snapshot, short_symbol)
        hedge_net = position_qty(snapshot, hedge_symbol)

        if short_net > 0:
            self.manual_review(trade, f"Short symbol unexpectedly has a LONG position: {short_symbol} netQty={short_net}")
            return
        if hedge_net < 0:
            self.manual_review(trade, f"Hedge symbol unexpectedly has a SHORT position: {hedge_symbol} netQty={hedge_net}")
            return

        try:
            short_qty = require_lot_quantity(abs(short_net), f"short position {short_symbol}")
            hedge_qty = require_lot_quantity(hedge_net, f"hedge position {hedge_symbol}")
            entry_short_qty = require_lot_quantity(
                safe_int(trade.get("short_filled_qty", trade.get("qty_requested", ORDER_QTY))),
                "recorded entry short quantity",
                allow_zero=False,
            )
            require_lot_quantity(
                safe_int(trade.get("hedge_filled_qty", trade.get("qty_requested", ORDER_QTY))),
                "recorded entry hedge quantity",
                allow_zero=False,
            )
            pending_exit_orders = pending_short_buy_orders(snapshot, short_symbol)
            for order in pending_exit_orders:
                require_lot_quantity(order_qty(order), f"pending short-exit order qty {order.get('id')}", allow_zero=False)
                require_lot_quantity(order_filled_qty(order), f"pending short-exit filledQty {order.get('id')}")
                remaining = strict_qty_int(order.get("remainingQuantity", order.get("remaining_quantity", 0)), f"remainingQuantity order {order.get('id', '')}")
                if remaining:
                    require_lot_quantity(remaining, f"pending short-exit remainingQuantity {order.get('id')}")
        except Exception as exc:
            self.manual_review(trade, f"Lot-size validation failed; no automated quantity adjustment will be sent: {exc}")
            return

        legacy_hedge_tag = str(trade.get("hedge_exit_order_tag") or "")
        if legacy_hedge_tag:
            legacy_order = find_order(snapshot, trade.get("hedge_exit_order_id", ""), legacy_hedge_tag)
            if legacy_order and order_status(legacy_order) in {4, 6}:
                return
            if legacy_order and order_status(legacy_order) == 2:
                if short_net == 0 and hedge_net == 0 and not pending_exit_orders:
                    trade["status"] = "CLOSED"
                    trade["closed_at"] = stamp()
                    with STATE_LOCK:
                        save_state()
                    emit("DATA", f"Legacy hedge exit confirmed; trade closed: {trade['key']}")
                else:
                    self.manual_review(trade, f"Legacy hedge exit reports filled but positions are short={short_net}, hedge={hedge_net}; inspect broker")
                return
            if legacy_order and order_status(legacy_order) in {1, 5, 7}:
                self.manual_review(trade, f"Legacy hedge exit order did not complete: {legacy_order}")
                return
            if not legacy_order and time.time() - safe_float(trade.get("hedge_exit_submitted_at"), time.time()) > 60:
                self.manual_review(trade, "Legacy hedge exit order is missing from broker order book; inspect before retrying")
            return

        # Validate every broker-reported pending/filled quantity too. We never
        # round a partial execution to a lot or submit an odd-lot adjustment.
        for order in pending_exit_orders:
            require_lot_quantity(order_qty(order), f"pending short-exit order qty {order.get('id')}", allow_zero=False)
            require_lot_quantity(order_filled_qty(order), f"pending short-exit filledQty {order.get('id')}")
            remaining = safe_int(order.get("remainingQuantity", order.get("remaining_quantity", 0)))
            if remaining:
                require_lot_quantity(remaining, f"pending short-exit remainingQuantity {order.get('id')}")

        gtt_id = str(trade.get("gtt_id") or "")
        gtt = find_gtt(snapshot, gtt_id, short_symbol)
        gtt_state = gtt_status(gtt) if gtt else 0
        if gtt and gtt_state in {4, 6}:
            try:
                require_lot_quantity(gtt.get("qty", 0), f"GTT first-leg qty {gtt_id}", allow_zero=False)
                second_qty = strict_qty_int(gtt.get("qty2", gtt.get("qty_2", 0)), f"GTT second-leg qty {gtt_id}")
                if second_qty:
                    require_lot_quantity(second_qty, f"GTT second-leg qty {gtt_id}", allow_zero=False)
            except Exception as exc:
                self.manual_review(trade, f"GTT lot-size validation failed; no hedge order will be sent: {exc}")
                return

        if gtt_state == 2:
            if not trade.get("gtt_triggered"):
                trade["gtt_triggered"] = True
                trade["gtt_triggered_at"] = time.time()
                emit("DATA", f"GTT OCO triggered for {short_symbol}; monitoring short BUY execution and hedge lots")
                with STATE_LOCK:
                    save_state()
        elif pending_exit_orders:
            # The GTT may have disappeared from the GTT book after triggering.
            if not trade.get("gtt_triggered"):
                trade["gtt_triggered"] = True
                trade["gtt_triggered_at"] = time.time()
                emit("DATA", f"Short BUY exit order detected for {short_symbol}; treating GTT as triggered")
                with STATE_LOCK:
                    save_state()

        # An open BUY order on the short contract together with a still-pending
        # OCO is ambiguous (the OCO may be racing its trigger). Do not trade the
        # hedge until FYERS reports a definitive GTT state.
        if gtt_state == 6 and pending_exit_orders:
            self.manual_review(
                trade,
                "A short BUY order and a pending OCO are both visible for the same contract. "
                "This could over-buy the short; inspect/cancel the duplicate protection before hedge adjustment.",
            )
            return

        if gtt_state == 4:
            emit("WARN", f"GTT {gtt_id} is in transit; waiting for a definitive broker status")
            return

        # A fully flat short must be handled even when its OCO is still pending
        # (for example, an emergency/manual short exit). Cancel and verify that
        # OCO first, otherwise it could later BUY and create a new long position.
        if short_qty == 0:
            if pending_exit_orders:
                emit("WARN", f"Short position is flat but a BUY exit order is still pending for {short_symbol}; hedge exit postponed")
                return
            if gtt_state in {4, 6}:
                if not self.cancel_gtt_and_confirm(trade, snapshot):
                    return
                snapshot = BROKER_SNAPSHOT
                if not snapshot.get("ok"):
                    return
                if position_qty(snapshot, short_symbol) != 0:
                    self.manual_review(trade, "Short position changed while cancelling stale OCO; broker state must be reviewed")
                    return
                hedge_net = position_qty(snapshot, hedge_symbol)
                if hedge_net < 0:
                    self.manual_review(trade, f"Hedge became short while cancelling OCO: {hedge_net}")
                    return
                try:
                    hedge_qty = require_lot_quantity(hedge_net, "refreshed hedge position")
                except Exception as exc:
                    self.manual_review(trade, f"Lot-size validation failed after OCO cancellation: {exc}")
                    return
            self.rebalance_hedge_to_short(trade, snapshot, 0, hedge_qty)
            return

        # If GTT is still pending, it must still protect the full short quantity.
        if gtt_state == 6 and not pending_exit_orders:
            if short_qty != entry_short_qty or hedge_qty != short_qty:
                self.manual_review(
                    trade,
                    f"OCO is still pending but positions differ from the protected entry size: short={short_qty}, "
                    f"hedge={hedge_qty}, entryShort={entry_short_qty}. Do not adjust until the pending GTT is reconciled.",
                )
                return
            trade["status"] = "OCO_ACTIVE"
            return

        if gtt_state in {1, 5, 7} and not pending_exit_orders:
            self.manual_review(
                trade,
                f"Protective GTT is no longer active (status={gtt_state}) while short quantity {short_qty} remains; "
                "manual protection/recovery is required.",
            )
            return

        if not gtt and not pending_exit_orders and not trade.get("gtt_triggered"):
            age = time.time() - safe_float(trade.get("oco_submitted_at"), time.time())
            if age > 30:
                self.manual_review(trade, "Protective GTT is missing from broker GTT book and no short BUY exit order is visible; inspect broker protection")
            return

        if trade.get("gtt_triggered") or pending_exit_orders or gtt_state == 2:
            if pending_exit_orders:
                self.rebalance_hedge_to_short(trade, snapshot, short_qty, hedge_qty)
                return

            triggered_at = safe_float(trade.get("gtt_triggered_at"), time.time())
            if time.time() - triggered_at < GTT_EXIT_SETTLE_SEC:
                return
            self.manual_review(
                trade,
                f"GTT has triggered but no working BUY exit order is visible and short quantity {short_qty} remains. "
                "The hedge was not force-closed; inspect the short exit order immediately.",
            )
            return

        if not gtt and short_qty > 0:
            self.manual_review(trade, "GTT and short BUY exit order are both missing while the short remains open")

        # Legacy state from the previous version: its 15:05 code may have
        # cancelled the GTT. Do not silently carry an unprotected short.
        if trade.get("status") == "OCO_CANCELLED_1505" and short_qty > 0:
            self.manual_review(trade, "Legacy 15:05 logic had cancelled this trade's GTT; short is still open and needs a fresh protective exit")

    def cancel_gtt_and_confirm(self, trade: Dict[str, Any], snapshot: Dict[str, Any]) -> bool:
        gtt_id = str(trade.get("gtt_id") or "")
        if not gtt_id:
            self.manual_review(trade, "Short is flat but no GTT ID is saved; verify there is no live OCO before hedge exit")
            return False
        gtt = find_gtt(snapshot, gtt_id, trade.get("short_symbol", ""))
        if not gtt:
            # An absent GTT is terminal from the order-book perspective. The
            # caller has already checked that no short BUY order is pending.
            return True
        status = gtt_status(gtt)
        if status in {1, 2, 5, 7}:
            return True
        if status == 4:
            emit("WARN", f"GTT {gtt_id} is in transit; not selling hedge until its state is definitive")
            return False
        if status != 6:
            self.manual_review(trade, f"Cannot determine GTT {gtt_id} status={status}; hedge exit blocked")
            return False

        try:
            response = api_call("cancel_gtt_order", data={"id": gtt_id})
            emit("ORDER", f"Cancelling stale OCO after short became flat: {response}")
        except Exception as exc:
            self.manual_review(trade, f"Short is flat but stale OCO cancellation outcome is unknown: {exc}")
            return False
        if response.get("s") != "ok" or safe_int(response.get("code")) != 1103:
            self.manual_review(trade, f"Could not cancel stale OCO after short became flat: {response}")
            return False

        self.sync_broker(force=True)
        refreshed = find_gtt(BROKER_SNAPSHOT, gtt_id, trade.get("short_symbol", ""))
        refreshed_status = gtt_status(refreshed) if refreshed else 0
        if refreshed_status == 1 or refreshed is None:
            return not pending_short_buy_orders(BROKER_SNAPSHOT, trade.get("short_symbol", ""))
        if refreshed_status == 2:
            trade["gtt_triggered"] = True
            trade["gtt_triggered_at"] = trade.get("gtt_triggered_at") or time.time()
            emit("WARN", f"GTT {gtt_id} triggered during cancellation; rechecking its BUY exit before hedge liquidation")
            return False
        emit("WARN", f"Stale OCO cancellation not yet confirmed (status={refreshed_status}); hedge exit postponed")
        return False

    def reconcile_hedge_rebalance_order(self, trade: Dict[str, Any], snapshot: Dict[str, Any]) -> bool:
        """Return True while a previously submitted hedge adjustment is unresolved."""
        tag = str(trade.get("hedge_rebalance_order_tag") or "")
        if not tag:
            return False
        order = find_order(snapshot, trade.get("hedge_rebalance_order_id", ""), tag)
        submitted_at = safe_float(trade.get("hedge_rebalance_submitted_at"), time.time())
        if not order:
            if time.time() - submitted_at > 60:
                self.manual_review(trade, "Hedge rebalance order is missing from broker order book; refusing to submit a duplicate")
            return True

        trade["hedge_rebalance_order_id"] = order.get("id")
        try:
            require_lot_quantity(order_qty(order), f"hedge rebalance order qty {order.get('id')}", allow_zero=False)
            require_lot_quantity(order_filled_qty(order), f"hedge rebalance filledQty {order.get('id')}")
        except Exception as exc:
            self.manual_review(trade, f"Hedge rebalance order quantity is not a whole-lot amount: {exc}")
            return True
        status = order_status(order)
        if status in {4, 6}:
            return True
        if status in {1, 5, 7}:
            self.manual_review(trade, f"Hedge rebalance order did not complete: {order}")
            return True
        if status != 2:
            self.manual_review(trade, f"Unknown hedge rebalance order status={status}: {order}")
            return True

        current_hedge = position_qty(snapshot, trade["hedge_symbol"])
        if current_hedge < 0:
            self.manual_review(trade, f"Hedge became a short position after rebalance: {current_hedge}")
            return True
        require_lot_quantity(current_hedge, "hedge position after rebalance")
        intended = safe_int(trade.get("hedge_rebalance_target_qty"), -1)
        if current_hedge == intended:
            trade["last_hedge_rebalance_order_id"] = order.get("id")
            trade["last_hedge_rebalance_qty"] = order_qty(order)
            trade["hedge_rebalance_order_id"] = ""
            trade["hedge_rebalance_order_tag"] = ""
            trade["hedge_rebalance_submitted_at"] = 0
            trade["hedge_rebalance_target_qty"] = 0
            trade["status"] = "OCO_TRIGGERED" if trade.get("gtt_triggered") else "OCO_ACTIVE"
            with STATE_LOCK:
                save_state()
            return False

        if time.time() - submitted_at > 15:
            self.manual_review(
                trade,
                f"Hedge rebalance order is marked filled but position is {current_hedge}; expected {intended}. "
                "Check broker positions before any further order.",
            )
        return True

    def rebalance_hedge_to_short(
        self,
        trade: Dict[str, Any],
        snapshot: Dict[str, Any],
        target_short_qty: int,
        current_hedge_qty: int,
    ) -> None:
        target = require_lot_quantity(target_short_qty, "target hedge quantity")
        current = require_lot_quantity(current_hedge_qty, "current hedge quantity")

        if self.reconcile_hedge_rebalance_order(trade, snapshot):
            return

        difference = target - current
        if difference == 0:
            trade["hedge_matched_qty"] = target
            if target == 0:
                trade["status"] = "CLOSED"
                trade["closed_at"] = stamp()
                emit("DATA", f"Both legs verified flat; trade closed: {trade['key']}")
                with STATE_LOCK:
                    save_state()
            else:
                trade["status"] = "OCO_TRIGGERED" if trade.get("gtt_triggered") else "OCO_ACTIVE"
                with STATE_LOCK:
                    save_state()
            return

        quantity = require_lot_quantity(abs(difference), "hedge rebalance order quantity", allow_zero=False)
        side = 1 if difference > 0 else -1
        action = "BUY" if side == 1 else "SELL"
        sequence = safe_int(trade.get("hedge_rebalance_sequence")) + 1
        trade["hedge_rebalance_sequence"] = sequence
        base_tag = order_tag(trade["date"], trade["leg"], "B")
        tag = f"{base_tag}{sequence:03d}"
        trade["hedge_rebalance_order_tag"] = tag
        trade["hedge_rebalance_order_id"] = ""
        trade["hedge_rebalance_submitted_at"] = time.time()
        trade["hedge_rebalance_target_qty"] = target
        trade["hedge_rebalance_side"] = side
        trade["hedge_rebalance_qty"] = quantity
        trade["status"] = "HEDGE_REBALANCE_SUBMITTING"
        with STATE_LOCK:
            save_state()

        emit(
            "ORDER",
            f"Reconcile spread: short={target} ({target // LOT_SIZE} lots), hedge={current} ({current // LOT_SIZE} lots); "
            f"{action} hedge {quantity} ({quantity // LOT_SIZE} lots)",
        )
        try:
            response = place_market_order(trade["hedge_symbol"], side, quantity, tag)
            if response.get("s") != "ok" or safe_int(response.get("code")) not in {1101, 201}:
                self.manual_review(trade, f"Hedge rebalance order rejected: {response}")
                return
            trade["hedge_rebalance_order_id"] = response.get("id") or ""
            trade["status"] = "HEDGE_REBALANCE_PENDING"
            with STATE_LOCK:
                save_state()
            emit("ORDER", f"Hedge rebalance submitted: id={response.get('id')} code={response.get('code')}")
        except Exception as exc:
            self.manual_review(trade, f"Hedge rebalance outcome unknown; refusing duplicate order: {exc}")

    def reconcile_entry_and_oco(self, trade: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        status = trade.get("status")
        hedge_tag = trade.get("hedge_order_tag") or order_tag(trade["date"], trade["leg"], "H")
        short_tag = trade.get("short_order_tag") or order_tag(trade["date"], trade["leg"], "S")
        hedge_order = find_order(snapshot, trade.get("hedge_order_id", ""), hedge_tag)
        short_order = find_order(snapshot, trade.get("short_order_id", ""), short_tag)

        if hedge_order and not trade.get("hedge_order_id"):
            trade["hedge_order_id"] = hedge_order.get("id")
        if short_order and not trade.get("short_order_id"):
            trade["short_order_id"] = short_order.get("id")

        if status in {"ENTRY_SUBMITTING", "ENTRY_UNKNOWN", "ENTRY_PENDING"}:
            results = trade.get("entry_leg_results", [])
            hedge_result = results[0] if len(results) > 0 else {}
            short_result = results[1] if len(results) > 1 else {}
            hedge_rejected = bool(hedge_result) and (hedge_result.get("s") == "error" or safe_int(hedge_result.get("code")) not in {1101, 201})
            short_rejected = bool(short_result) and (short_result.get("s") == "error" or safe_int(short_result.get("code")) not in {1101, 201})
            hedge_done = (hedge_order is not None and order_status(hedge_order) in {1, 2, 5, 7}) or hedge_rejected
            short_done = (short_order is not None and order_status(short_order) in {1, 2, 5, 7}) or short_rejected
            if not hedge_done or not short_done:
                age = time.time() - safe_float(trade.get("entry_submitted_at"), time.time())
                if age > 60:
                    self.manual_review(trade, "Could not confirm both entry orders in broker order book; not resubmitting to avoid duplicates")
                return
            # Allow the positions endpoint a short interval to reflect terminal fills.
            if not trade.get("entry_terminal_seen_at"):
                trade["entry_terminal_seen_at"] = time.time()
                with STATE_LOCK:
                    save_state()
                return
            if time.time() - safe_float(trade.get("entry_terminal_seen_at")) < 3:
                return
            self.align_hedge_and_short(trade, snapshot)
            return

        if status == "RECOVERY_PENDING":
            recovery_tag = trade.get("recovery_order_tag", "")
            recovery_order = find_order(snapshot, trade.get("recovery_order_id", ""), recovery_tag)
            if recovery_order:
                trade["recovery_order_id"] = recovery_order.get("id")
                if order_status(recovery_order) not in {1, 2, 5, 7}:
                    return
                # Give position book a moment to reflect completed recovery.
                if time.time() - safe_float(trade.get("recovery_submitted_at"), 0) < 3:
                    return
                self.align_hedge_and_short(trade, snapshot, allow_repair=False)
            elif time.time() - safe_float(trade.get("recovery_submitted_at"), time.time()) > 60:
                self.manual_review(trade, "Hedge recovery order not found in broker order book; check account before retrying")
            return

        if status in {"OCO_SUBMITTING", "OCO_UNKNOWN", "OCO_PENDING"}:
            gtt = find_gtt(snapshot, trade.get("gtt_id", ""), trade.get("short_symbol", ""))
            if gtt:
                trade["gtt_id"] = gtt.get("id")
                st = gtt_status(gtt)
                if st == 6:
                    trade["status"] = "OCO_ACTIVE"
                    emit("DATA", f"OCO confirmed active for {trade['short_symbol']} id={trade['gtt_id']}")
                    with STATE_LOCK:
                        save_state()
                elif st == 2:
                    trade["gtt_triggered"] = True
                    trade["gtt_triggered_at"] = trade.get("gtt_triggered_at") or time.time()
                    trade["status"] = "OCO_TRIGGERED"
                    with STATE_LOCK:
                        save_state()
                    self.reconcile_active_spread(trade, snapshot)
                elif st in {1, 5, 7}:
                    trade["status"] = "OCO_ACTIVE"
                    with STATE_LOCK:
                        save_state()
                    self.reconcile_active_spread(trade, snapshot)
            else:
                age = time.time() - safe_float(trade.get("oco_submitted_at"), time.time())
                if age > 30:
                    self.manual_review(trade, "OCO submission not found in GTT order book; no duplicate OCO will be sent automatically")
            with STATE_LOCK:
                save_state()

    def align_hedge_and_short(self, trade: Dict[str, Any], snapshot: Dict[str, Any], allow_repair: bool = True) -> None:
        short_net = position_qty(snapshot, trade["short_symbol"])
        hedge_net = position_qty(snapshot, trade["hedge_symbol"])
        if short_net > 0 or hedge_net < 0:
            self.manual_review(trade, f"Unexpected entry position directions: shortNet={short_net}, hedgeNet={hedge_net}")
            return
        try:
            short_qty = require_lot_quantity(abs(short_net), "entry short filled quantity")
            hedge_qty = require_lot_quantity(hedge_net, "entry hedge filled quantity")
        except Exception as exc:
            self.manual_review(trade, f"Entry position lot-size validation failed; no recovery order will be sent: {exc}")
            return

        if short_qty == 0 and hedge_qty == 0:
            trade["status"] = "ENTRY_FAILED"
            emit("WARN", f"Both entry legs have zero position for {trade['key']}; no OCO placed")
            with STATE_LOCK:
                save_state()
            return

        if short_qty == hedge_qty and short_qty > 0:
            trade["short_filled_qty"] = short_qty
            trade["hedge_filled_qty"] = hedge_qty
            trade["qty_requested"] = ORDER_QTY
            quotes = get_quotes([trade["short_symbol"]])
            current_ltp = safe_float(quotes.get(trade["short_symbol"], {}).get("lp"))
            if current_ltp <= 0:
                self.emergency_cover_short(trade, short_qty, "Could not obtain current short-option LTP before placing protective OCO")
                return
            try:
                submit_gtt_oco(trade, short_qty, current_ltp)
            except OcoLevelCrossed as exc:
                self.emergency_cover_short(trade, short_qty, str(exc))
            except OcoSubmissionUnknown as exc:
                emit("ERROR", f"GTT outcome ambiguous for {trade['key']}; waiting for broker reconciliation: {exc}")
            except Exception as exc:
                self.emergency_cover_short(trade, short_qty, f"OCO placement failed: {exc}")
            return

        if not allow_repair:
            self.manual_review(trade, f"Entry hedge/short quantities still differ after recovery: hedge={hedge_qty}, short={short_qty}")
            return

        difference = abs(short_qty - hedge_qty)
        if difference == 0:
            return
        difference = require_lot_quantity(difference, "entry hedge recovery quantity", allow_zero=False)
        if trade.get("recovery_order_tag"):
            self.manual_review(trade, f"Entry recovery order already submitted but quantities still differ: hedge={hedge_qty}, short={short_qty}")
            return

        side = 1 if short_qty > hedge_qty else -1
        action = "BUY additional hedge" if side == 1 else "SELL excess hedge"
        tag = order_tag(trade["date"], trade["leg"], "R")
        trade["status"] = "RECOVERY_PENDING"
        trade["recovery_order_tag"] = tag
        trade["recovery_submitted_at"] = time.time()
        trade["recovery_target_hedge_qty"] = short_qty
        with STATE_LOCK:
            save_state()
        emit("WARN", f"Unequal entry fills: hedge={hedge_qty}, short={short_qty}; {action} {difference} ({difference // LOT_SIZE} lots)")
        try:
            response = place_market_order(trade["hedge_symbol"], side, difference, tag)
            trade["recovery_order_id"] = response.get("id")
            if response.get("s") != "ok" or safe_int(response.get("code")) not in {1101, 201}:
                self.manual_review(trade, f"Entry hedge quantity repair rejected: {response}")
                return
            with STATE_LOCK:
                save_state()
        except Exception as exc:
            self.manual_review(trade, f"Entry hedge recovery outcome unknown: {exc}")

    def emergency_cover_short(self, trade: Dict[str, Any], qty: int, reason: str) -> None:
        """Market-buy the short if protective OCO cannot be installed."""
        try:
            qty = require_lot_quantity(qty, "emergency short-cover quantity", allow_zero=False)
        except Exception as exc:
            self.manual_review(trade, f"Cannot emergency-cover with an invalid lot quantity: {exc}. {reason}")
            return
        if qty <= 0:
            self.manual_review(trade, f"Cannot emergency-cover; invalid short qty={qty}. {reason}")
            return
        tag = order_tag(trade["date"], trade["leg"], "C")
        trade["status"] = "SHORT_EXIT_SUBMITTING"
        trade["short_exit_order_tag"] = tag
        trade["short_exit_submitted_at"] = time.time()
        trade["last_error"] = reason
        with STATE_LOCK:
            save_state()
        emit("ERROR", f"Protective OCO unavailable; emergency BUY to cover {trade['short_symbol']} qty={qty}. Reason: {reason}")
        try:
            resp = place_market_order(trade["short_symbol"], 1, qty, tag)
            trade["short_exit_order_id"] = resp.get("id")
            trade["status"] = "SHORT_EXIT_PENDING" if resp.get("s") == "ok" and safe_int(resp.get("code")) in {1101, 201} else "MANUAL_REVIEW"
            with STATE_LOCK:
                save_state()
            emit("ORDER", f"Emergency short cover response: {resp}")
            if trade["status"] == "MANUAL_REVIEW":
                emit("ERROR", "Emergency short cover was rejected. Immediate manual broker action required.")
        except Exception as exc:
            self.manual_review(trade, f"Emergency short-cover outcome unknown; check broker immediately: {exc}")

    def manual_review(self, trade: Dict[str, Any], reason: str) -> None:
        trade["status"] = "MANUAL_REVIEW"
        trade["last_error"] = reason
        trade["manual_review_at"] = stamp()
        with STATE_LOCK:
            save_state()
        emit("ERROR", f"MANUAL ACTION REQUIRED for {trade.get('key')}: {reason}")

    def render_dashboard(self) -> None:
        current = self.get_session(today_key())
        snapshot = BROKER_SNAPSHOT
        now = now_ist()
        lines = []
        lines.append("=" * 100)
        lines.append(" NIFTY PREMIUM-CRASH BOT  |  FYERS API v3  |  AWS / IST")
        lines.append("=" * 100)
        lines.append(f" Time: {now.strftime('%Y-%m-%d %H:%M:%S IST')}   Trading: {'LIVE ENABLED' if LIVE_TRADING else 'DRY-RUN (NO ORDERS)'}   Product: {PRODUCT_TYPE}   Qty/leg: {ORDER_QTY}")
        ws_age = time.time() - safe_float(WS_STATUS.get("last_event"), 0) if WS_STATUS.get("last_event") else None
        ws_text = "CONNECTED" if WS_STATUS.get("connected") else "DISCONNECTED"
        if ws_age is not None:
            ws_text += f" | last message {ws_age:.1f}s ago"
        lines.append(f" FYERS WS: {ws_text} ({WS_STATUS.get('message', '')})")
        if snapshot.get("ok"):
            age = time.time() - safe_float(snapshot.get("updated_at"))
            lines.append(f" Broker sync: OK ({age:.1f}s ago) | Open MARGIN symbols: {sum(1 for v in snapshot.get('positions', {}).values() if safe_int(v.get('netQty')) != 0)} | Orders: {len(snapshot.get('orders', []))} | GTT: {len(snapshot.get('gtt', []))}")
        else:
            lines.append(f" Broker sync: ERROR — {snapshot.get('error')}")

        lines.append("-" * 100)
        display = current or self.preview_snapshot
        if display:
            is_active_session = current is not None
            with PRICE_LOCK:
                index_ltp = CURRENT_PRICES.get(display["index_symbol"], safe_float(display.get("index_ltp")))
            index_open = safe_float(display.get("index_open"))
            index_change = index_ltp - index_open if index_ltp > 0 and index_open > 0 else 0.0
            index_change_pct = (index_change / index_open * 100.0) if index_open > 0 and index_ltp > 0 else 0.0
            index_ltp_text = f"{index_ltp:.2f}" if index_ltp > 0 else "--"
            index_open_text = f"{index_open:.2f}" if index_open > 0 else "--"
            index_change_text = (
                f"{index_change:+.2f} ({index_change_pct:+.2f}%)"
                if index_ltp > 0 and index_open > 0 else "--"
            )
            if is_active_session:
                lines.append(f" ACTIVE TRADING SESSION {current['date']} | EXPIRY {current.get('expiry_date')}")
            else:
                lines.append(
                    f" LAST AVAILABLE QUOTE SNAPSHOT | fetched {display.get('fetched_at', 'unknown')} "
                    "| DISPLAY ONLY — NOT A TODAY'S TRADING BASELINE"
                )
            lines.append(
                f" INDEX {display['index_symbol']:<28} OPEN {index_open_text:>10} "
                f"LTP {index_ltp_text:>10}  CHANGE {index_change_text}"
            )
            lines.append(f" SELECTED ATM STRIKE (based on index open): {display['atm']}")
            lines.append(f" OPTION EXPIRY: {display.get('expiry_date') or '--'}")
            lines.append(f" {'LEG':<5} {'SYMBOL':<34} {'OPEN':>10} {'LTP':>10} {'50% LEVEL':>11} {'STATUS':<18}")
            for leg in ("CE", "PE"):
                sym = display[f"{leg.lower()}_symbol"]
                opx = safe_float(display.get(f"{leg.lower()}_open"))
                trigger = opx * (1.0 - DROP_PCT) if opx > 0 else 0.0
                with PRICE_LOCK:
                    ltp = CURRENT_PRICES.get(sym, safe_float(display.get(f"{leg.lower()}_ltp")))
                open_text = f"{opx:.2f}" if opx > 0 else "--"
                ltp_text = f"{ltp:.2f}" if ltp > 0 else "--"
                trigger_text = f"{trigger:.2f}" if trigger > 0 else "--"
                signal_state = current.get("signals", {}).get(leg, "WAITING") if is_active_session else "DISPLAY ONLY"
                lines.append(
                    f" {leg:<5} {sym:<34} {open_text:>10} {ltp_text:>10} "
                    f"{trigger_text:>11} {signal_state:<18}"
                )
            if not is_active_session:
                lines.append(" NOTE: Closed-market LTPs are the last available Quotes API prices; no signals/orders are evaluated from this preview.")
        else:
            lines.append(" MARKET DATA SNAPSHOT: waiting for FYERS Quotes / Option Chain response.")
            lines.append(" NIFTY OPEN/LTP, opening ATM, and ATM CE/PE OPEN/LTP will appear here when quote data is available.")
            if self.preview_error:
                lines.append(f" Last snapshot error: {self.preview_error[:180]}")

        lines.append("-" * 100)
        lines.append(" TRADES")
        with STATE_LOCK:
            trades = list(STATE.get("trades", {}).values())
        if not trades:
            lines.append(" No strategy trades recorded.")
        else:
            lines.append(f" {'DATE/LEG':<16} {'STATUS':<20} {'SHORT':<28} {'HEDGE':<28} {'SL':>8} {'TARGET':>8}")
            for trade in sorted(trades, key=lambda t: (t.get("date", ""), t.get("leg", "")))[-8:]:
                lines.append(
                    f" {trade.get('date','')} {trade.get('leg',''):<3} "
                    f"{trade.get('status',''):<20} {trade.get('short_symbol',''):<28} "
                    f"{trade.get('hedge_symbol',''):<28} {safe_float(trade.get('sl_trigger')):>8.2f} "
                    f"{safe_float(trade.get('target_trigger')):>8.2f}"
                )
        lines.append("-" * 100)
        lines.append(" BROKER POSITIONS (MARGIN)")
        positions = [
            (sym, row) for sym, row in snapshot.get("positions", {}).items()
            if safe_int(row.get("netQty")) != 0
        ] if snapshot.get("ok") else []
        if positions:
            for sym, row in positions[:6]:
                details = row.get("rows", [{}])[0]
                lines.append(f" {sym:<42} netQty={safe_int(row.get('netQty')):>7} netAvg={safe_float(details.get('netAvg')):>9.2f} P&L={safe_float(details.get('pl')):>10.2f}")
        else:
            lines.append(" (No open MARGIN positions returned)")

        lines.append(" PENDING REGULAR ORDERS")
        pending_orders = [o for o in snapshot.get("orders", []) if order_status(o) in {4, 6}] if snapshot.get("ok") else []
        if pending_orders:
            for o in pending_orders[:5]:
                lines.append(f" {str(o.get('id','')):<16} {str(o.get('symbol','')):<34} side={str(o.get('side','')):<3} qty={order_qty(o):<6} filled={order_filled_qty(o):<6} status={order_status(o)}")
        else:
            lines.append(" (No pending regular orders)")

        lines.append(" ACTIVE GTT / OCO ORDERS")
        active_gtt = [g for g in snapshot.get("gtt", []) if gtt_status(g) in {4, 6}] if snapshot.get("ok") else []
        if active_gtt:
            for g in active_gtt[:5]:
                lines.append(f" {str(g.get('id','')):<16} {str(g.get('symbol','')):<34} OCO={safe_int(g.get('gtt_oco_ind')) == 2} status={gtt_status(g)}")
        else:
            lines.append(" (No pending GTT orders returned)")

        lines.append("-" * 100)
        lines.append(" RECENT EVENTS")
        with EVENT_LOCK:
            events = list(EVENTS)[:8]
        lines.extend(events if events else ["(No events yet)"])
        lines.append("=" * 100)
        lines.append(f"State file: {STATE_FILE} | Log file: {LOG_FILE} | Press Ctrl+C to stop (open broker positions/GTT remain).")
        # A continuously refreshing dashboard is intended for an interactive
        # terminal (PowerShell/Windows Terminal, SSH, tmux, etc.). Avoid dumping
        # the full screen once per second when stdout is redirected to a file.
        if not sys.stdout.isatty():
            return
        clear_terminal()
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    def run(self) -> None:
        emit("DATA", "Bot booting")
        emit("DATA", f"Symbol={NIFTY_SYMBOL} strike_step={STRIKE_STEP} qty={ORDER_QTY} drop={DROP_PCT:.0%} hedge_ratio={HEDGE_RATIO:.0%}")
        emit("DATA", f"State file: {STATE_FILE}")
        if not LIVE_TRADING:
            emit("WARN", "LIVE_TRADING is false: monitoring and hedge-selection only; no orders will be sent")
        emit("WARN", f"Configured LOT_SIZE={LOT_SIZE}; all entry, GTT and hedge-adjustment quantities are required to be exact lot multiples.")
        self.sync_broker(force=True)

        while self.running:
            try:
                session = self.create_session_if_needed()
                if session:
                    self.start_websocket(session)
                    self.refresh_index_ltp()
                else:
                    # Always populate the requested NIFTY / ATM CE / ATM PE
                    # dashboard fields, including on weekends and after hours.
                    # This preview is display-only and cannot trigger trades.
                    self.refresh_dashboard_preview()
                    self.refresh_preview_quotes()
                self.sync_broker()
                self.reconcile_all_trades()
                if time.time() - self.last_dashboard >= DASHBOARD_INTERVAL_SEC:
                    self.render_dashboard()
                    self.last_dashboard = time.time()
                time.sleep(0.5)
            except KeyboardInterrupt:
                self.running = False
                emit("WARN", "Ctrl+C received. Exiting bot; broker-side GTT orders and positions are not cancelled automatically.")
            except Exception as exc:
                self.last_error = str(exc)
                emit("ERROR", f"Main loop exception: {exc}\n{traceback.format_exc()}")
                time.sleep(2)


def main() -> None:
    global STATE
    if LOT_SIZE <= 0 or LOTS <= 0:
        raise RuntimeError(f"LOT_SIZE and LOTS must be positive; LOT_SIZE={LOT_SIZE}, LOTS={LOTS}")
    require_lot_quantity(ORDER_QTY, "configured entry quantity", allow_zero=False)
    lock_handle = acquire_single_instance_lock()
    try:
        STATE = load_state()
        bot = PremiumCrashBot()
        bot.run()
    finally:
        try:
            release_single_instance_lock(lock_handle)
        except Exception as exc:
            logging.exception("Could not release single-instance lock: %s", exc)


if __name__ == "__main__":
    main()
