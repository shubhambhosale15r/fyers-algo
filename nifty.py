#!/usr/bin/env python3
"""
NIFTY ATM Premium-Crash Bot - FYERS API v3 (broker-driven, no state file)

Design
  * The broker is the only source of truth. There is NO state.json. Every
    decision is derived from FYERS positions, order book (matched by orderTag)
    and GTT order book. Restarting the bot loses nothing.
  * REST read calls (quotes, optionchain, funds, positions, orderbook,
    gtt_orderbook) are throttled: each endpoint is called at most once per
    READ_GAP_SEC (minimum 15s, enforced). Order placement / cancellation is
    never throttled. Live prices come from the Data WebSocket (index + ATM CE/PE).
  * Entry: when an ATM option's LTP integer part equals the integer part of its
    50% level (int(ltp) == int(open * (1 - DROP_PCT))), buy a same-side OTM hedge
    (~25% of LTP) and sell the ATM option via POST /api/v3/multi-order/sync.
  * Failed entry (short leg rejected/cancelled/expired with no fill): retry after
    RETRY_COOLDOWN_SEC, at most MAX_ENTRY_ATTEMPTS per symbol per day. The count
    comes from today's order book (orders tagged by this bot), so it survives
    restarts.
  * After the broker shows short == hedge, a BUY-side GTT OCO (SL above LTP,
    target below LTP) is placed on the short.
  * When the short is flat, any pending OCO is cancelled and the hedge is closed.

Safety defaults
  * LIVE_TRADING is false unless the environment variable is 1/true/yes/y.
  * Credentials come from FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN only.
  * The account is assumed to be dedicated to this strategy: any NIFTY option
    short + same-expiry same-side long in the account is treated as a trade.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws


# =============================================================================
# CONFIGURATION
# =============================================================================
def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


CLIENT_ID=""
ACCESS_TOKEN=""

#LIVE_TRADING = env_bool("LIVE_TRADING", False)
LIVE_TRADING = True

NIFTY_SYMBOL = os.getenv("NIFTY_SYMBOL", "NSE:NIFTY50-INDEX").strip()
STRIKE_STEP = int(os.getenv("STRIKE_STEP", "50"))
STRIKE_COUNT = 50
PRODUCT_TYPE = "MARGIN"
LOT_SIZE = int(os.getenv("LOT_SIZE", "65"))  # VERIFY current exchange lot size
LOTS = int(os.getenv("LOTS", "1"))
ORDER_QTY = LOT_SIZE * LOTS

DROP_PCT = float(os.getenv("DROP_PCT", "0.50"))
HEDGE_RATIO = float(os.getenv("HEDGE_RATIO", "0.25"))
TARGET_PROFIT_PCT = float(os.getenv("TARGET_PROFIT_PCT", "0.99"))
TICK_SIZE = float(os.getenv("OPTION_TICK_SIZE", "0.05"))

# Read-call throttle: never below 15 seconds per endpoint.
READ_GAP_SEC = max(15.0, float(os.getenv("READ_GAP_SEC", "15")))

MAX_ENTRY_ATTEMPTS = int(os.getenv("MAX_ENTRY_ATTEMPTS", "3"))
RETRY_COOLDOWN_SEC = int(os.getenv("RETRY_COOLDOWN_SEC", "60"))
FUNDS_RETRY_COOLDOWN_SEC = int(os.getenv("FUNDS_RETRY_COOLDOWN_SEC", "60"))
MIN_REQUIRED_FUNDS = float(os.getenv("MIN_REQUIRED_FUNDS", "0"))

ENTRY_SETTLE_SEC = 3          # snapshot must start this long after submit to be trusted
INFLIGHT_TIMEOUT_SEC = 120    # give up waiting for entry orders to show up
ACTION_COOLDOWN_SEC = 45      # do not repeat the same corrective order sooner
OCO_RECHECK_SEC = 60          # wait before placing another OCO for the same short
MAX_OCO_TRIES = 3
MAX_HEDGE_REPAIRS = 2         # tries to restore a missing hedge before covering the short
STUCK_EXIT_WARN_SEC = 90      # warn if a triggered BUY exit order stays unfilled this long
SNAPSHOT_MAX_AGE_SEC = READ_GAP_SEC * 2 + 15
AUTO_CLOSE_ORPHAN_HEDGES = env_bool("AUTO_CLOSE_ORPHAN_HEDGES", True)

DASHBOARD_INTERVAL_SEC = float(os.getenv("DASHBOARD_INTERVAL_SEC", "1"))
WS_RECONNECT_RETRIES = 50

DATA_DIR = Path(os.getenv("BOT_DATA_DIR", str(Path.home() / "nifty_premium_bot")))
LOCK_FILE = Path(os.getenv("BOT_LOCK_FILE", str(DATA_DIR / "bot.lock")))
LOG_FILE = Path(os.getenv("BOT_LOG_FILE", str(DATA_DIR / "bot.log")))

IST = timezone(timedelta(hours=5, minutes=30))

# =============================================================================
# LOGGING / SHARED
# =============================================================================
DATA_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

EVENTS = deque(maxlen=12)
EVENT_LOCK = threading.Lock()
API_LOCK = threading.RLock()
PRICE_LOCK = threading.RLock()
CURRENT_PRICES: Dict[str, float] = {}
LAST_TICK_AT: Dict[str, float] = {}
WS_STATUS: Dict[str, Any] = {"connected": False, "message": "not started", "last_event": 0.0}


def now_ist() -> datetime:
    return datetime.now(IST)


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
    if value in (None, ""):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} quantity from broker: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"Non-integer {label} quantity from broker: {value!r}")
    return int(numeric)


#def market_session_open() -> bool:
#    """Live-trading window. Starts at 09:16, not 09:15, so the Quotes API has had a
#    minute to publish today's option open_price instead of a stale/zero value."""
#    now = now_ist()
#    return now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(15, 30)

def market_session_open() -> bool:
    """Live-trading window: 09:15 to 15:30 IST."""
    now = now_ist()
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
    return int(math.floor(index_open / STRIKE_STEP + 0.5) * STRIKE_STEP)


# -----------------------------------------------------------------------------
# Option symbol parsing (NIFTY options only; BANKNIFTY/FINNIFTY/NIFTYNXT50 excluded)
# Strike is assumed to be the last 5 digits before CE/PE.
# -----------------------------------------------------------------------------
OPT_RE = re.compile(r"^(NSE:NIFTY\d[0-9A-Z]*?)(\d{5})(CE|PE)$")


def parse_option(symbol: Any) -> Optional[Tuple[str, int, str]]:
    m = OPT_RE.match(str(symbol or ""))
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def group_key(symbol: Any) -> Optional[Tuple[str, str]]:
    p = parse_option(symbol)
    return (p[0], p[2]) if p else None


# -----------------------------------------------------------------------------
# Order tags: NPC + YYYYMMDD + leg + role + suffix  (alphanumeric, <= 30 chars)
#   S = entry short, H = entry hedge, B = hedge buy repair, X = hedge sell/close,
#   C = emergency short cover
# -----------------------------------------------------------------------------
def tag_prefix(date: str, leg: str, role: str) -> str:
    return f"NPC{date.replace('-', '')}{leg}{role}"


def new_tag(leg: str, role: str) -> str:
    return f"{tag_prefix(today_key(), leg, role)}{now_ist().strftime('%H%M%S')}"


def order_tag_of(order: Dict[str, Any]) -> str:
    tag = str(order.get("orderTag", order.get("order_tag", order.get("tag", ""))) or "")
    i = tag.find("NPC")
    return tag[i:] if i >= 0 else tag


def orders_with_prefix(snap: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
    return [o for o in snap.get("orders", []) if order_tag_of(o).startswith(prefix)]


def order_epoch(order: Dict[str, Any]) -> float:
    raw = str(order.get("orderDateTime", "") or "").strip()
    if not raw:
        return 0.0
    if raw.isdigit():
        return float(raw)
    try:
        return datetime.strptime(raw, "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).timestamp()
    except ValueError:
        return 0.0


# =============================================================================
# READ THROTTLE + FYERS CLIENT
# =============================================================================
class ReadGate:
    """Allows each read endpoint at most one call per `gap` seconds."""

    def __init__(self, gap: float):
        self.gap = gap
        self._next: Dict[str, float] = {}
        self._lock = threading.Lock()

    def acquire(self, endpoint: str, block: bool = True) -> bool:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next.get(endpoint, 0.0) - now)
            if wait > 0 and not block:
                return False
            self._next[endpoint] = now + wait + self.gap + 0.05
        if wait > 0:
            time.sleep(wait)
        return True


GATE = ReadGate(READ_GAP_SEC)

fyers = None


def init_client() -> None:
    global fyers
    if not CLIENT_ID or not ACCESS_TOKEN:
        print("ERROR: set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN environment variables.", file=sys.stderr)
        sys.exit(2)
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")


def _call(method_name: str, *args, **kwargs) -> Dict[str, Any]:
    method = getattr(fyers, method_name)
    with API_LOCK:                      # lock held only for the HTTP call itself
        response = method(*args, **kwargs)
    if not isinstance(response, dict):
        raise RuntimeError(f"FYERS {method_name} returned non-dict response: {response!r}")
    return response


def api_read(method_name: str, *args, **kwargs) -> Dict[str, Any]:
    """Throttled read call (waits for its per-endpoint slot BEFORE taking the API lock)."""
    GATE.acquire(method_name)
    return _call(method_name, *args, **kwargs)


def api_write(method_name: str, *args, **kwargs) -> Dict[str, Any]:
    """Order placement / cancellation: never throttled."""
    return _call(method_name, *args, **kwargs)


def response_ok(resp: Dict[str, Any]) -> bool:
    return resp.get("s") == "ok"


def get_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    symbols = list(dict.fromkeys(s for s in symbols if s))
    for start in range(0, len(symbols), 50):
        resp = api_read("quotes", {"symbols": ",".join(symbols[start:start + 50])})
        if not response_ok(resp):
            raise RuntimeError(f"Quotes API failed: {resp}")
        for row in resp.get("d", []):
            if row.get("s") != "ok":
                emit("WARN", f"Quote failed for {row.get('n')}: {row}")
                continue
            if row.get("n"):
                result[row["n"]] = row.get("v", {})
    return result


def quote_ltp(values: Dict[str, Any]) -> float:
    return safe_float(values.get("lp", values.get("ltp", 0.0)))


def fetch_available_funds() -> Optional[float]:
    resp = api_read("funds")
    if not response_ok(resp):
        raise RuntimeError(f"Funds API failed: {resp}")
    for row in resp.get("fund_limit", []) or []:
        if str(row.get("title", "")).strip().lower().startswith("available balance"):
            return safe_float(row.get("equityAmount"))
    return None


def fetch_option_chain(timestamp: str = "", strikecount: int = STRIKE_COUNT) -> Dict[str, Any]:
    payload = {"symbol": NIFTY_SYMBOL, "strikecount": min(max(1, int(strikecount)), 50),
               "timestamp": str(timestamp), "greeks": "1"}
    resp = api_read("optionchain", data=payload)
    if not response_ok(resp):
        raise RuntimeError(f"Option Chain API failed: {resp}")
    return resp.get("data", {})


def choose_nearest_expiry(expiry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    today = now_ist().date()
    candidates = []
    for item in expiry_data:
        try:
            d = datetime.strptime(item["date"], "%d-%m-%Y").date()
            str(item["expiry"])
        except (KeyError, TypeError, ValueError):
            continue
        if d >= today:
            candidates.append((d, item))
    if not candidates:
        raise RuntimeError(f"No current/future expiry available: {expiry_data}")
    candidates.sort(key=lambda p: p[0])
    return candidates[0][1]


def build_session(live: bool) -> Dict[str, Any]:
    """Rebuild the in-memory baseline from the broker (index open, ATM, option opens).

    Everything here is re-derivable after a restart: index open and option
    open_price come from the Quotes API, so no persistence is needed.
    """
    idx = get_quotes([NIFTY_SYMBOL]).get(NIFTY_SYMBOL)
    if not idx:
        raise RuntimeError(f"No quote row returned for {NIFTY_SYMBOL}")
    index_open = safe_float(idx.get("open_price"))
    index_ltp = quote_ltp(idx)
    if index_open <= 0:
        raise RuntimeError(f"Invalid NIFTY open_price: {idx}")
    atm = nearest_atm(index_open)

    initial = fetch_option_chain("", 1)
    expiry = choose_nearest_expiry(initial.get("expiryData", []))
    chain = fetch_option_chain(str(expiry["expiry"]), STRIKE_COUNT).get("optionsChain", [])

    rows: Dict[str, List[Tuple[int, str]]] = {"CE": [], "PE": []}
    ce = pe = None
    for r in chain:
        sym, typ = r.get("symbol"), r.get("option_type")
        strike = safe_int(r.get("strike_price"))
        if not sym or typ not in rows:
            continue
        rows[typ].append((strike, sym))
        if strike == atm:
            if typ == "CE":
                ce = sym
            else:
                pe = sym
    if not ce or not pe:
        raise RuntimeError(f"ATM {atm} CE/PE not in returned option chain (underlying may have moved out of the window)")

    q = get_quotes([ce, pe])
    ce_open, pe_open = safe_float(q.get(ce, {}).get("open_price")), safe_float(q.get(pe, {}).get("open_price"))
    if ce_open <= 0 or pe_open <= 0:
        raise RuntimeError(f"Option open price not available yet (CE={ce_open}, PE={pe_open}); will retry")

    with PRICE_LOCK:
        for sym, v in ((NIFTY_SYMBOL, index_ltp), (ce, quote_ltp(q.get(ce, {}))), (pe, quote_ltp(q.get(pe, {})))):
            if v > 0:
                CURRENT_PRICES[sym] = v
    return {
        "date": today_key(), "live": live, "built_at": stamp(),
        "index_symbol": NIFTY_SYMBOL, "index_open": index_open, "atm": atm,
        "expiry_date": expiry.get("date"), "expiry_timestamp": str(expiry["expiry"]),
        "ce_symbol": ce, "pe_symbol": pe, "ce_open": ce_open, "pe_open": pe_open,
        "chain": rows,
    }


# =============================================================================
# BROKER SNAPSHOT (the only source of truth)
# =============================================================================
def list_from_response(resp: Dict[str, Any], *keys: str) -> List[Dict[str, Any]]:
    for k in keys:
        v = resp.get(k)
        if isinstance(v, list):
            return v
    return []


def build_broker_snapshot() -> Dict[str, Any]:
    started = time.time()
    positions_resp = api_read("positions")
    orders_resp = api_read("orderbook")
    gtt_resp = api_read("gtt_orderbook")
    for name, resp in (("positions", positions_resp), ("orderbook", orders_resp), ("gtt_orderbook", gtt_resp)):
        if not response_ok(resp):
            raise RuntimeError(f"FYERS {name} failed: {resp}")
    pos: Dict[str, Dict[str, Any]] = {}
    for p in positions_resp.get("netPositions", []):
        sym = p.get("symbol")
        if not sym:
            continue
        product = p.get("productType", "")
        if product and product != PRODUCT_TYPE:
            continue
        entry = pos.setdefault(sym, {"netQty": 0, "rows": []})
        entry["netQty"] += strict_qty_int(p.get("netQty"), f"position {sym}")
        entry["rows"].append(p)
    return {
        "ok": True, "positions": pos,
        "orders": list_from_response(orders_resp, "orderBook", "data"),
        "gtt": list_from_response(gtt_resp, "orderBook", "data"),
        "started_at": started, "updated_at": time.time(), "error": "",
    }


def position_qty(snap: Dict[str, Any], symbol: str) -> int:
    return strict_qty_int(snap.get("positions", {}).get(symbol, {}).get("netQty", 0), f"position {symbol}")


def position_avg(snap: Dict[str, Any], symbol: str) -> float:
    for row in snap.get("positions", {}).get(symbol, {}).get("rows", []):
        if safe_int(row.get("netQty")) != 0:
            for key in ("netAvg", "sellAvg", "avgPrice"):
                v = safe_float(row.get(key))
                if v > 0:
                    return v
    return 0.0


def require_lot_quantity(qty: Any, label: str, allow_zero: bool = True) -> int:
    value = strict_qty_int(qty, label)
    if value < 0:
        raise ValueError(f"{label} must be non-negative; got {qty!r}")
    if not allow_zero and value == 0:
        raise ValueError(f"{label} must be greater than zero")
    if LOT_SIZE <= 0 or value % LOT_SIZE != 0:
        raise ValueError(f"{label}={value} is not a multiple of LOT_SIZE={LOT_SIZE}; automated adjustment blocked")
    return value


def order_side(o: Dict[str, Any]) -> int:
    return safe_int(o.get("side", o.get("tran_side", 0)))


def order_status(o: Dict[str, Any]) -> int:
    return safe_int(o.get("status", o.get("order_status", 0)))


def gtt_status(o: Dict[str, Any]) -> int:
    return safe_int(o.get("ord_status", o.get("status", 0)))


# FYERS order status: 1 cancelled, 2 traded, 4 transit, 5 rejected, 6 pending, 7 expired
PENDING = {4, 6}


def pending_orders(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [o for o in snap.get("orders", []) if order_status(o) in PENDING]


def pending_ocos(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [g for g in snap.get("gtt", []) if gtt_status(g) in PENDING and safe_int(g.get("gtt_oco_ind")) == 2]


def find_order(snap: Dict[str, Any], order_id: str = "", tag: str = "") -> Optional[Dict[str, Any]]:
    for o in snap.get("orders", []):
        if order_id and str(o.get("id", "")) == str(order_id):
            return o
        if tag and order_tag_of(o) == tag:
            return o
    return None


def scan_groups(snap: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Group NIFTY option symbols by (expiry prefix, CE/PE) from broker data only."""
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def grp(key):
        return groups.setdefault(key, {"symbols": set(), "short": None, "short_qty": 0,
                                       "long": None, "long_qty": 0, "conflict": ""})

    for sym, row in snap.get("positions", {}).items():
        key = group_key(sym)
        if not key:
            continue
        g = grp(key)
        g["symbols"].add(sym)
        net = safe_int(row.get("netQty"))
        if net < 0:
            if g["short"] and g["short"] != sym:
                g["conflict"] = f"multiple short symbols {g['short']}, {sym}"
            else:
                g["short"], g["short_qty"] = sym, abs(net)
        elif net > 0:
            if g["long"] and g["long"] != sym:
                g["conflict"] = f"multiple long symbols {g['long']}, {sym}"
            else:
                g["long"], g["long_qty"] = sym, net
    for o in pending_orders(snap):
        key = group_key(o.get("symbol"))
        if key:
            grp(key)["symbols"].add(o["symbol"])
    for gtt in pending_ocos(snap):
        key = group_key(gtt.get("symbol"))
        if key:
            grp(key)["symbols"].add(gtt["symbol"])
    return groups


def group_busy_reason(snap: Dict[str, Any], key: Tuple[str, str]) -> Optional[str]:
    for sym, row in snap.get("positions", {}).items():
        if group_key(sym) == key and safe_int(row.get("netQty")) != 0:
            return f"open position in {sym}"
    for o in pending_orders(snap):
        if group_key(o.get("symbol")) == key:
            return f"pending order on {o.get('symbol')}"
    for g in snap.get("gtt", []):
        if gtt_status(g) in PENDING and group_key(g.get("symbol")) == key:
            return f"pending GTT on {g.get('symbol')}"
    return None


# =============================================================================
# ORDER HELPERS
# =============================================================================
class OcoLevelCrossed(RuntimeError):
    pass


class OcoSubmissionUnknown(RuntimeError):
    pass


class InsufficientFunds(RuntimeError):
    pass


def find_hedge_contract(session: Dict[str, Any], leg: str, short_strike: int, short_prefix: str,
                        reference_price: float) -> Tuple[str, int, float]:
    """Same-side OTM contract (same expiry) whose LTP is closest to reference * HEDGE_RATIO."""
    target = reference_price * HEDGE_RATIO
    cands = []
    for strike, sym in session["chain"][leg]:
        p = parse_option(sym)
        if not p or p[0] != short_prefix:
            continue
        if (leg == "CE" and strike > short_strike) or (leg == "PE" and strike < short_strike):
            cands.append((abs(strike - short_strike), strike, sym))
    cands.sort()
    if not cands:
        raise RuntimeError(f"No OTM {leg} hedge candidates for expiry prefix {short_prefix}")
    quotes = get_quotes([c[2] for c in cands[:50]])
    best = None
    for _, strike, sym in cands[:50]:
        ltp = quote_ltp(quotes.get(sym, {}))
        if ltp <= 0:
            continue
        d = abs(ltp - target)
        if best is None or d < best[0]:
            best = (d, sym, strike, ltp)
    if best is None:
        raise RuntimeError(f"No valid hedge LTPs for {leg}; target={target:.2f}")
    return best[1], best[2], best[3]


def entry_order_payload(symbol: str, side: int, qty: int, tag: str) -> Dict[str, Any]:
    return {"symbol": symbol, "qty": qty, "type": 2, "side": side, "productType": PRODUCT_TYPE,
            "limitPrice": 0, "stopPrice": 0, "validity": "DAY", "disclosedQty": 0,
            "offlineOrder": False, "orderTag": tag}


def parse_multi_order_response(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    rows = resp.get("data", [])
    if isinstance(rows, list):
        for row in rows:
            body = row.get("body", row) if isinstance(row, dict) else {}
            out.append({"statusCode": row.get("statusCode") if isinstance(row, dict) else None,
                        "s": body.get("s"), "code": body.get("code"), "id": body.get("id") or "",
                        "message": body.get("message", "")})
    return out


def leg_rejected(row: Dict[str, Any]) -> bool:
    # Not assuming specific success codes: error flag or HTTP-style failure only.
    sc = row.get("statusCode")
    return row.get("s") == "error" or (isinstance(sc, int) and sc >= 400)


def place_market_order(symbol: str, side: int, qty: int, tag: str) -> Dict[str, Any]:
    qty = require_lot_quantity(qty, f"order quantity for {symbol}", allow_zero=False)
    if side not in {-1, 1}:
        raise ValueError(f"Invalid side {side}")
    return api_write("place_order", data=entry_order_payload(symbol, side, qty, tag))


def submit_gtt_oco(short_symbol: str, qty: int, short_avg: float, current_ltp: float) -> Tuple[float, float]:
    qty = require_lot_quantity(qty, "GTT OCO quantity", allow_zero=False)
    if not 0.0 < TARGET_PROFIT_PCT < 1.0:
        raise ValueError("TARGET_PROFIT_PCT must be between 0 and 1")
    # Levels derived from the broker's average sell price (no saved signal LTP):
    # open is implied by the drop rule, SL sits halfway between fill and implied open.
#    implied_open = short_avg / (1.0 - DROP_PCT)
#    sl_trigger = round_to_tick(short_avg + (implied_open - short_avg) / 2.0, "up")
    # SL at the implied option open price
    implied_open = short_avg / (1.0 - DROP_PCT)
    sl_trigger = round_to_tick(implied_open, "up")
    target_trigger = round_to_tick(short_avg * (1.0 - TARGET_PROFIT_PCT), "down")
    if not (sl_trigger > current_ltp and target_trigger < current_ltp):
        raise OcoLevelCrossed(f"OCO levels do not bracket LTP: ltp={current_ltp:.2f} SL={sl_trigger:.2f} target={target_trigger:.2f}")
    payload = {
        "side": 1, "symbol": short_symbol, "productType": PRODUCT_TYPE,
        "orderInfo": {
            "leg1": {"price": round_to_tick(sl_trigger * 1.05, "up"), "triggerPrice": sl_trigger, "qty": qty},
            "leg2": {"price": round_to_tick(target_trigger + TICK_SIZE, "up"), "triggerPrice": target_trigger, "qty": qty},
        },
    }
    emit("ORDER", f"Placing BUY OCO {short_symbol} qty={qty}: SL={sl_trigger:.2f} target={target_trigger:.2f}")
    try:
        resp = api_write("place_gtt_order", data=payload)
    except Exception as exc:
        raise OcoSubmissionUnknown(str(exc)) from exc
    code = safe_int(resp.get("code"))
    if resp.get("s") == "ok" and code in {1101, 201}:   # documented: 1101 placed, 201 in transit
        emit("ORDER", f"GTT OCO accepted id={resp.get('id')} code={code}; next broker sync will confirm it")
        return sl_trigger, target_trigger
    raise RuntimeError(f"GTT OCO rejected: {resp}")


# =============================================================================
# BOT
# =============================================================================
class PremiumCrashBot:
    def __init__(self):
        self.running = True
        self.snapshot: Dict[str, Any] = {"ok": False, "positions": {}, "orders": [], "gtt": [],
                                         "started_at": 0.0, "updated_at": 0.0, "error": "not synced"}
        self.last_snapshot_start = 0.0
        self.session: Optional[Dict[str, Any]] = None
        self.session_thread: Optional[threading.Thread] = None
        self.last_setup_attempt = 0.0
        self.ws_started: set = set()
        self.last_dashboard = 0.0
        self.last_error = ""
        # Short-lived in-memory guards only (nothing here is needed after a restart):
        self.inflight: Dict[str, Dict[str, Any]] = {}      # leg -> submitted entry basket
        self.entry_busy: Dict[str, bool] = {}
        self.retry_after_mem: Dict[str, float] = {}
        self.presubmit_failures: Dict[str, int] = {}       # failures that leave no broker order
        self.resp_failed_ids: Dict[str, set] = {}          # rejected short ids seen in responses
        self.submissions: Dict[str, int] = {}
        self.actions: Dict[Any, float] = {}
        self.oco_tries: Dict[str, int] = {}
        self.repair_tries: Dict[str, int] = {}
        self.dry_run_done: set = set()
        self.blocker_log: Dict[str, Tuple[str, float]] = {}
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ session
    def ensure_session(self) -> None:
        s, live = self.session, market_session_open()
        need = s is None or s["date"] != today_key() or (live and not s["live"])
        if not need or (self.session_thread and self.session_thread.is_alive()):
            return
        if time.time() - self.last_setup_attempt < READ_GAP_SEC:
            return
        self.last_setup_attempt = time.time()
        self.session_thread = threading.Thread(target=self._session_worker, args=(live,), daemon=True, name="session")
        self.session_thread.start()

    def _session_worker(self, live: bool) -> None:
        try:
            s = build_session(live)
            self.session = s
            emit("DATA", f"Session {'LIVE' if live else 'preview (market closed)'}: open={s['index_open']:.2f} ATM={s['atm']} "
                         f"expiry={s['expiry_date']} CE open={s['ce_open']:.2f} PE open={s['pe_open']:.2f}")
        except Exception as exc:
            msg = f"Session setup not ready; retrying: {exc}"
            if msg != self.last_error:
                emit("WARN", msg)
            self.last_error = msg

    # --------------------------------------------------------------- websocket
    def start_websocket(self, session: Dict[str, Any]) -> None:
        if session["date"] in self.ws_started:
            return
        self.ws_started.add(session["date"])
        threading.Thread(target=self._ws_worker, args=(session,), daemon=True, name="fyers-ws").start()

    def _ws_worker(self, session: Dict[str, Any]) -> None:
        symbols = [session["ce_symbol"], session["pe_symbol"], NIFTY_SYMBOL]

        def on_message(message):
            try:
                # SDK tags option/stock ticks "sf" and INDEX ticks "if" (verified in data_ws.py).
                if not isinstance(message, dict) or message.get("type") not in {"sf", "if"}:
                    return
                symbol, ltp = message.get("symbol"), safe_float(message.get("ltp"))
                if not symbol or ltp <= 0:
                    return
                now = time.time()
                with PRICE_LOCK:
                    CURRENT_PRICES[symbol] = ltp
                    LAST_TICK_AT[symbol] = now
                WS_STATUS["last_event"] = now
                cur = self.session
                if cur and cur["date"] == session["date"]:
                    if symbol == cur["ce_symbol"]:
                        self.check_trigger("CE", ltp)
                    elif symbol == cur["pe_symbol"]:
                        self.check_trigger("PE", ltp)
            except Exception as exc:
                emit("WARN", f"WebSocket message error: {exc}")

        def on_connect():
            WS_STATUS.update(connected=True, message="connected; subscribing")
            try:
                socket.subscribe(symbols=symbols, data_type="SymbolUpdate")
                socket.keep_running()
            except Exception as exc:
                WS_STATUS.update(connected=False, message=f"subscribe error: {exc}")
                emit("ERROR", f"WebSocket subscribe failed: {exc}")

        def on_close(*args):
            WS_STATUS.update(connected=False, message="closed/reconnecting")

        def on_error(error):
            WS_STATUS.update(connected=False, message=f"error: {error}")
            emit("ERROR", f"WebSocket error: {error}")

        socket = data_ws.FyersDataSocket(
            access_token=f"{CLIENT_ID}:{ACCESS_TOKEN}", log_path="", litemode=True, write_to_file=False,
            reconnect=True, reconnect_retry=WS_RECONNECT_RETRIES,
            on_connect=on_connect, on_close=on_close, on_error=on_error, on_message=on_message)
        try:
            WS_STATUS.update(connected=False, message="connecting")
            socket.connect()
        except Exception as exc:
            WS_STATUS.update(connected=False, message=f"connection failed: {exc}")
            emit("ERROR", f"WebSocket connection failed: {exc}")

    # ------------------------------------------------------------- entry logic
    def failed_short_orders(self, snap: Dict[str, Any], date: str, leg: str, symbol: str) -> List[Dict[str, Any]]:
        out = []
        for o in orders_with_prefix(snap, tag_prefix(date, leg, "S")):
            if o.get("symbol") != symbol:
                continue
            st, filled = order_status(o), safe_int(o.get("filledQty", o.get("filled_qty", 0)))
            if st == 5 or (st in {1, 7} and filled == 0):
                out.append(o)
        return out

    def traded_today(self, snap: Dict[str, Any], date: str, leg: str, symbol: str) -> bool:
        for o in orders_with_prefix(snap, tag_prefix(date, leg, "S")):
            if o.get("symbol") == symbol and (order_status(o) == 2 or safe_int(o.get("filledQty", 0)) > 0):
                return True
        return False

    def attempts_and_retry_time(self, snap: Dict[str, Any], date: str, leg: str, symbol: str) -> Tuple[int, float]:
        failed = self.failed_short_orders(snap, date, leg, symbol)
        ids = {str(o.get("id")) for o in failed} | self.resp_failed_ids.get(leg, set())
        attempts = len(ids) + self.presubmit_failures.get(leg, 0)
        last_failed = max([order_epoch(o) for o in failed] or [0.0])
        retry_at = max(self.retry_after_mem.get(leg, 0.0),
                       last_failed + RETRY_COOLDOWN_SEC if last_failed else 0.0)
        return attempts, retry_at

    def entry_blocker(self, leg: str, ignore_busy: bool = False) -> Optional[str]:
        s, snap = self.session, self.snapshot
        if not s or not s.get("live"):
            return "no live session"
        if not snap.get("ok") or time.time() - snap.get("updated_at", 0.0) > SNAPSHOT_MAX_AGE_SEC:
            return "broker snapshot unavailable or stale"
        if leg in self.inflight or (self.entry_busy.get(leg) and not ignore_busy):
            return "entry already in flight"
        date, symbol = s["date"], s[f"{leg.lower()}_symbol"]
        if self.traded_today(snap, date, leg, symbol):
            return "already traded today (found in broker order book)"
        attempts, retry_at = self.attempts_and_retry_time(snap, date, leg, symbol)
        if attempts >= MAX_ENTRY_ATTEMPTS:
            return f"max attempts reached ({attempts}/{MAX_ENTRY_ATTEMPTS}); disabled for today"
        if time.time() < retry_at:
            return f"retry cooldown {int(retry_at - time.time())}s"
        busy = group_busy_reason(snap, group_key(symbol))
        if busy:
            return f"position/orders exist: {busy}"
        return None

    def log_blocker(self, leg: str, reason: str) -> None:
        norm = re.sub(r"\d+", "#", reason)
        last = self.blocker_log.get(leg)
        if last is None or last[0] != norm or time.time() - last[1] > 30:
            self.blocker_log[leg] = (norm, time.time())
            emit("DATA", f"{leg} trigger level seen but entry blocked: {reason}")

    def check_trigger(self, leg: str, ltp: float) -> None:
        s = self.session
        if not s or not s.get("live"):
            return
        trigger = safe_float(s[f"{leg.lower()}_open"]) * (1.0 - DROP_PCT)
        # Entry rule unchanged: integer part of LTP equals integer part of the 50% level.
        if trigger <= 0 or int(ltp) != int(trigger):
            return
        with self.lock:
            if leg in self.dry_run_done and not LIVE_TRADING:
                return
            reason = self.entry_blocker(leg)
            if reason:
                self.log_blocker(leg, reason)
                return
            self.entry_busy[leg] = True
        emit("SIGNAL", f"{leg} trigger: LTP={ltp:.2f} matches 50% level={trigger:.2f} (int match)")
        threading.Thread(target=self.execute_entry, args=(leg, ltp), daemon=True, name=f"entry-{leg}").start()

    def check_funds(self) -> None:
        try:
            available = fetch_available_funds()
        except Exception as exc:
            emit("WARN", f"Funds check unavailable ({exc}); broker will still enforce margin")
            return
        if available is None:
            emit("WARN", "Could not read available balance; skipping funds check")
            return
        if available <= 0 or (MIN_REQUIRED_FUNDS > 0 and available < MIN_REQUIRED_FUNDS):
            raise InsufficientFunds(f"Available funds INR {available:,.2f} below required INR {MIN_REQUIRED_FUNDS:,.2f}")

    def execute_entry(self, leg: str, signal_ltp: float) -> None:
        try:
            s = self.session
            short_symbol = s[f"{leg.lower()}_symbol"]
            prefix, strike, _ = parse_option(short_symbol)
            try:
                hedge_symbol, hedge_strike, hedge_ltp = find_hedge_contract(s, leg, strike, prefix, signal_ltp)
                if not LIVE_TRADING:
                    self.dry_run_done.add(leg)
                    emit("DRYRUN", f"Would BUY hedge {hedge_symbol} (LTP {hedge_ltp:.2f}, target {signal_ltp * HEDGE_RATIO:.2f}) "
                                   f"and SELL {short_symbol} qty={ORDER_QTY}; no orders sent")
                    return
                self.check_funds()
                reason = self.entry_blocker(leg, ignore_busy=True)
                if reason:
                    emit("WARN", f"{leg} entry cancelled before submission: {reason}")
                    return
            except InsufficientFunds as exc:
                self.retry_after_mem[leg] = time.time() + FUNDS_RETRY_COOLDOWN_SEC
                emit("ERROR", f"{leg} entry blocked (attempt not consumed): {exc}")
                return
            except Exception as exc:
                self.presubmit_failures[leg] = self.presubmit_failures.get(leg, 0) + 1
                self.retry_after_mem[leg] = time.time() + RETRY_COOLDOWN_SEC
                emit("ERROR", f"{leg} entry failed before any order was sent "
                              f"({self.presubmit_failures[leg]} pre-submission failure(s)): {exc}")
                return
            self.submit_entry_basket(leg, short_symbol, hedge_symbol)
        finally:
            self.entry_busy[leg] = False

    def submit_entry_basket(self, leg: str, short_symbol: str, hedge_symbol: str) -> None:
        require_lot_quantity(ORDER_QTY, "entry quantity", allow_zero=False)
        date = today_key()
        snap = self.snapshot
        n = max(len(orders_with_prefix(snap, tag_prefix(date, leg, "S"))), self.submissions.get(leg, 0)) + 1
        self.submissions[leg] = n
        hedge_tag = f"{tag_prefix(date, leg, 'H')}{n}"
        short_tag = f"{tag_prefix(date, leg, 'S')}{n}"
        inf = {"submitted_at": time.time(), "leg": leg, "attempt": n, "group": group_key(short_symbol),
               "short_symbol": short_symbol, "hedge_symbol": hedge_symbol,
               "short_tag": short_tag, "hedge_tag": hedge_tag,
               "short_id": "", "hedge_id": "", "short_rejected": False, "hedge_rejected": False}
        self.inflight[leg] = inf            # recorded BEFORE sending: never a window without a guard
        payload = [entry_order_payload(hedge_symbol, 1, ORDER_QTY, hedge_tag),
                   entry_order_payload(short_symbol, -1, ORDER_QTY, short_tag)]
        emit("ORDER", f"Submitting SYNC multi-order (attempt {n}/{MAX_ENTRY_ATTEMPTS}): BUY {hedge_symbol} then SELL {short_symbol} qty={ORDER_QTY}")
        try:
            resp = api_write("place_basket_orders", data=payload)
        except Exception as exc:
            emit("ERROR", f"Multi-order outcome unknown; waiting for broker order book, NOT resubmitting: {exc}")
            return
        rows = parse_multi_order_response(resp)
        if not rows:
            if resp.get("s") == "error":
                self.inflight.pop(leg, None)
                self.presubmit_failures[leg] = self.presubmit_failures.get(leg, 0) + 1
                self.retry_after_mem[leg] = time.time() + RETRY_COOLDOWN_SEC
                emit("ERROR", f"Multi-order request rejected as a whole: {resp}")
            else:
                emit("ERROR", f"Multi-order returned no per-leg results; waiting for order book: {resp}")
            return
        for i, row in enumerate(rows[:2]):
            emit("ORDER", f"Multi-order leg {i + 1}: s={row['s']} code={row['code']} id={row['id']} msg={row['message']}")
        inf["hedge_id"], inf["hedge_rejected"] = rows[0]["id"], leg_rejected(rows[0])
        if len(rows) > 1:
            inf["short_id"], inf["short_rejected"] = rows[1]["id"], leg_rejected(rows[1])
        if inf["short_rejected"]:
            if inf["short_id"]:
                self.resp_failed_ids.setdefault(leg, set()).add(str(inf["short_id"]))
            else:
                self.presubmit_failures[leg] = self.presubmit_failures.get(leg, 0) + 1
            self.retry_after_mem[leg] = time.time() + RETRY_COOLDOWN_SEC
            emit("WARN", f"{leg} short leg rejected; retry allowed in {RETRY_COOLDOWN_SEC}s (attempt {n}/{MAX_ENTRY_ATTEMPTS} used)")

    # ------------------------------------------------------- broker sync / rules
    def maybe_sync(self) -> None:
        if time.time() - self.last_snapshot_start < READ_GAP_SEC + 0.1:
            return
        self.last_snapshot_start = time.time()
        try:
            self.snapshot = build_broker_snapshot()
        except Exception as exc:
            self.snapshot = dict(self.snapshot, ok=False, error=str(exc))
            self.last_error = str(exc)
            emit("ERROR", f"Broker sync failed: {exc}")
            return
        try:
            self.reconcile(self.snapshot)
        except Exception as exc:
            emit("ERROR", f"Reconcile error: {exc}\n{traceback.format_exc()}")

    def action_recent(self, key: Any, seconds: float) -> bool:
        return time.time() - self.actions.get(key, 0.0) < seconds

    def mark_action(self, key: Any) -> None:
        self.actions[key] = time.time()

    def reconcile(self, snap: Dict[str, Any]) -> None:
        self.clear_inflight(snap)
        for key, grp in scan_groups(snap).items():
            try:
                self.reconcile_group(key, grp, snap)
            except Exception as exc:
                emit("ERROR", f"Group {key} reconcile failed: {exc}")

    def clear_inflight(self, snap: Dict[str, Any]) -> None:
        for leg, inf in list(self.inflight.items()):
            if snap["started_at"] < inf["submitted_at"] + ENTRY_SETTLE_SEC:
                continue
            h = find_order(snap, inf["hedge_id"], inf["hedge_tag"])
            s = find_order(snap, inf["short_id"], inf["short_tag"])
            h_done = inf["hedge_rejected"] or (h is not None and order_status(h) not in PENDING)
            s_done = inf["short_rejected"] or (s is not None and order_status(s) not in PENDING)
            if h_done and s_done:
                self.inflight.pop(leg, None)
                emit("DATA", f"{leg} entry orders settled at broker (attempt {inf['attempt']})")
            elif time.time() - inf["submitted_at"] > INFLIGHT_TIMEOUT_SEC:
                self.inflight.pop(leg, None)
                emit("ERROR", f"{leg} entry orders not confirmed in order book after {INFLIGHT_TIMEOUT_SEC}s; "
                              "continuing from broker positions only")

    def reconcile_group(self, key: Tuple[str, str], grp: Dict[str, Any], snap: Dict[str, Any]) -> None:
        leg = key[1]
        inf = self.inflight.get(leg)
        if inf and inf.get("group") == key:
            return                                   # entry still settling
        if grp["conflict"]:
            self.log_blocker(f"{leg}-group", f"ambiguous positions, not auto-managed: {grp['conflict']}")
            return
        short_sym, short_qty = grp["short"], grp["short_qty"]
        hedge_sym, hedge_qty = grp["long"], grp["long_qty"]
        syms = grp["symbols"]
        group_pending = [o for o in pending_orders(snap) if o.get("symbol") in syms]
        group_ocos = [g for g in pending_ocos(snap) if g.get("symbol") in syms]
        require_lot_quantity(short_qty, "short position")
        require_lot_quantity(hedge_qty, "hedge position")

        # ---- short is flat: clean up stale OCO, then close the hedge ----------
        if not short_sym:
            if group_ocos:
                for g in group_ocos:
                    gid = str(g.get("id"))
                    if gtt_status(g) == 6 and not self.action_recent(("cancel", gid), 60):
                        self.mark_action(("cancel", gid))
                        try:
                            r = api_write("cancel_gtt_order", data={"id": gid})
                            emit("ORDER", f"Cancelling stale OCO {gid} (short is flat): {r}")
                        except Exception as exc:
                            emit("ERROR", f"Stale OCO cancel outcome unknown for {gid}: {exc}")
                return
            if group_pending or hedge_qty == 0:
                return
            if not AUTO_CLOSE_ORPHAN_HEDGES:
                self.log_blocker(f"{leg}-orphan", f"long {hedge_sym} qty={hedge_qty} has no short; auto-close disabled")
                return
            self.send_hedge_order(leg, hedge_sym, -1, hedge_qty, "X", "short is flat: closing hedge")
            return

        # ---- short is open -----------------------------------------------------
        gtt_transit = [g for g in group_ocos if gtt_status(g) == 4]
        gtt_active = [g for g in group_ocos if gtt_status(g) == 6 and g.get("symbol") == short_sym]
        pending_exit = [o for o in group_pending if o.get("symbol") == short_sym and order_side(o) == 1]
        if gtt_transit:
            return                                   # wait for a definitive GTT state
        if gtt_active and pending_exit:
            self.log_blocker(f"{leg}-dup", f"MANUAL: BUY exit order and pending OCO both live on {short_sym}")
            return

        if hedge_qty < short_qty:                    # hedge missing/short: buy it (risk-reducing)
            if group_pending:
                return
            # A short without its full hedge is also unprotected (OCO waits for the hedge),
            # so this is bounded: after MAX_HEDGE_REPAIRS tries the short is covered.
            tries = self.repair_tries.get(short_sym, 0)
            if tries >= MAX_HEDGE_REPAIRS:
                self.emergency_cover(leg, short_sym, short_qty,
                                     f"hedge could not be restored after {tries} attempts (hedge={hedge_qty}, short={short_qty})")
                return
            target_sym = hedge_sym or self.resolve_hedge_symbol(leg, key, short_sym, snap)
            if not target_sym:
                self.repair_tries[short_sym] = tries + 1   # unresolved passes count too
                return
            if self.send_hedge_order(leg, target_sym, 1, short_qty - hedge_qty, "B", "hedge smaller than short: buying hedge"):
                self.repair_tries[short_sym] = tries + 1
            return
        if hedge_qty > short_qty:                    # short shrinking: trim hedge to match
            if gtt_active:
                self.log_blocker(f"{leg}-over", f"MANUAL: hedge {hedge_qty} > short {short_qty} while OCO still pending")
                return
            self.send_hedge_order(leg, hedge_sym, -1, hedge_qty - short_qty, "X", "hedge larger than short: trimming hedge")
            return
        if gtt_active:
            for g in gtt_active:
                q1 = safe_int(g.get("qty"))
                if q1 != short_qty:
                    self.log_blocker(f"{leg}-ocoqty", f"MANUAL: OCO {g.get('id')} qty={q1} differs from short qty={short_qty}")
            return                                   # protected
        if pending_exit:
            age = max([time.time() - order_epoch(o) for o in pending_exit if order_epoch(o) > 0] or [0.0])
            if age > STUCK_EXIT_WARN_SEC:
                self.log_blocker(f"{leg}-stuck", f"MANUAL: BUY exit order on {short_sym} still unfilled after {int(age)}s "
                                                 "(limit order not executing?) - short remains open")
            return                                   # exit in progress
        self.protect_short(leg, snap, short_sym, short_qty)

    def resolve_hedge_symbol(self, leg: str, key: Tuple[str, str], short_sym: str, snap: Dict[str, Any]) -> Optional[str]:
        date = today_key()
        for o in orders_with_prefix(snap, tag_prefix(date, leg, "H")):
            if group_key(o.get("symbol")) == key and o.get("symbol"):
                return o["symbol"]
        s = self.session
        p = parse_option(short_sym)
        avg = position_avg(snap, short_sym)
        if not s or not p or avg <= 0:
            self.log_blocker(f"{leg}-nohedge", f"MANUAL: short {short_sym} has no hedge and one cannot be selected")
            return None
        try:
            return find_hedge_contract(s, leg, p[1], p[0], avg)[0]
        except Exception as exc:
            self.log_blocker(f"{leg}-nohedge", f"MANUAL: hedge selection failed for {short_sym}: {exc}")
            return None

    def send_hedge_order(self, leg: str, symbol: str, side: int, qty: int, role: str, why: str) -> bool:
        """Returns True only if an order request was actually sent."""
        if not LIVE_TRADING:
            self.log_blocker(f"{leg}-{role}", f"DRY-RUN: would {'BUY' if side == 1 else 'SELL'} {qty} {symbol} ({why})")
            return False
        if self.action_recent((role, symbol), ACTION_COOLDOWN_SEC):
            return False
        self.mark_action((role, symbol))
        try:
            qty = require_lot_quantity(qty, "hedge order quantity", allow_zero=False)
            r = place_market_order(symbol, side, qty, new_tag(leg, role))
            emit("ORDER", f"{why}: {'BUY' if side == 1 else 'SELL'} {symbol} qty={qty} -> s={r.get('s')} id={r.get('id')} msg={r.get('message')}")
            if r.get("s") != "ok":
                emit("ERROR", f"Hedge order rejected: {r}")
        except Exception as exc:
            emit("ERROR", f"Hedge order for {symbol} failed/unknown (will re-check broker): {exc}")
        return True

    def current_ltp(self, symbol: str) -> float:
        with PRICE_LOCK:
            ltp, t = CURRENT_PRICES.get(symbol, 0.0), LAST_TICK_AT.get(symbol, 0.0)
        if ltp > 0 and time.time() - t < 30:
            return ltp
        return quote_ltp(get_quotes([symbol]).get(symbol, {}))

    def protect_short(self, leg: str, snap: Dict[str, Any], short_sym: str, short_qty: int) -> None:
        if not LIVE_TRADING:
            self.log_blocker(f"{leg}-oco", f"DRY-RUN: {short_sym} has no OCO; would place one")
            return
        if self.action_recent(("oco", short_sym), OCO_RECHECK_SEC):
            return
        if self.oco_tries.get(short_sym, 0) >= MAX_OCO_TRIES:
            self.log_blocker(f"{leg}-oco", f"MANUAL: {short_sym} unprotected, OCO already tried {MAX_OCO_TRIES} times")
            return
        self.mark_action(("oco", short_sym))
        self.oco_tries[short_sym] = self.oco_tries.get(short_sym, 0) + 1
        avg = position_avg(snap, short_sym)
        try:
            ltp = self.current_ltp(short_sym)
            if avg <= 0 or ltp <= 0:
                raise OcoLevelCrossed(f"missing price data (avg={avg}, ltp={ltp})")
            submit_gtt_oco(short_sym, short_qty, avg, ltp)
        except OcoSubmissionUnknown as exc:
            emit("ERROR", f"OCO outcome unknown; waiting for GTT order book: {exc}")
        except Exception as exc:
            self.emergency_cover(leg, short_sym, short_qty, str(exc))

    def emergency_cover(self, leg: str, short_sym: str, qty: int, reason: str) -> None:
        if not LIVE_TRADING:
            self.log_blocker(f"{leg}-cover", f"DRY-RUN: would emergency-cover {short_sym}: {reason}")
            return
        if self.action_recent(("cover", short_sym), ACTION_COOLDOWN_SEC):
            return
        self.mark_action(("cover", short_sym))
        emit("ERROR", f"Protective OCO unavailable; emergency BUY to cover {short_sym} qty={qty}. Reason: {reason}")
        try:
            r = place_market_order(short_sym, 1, qty, new_tag(leg, "C"))
            emit("ORDER", f"Emergency cover response: {r}")
            if r.get("s") != "ok":
                emit("ERROR", "Emergency cover rejected - IMMEDIATE MANUAL ACTION REQUIRED")
        except Exception as exc:
            emit("ERROR", f"Emergency cover outcome unknown; check broker NOW: {exc}")

    # ---------------------------------------------------------------- dashboard
    def leg_status_text(self, leg: str) -> str:
        s, snap = self.session, self.snapshot
        if not s:
            return "NO SESSION"
        if not s.get("live"):
            return "DISPLAY ONLY"
        if not snap.get("ok"):
            return "BROKER SYNC ERR"
        if leg in self.inflight:
            return "ENTRY IN FLIGHT"
        sym, date = s[f"{leg.lower()}_symbol"], s["date"]
        if self.traded_today(snap, date, leg, sym):
            return "TRADED TODAY"
        attempts, retry_at = self.attempts_and_retry_time(snap, date, leg, sym)
        if attempts >= MAX_ENTRY_ATTEMPTS:
            return f"EXHAUSTED {attempts}/{MAX_ENTRY_ATTEMPTS}"
        if time.time() < retry_at:
            return f"RETRY IN {int(retry_at - time.time())}s ({attempts}/{MAX_ENTRY_ATTEMPTS})"
        return f"WAITING ({attempts}/{MAX_ENTRY_ATTEMPTS} used)" if attempts else "WAITING"

    def render_dashboard(self) -> None:
        if not sys.stdout.isatty():
            return
        s, snap = self.session, self.snapshot
        L = ["=" * 100, " NIFTY PREMIUM-CRASH BOT | FYERS API v3 | broker-driven (no state file)", "=" * 100,
             f" {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}  Trading: {'LIVE' if LIVE_TRADING else 'DRY-RUN'}  "
             f"Qty/leg: {ORDER_QTY}  Read gap: {READ_GAP_SEC:.0f}s"]
        ws_age = time.time() - WS_STATUS["last_event"] if WS_STATUS.get("last_event") else None
        L.append(f" WS: {'CONNECTED' if WS_STATUS.get('connected') else 'DISCONNECTED'} ({WS_STATUS.get('message')})"
                 + (f" | last msg {ws_age:.1f}s ago" if ws_age is not None else ""))
        if snap.get("ok"):
            L.append(f" Broker sync OK ({time.time() - snap['updated_at']:.0f}s ago) | positions open: "
                     f"{sum(1 for v in snap['positions'].values() if safe_int(v.get('netQty')) != 0)} | "
                     f"orders: {len(snap['orders'])} | GTT: {len(snap['gtt'])}")
        else:
            L.append(f" Broker sync ERROR: {snap.get('error')}")
        L.append("-" * 100)
        if s:
            with PRICE_LOCK:
                idx = CURRENT_PRICES.get(NIFTY_SYMBOL, 0.0)
            L.append(f" {'LIVE SESSION' if s['live'] else 'PREVIEW (market closed, not tradable)'} {s['date']} | expiry {s['expiry_date']} | "
                     f"ATM {s['atm']} | NIFTY open {s['index_open']:.2f} LTP {idx:.2f}")
            L.append(f" {'LEG':<4} {'SYMBOL':<30} {'OPEN':>9} {'LTP':>9} {'50% LVL':>9}  STATUS")
            for leg in ("CE", "PE"):
                sym, op = s[f"{leg.lower()}_symbol"], s[f"{leg.lower()}_open"]
                with PRICE_LOCK:
                    ltp = CURRENT_PRICES.get(sym, 0.0)
                L.append(f" {leg:<4} {sym:<30} {op:>9.2f} {ltp:>9.2f} {op * (1 - DROP_PCT):>9.2f}  {self.leg_status_text(leg)}")
        else:
            L.append(" Waiting for session data (index open / ATM / option opens)...")
        L.append("-" * 100)
        L.append(" BROKER POSITIONS (non-zero)")
        open_pos = [(k, v) for k, v in snap.get("positions", {}).items() if safe_int(v.get("netQty")) != 0] if snap.get("ok") else []
        for sym, row in open_pos[:6]:
            d = row["rows"][0]
            L.append(f" {sym:<38} netQty={safe_int(row['netQty']):>6} netAvg={safe_float(d.get('netAvg')):>8.2f} P&L={safe_float(d.get('pl')):>9.2f}")
        if not open_pos:
            L.append(" (none)")
        L.append(" PENDING ORDERS / ACTIVE GTT")
        for o in (pending_orders(snap) if snap.get("ok") else [])[:4]:
            L.append(f" order {o.get('id')} {o.get('symbol')} side={order_side(o)} status={order_status(o)} tag={order_tag_of(o)}")
        for g in (pending_ocos(snap) if snap.get("ok") else [])[:4]:
            L.append(f" OCO   {g.get('id')} {g.get('symbol')} status={gtt_status(g)}")
        L.append("-" * 100)
        L.append(" STRATEGY ORDERS TODAY (from broker order book)")
        mine = orders_with_prefix(snap, f"NPC{today_key().replace('-', '')}") if snap.get("ok") else []
        for o in mine[-6:]:
            L.append(f" {order_tag_of(o):<24} {str(o.get('symbol')):<28} side={order_side(o):>2} status={order_status(o)} "
                     f"filled={safe_int(o.get('filledQty'))}/{safe_int(o.get('qty'))}")
        if not mine:
            L.append(" (none)")
        L.append("-" * 100)
        # Recent events are written to bot.log, not displayed in the terminal dashboard.
        L.append("=" * 100)
        L.append(f" Log: {LOG_FILE} | Ctrl+C stops the bot (positions/GTT stay at the broker)")
        sys.stdout.write("\033[2J\033[H" + "\n".join(L) + "\n")
        sys.stdout.flush()

    def run(self) -> None:
        emit("DATA", f"Bot booting | live={LIVE_TRADING} qty={ORDER_QTY} drop={DROP_PCT:.0%} hedge={HEDGE_RATIO:.0%} "
                     f"read gap={READ_GAP_SEC:.0f}s retries={MAX_ENTRY_ATTEMPTS} cooldown={RETRY_COOLDOWN_SEC}s")
        if not LIVE_TRADING:
            emit("WARN", "LIVE_TRADING is off: no orders will be sent")
        while self.running:
            try:
                self.ensure_session()
                s = self.session
                if s: #and s["live"] and market_session_open():
                    self.start_websocket(s)
                self.maybe_sync()
                if time.time() - self.last_dashboard >= DASHBOARD_INTERVAL_SEC:
                    self.render_dashboard()
                    self.last_dashboard = time.time()
                time.sleep(0.5)
            except KeyboardInterrupt:
                self.running = False
                emit("WARN", "Ctrl+C: exiting; broker positions and GTT orders are untouched.")
            except Exception as exc:
                emit("ERROR", f"Main loop exception: {exc}\n{traceback.format_exc()}")
                time.sleep(2)


# =============================================================================
# SINGLE-INSTANCE LOCK (process lock only, not trading state)
# =============================================================================
def acquire_single_instance_lock():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.touch(exist_ok=True)
    handle = LOCK_FILE.open("r+b")
    try:
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
        raise RuntimeError(f"Another bot process holds the lock: {LOCK_FILE} ({exc})") from exc
    return handle


def release_single_instance_lock(handle) -> None:
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


def main() -> None:
    if LOT_SIZE <= 0 or LOTS <= 0:
        raise RuntimeError("LOT_SIZE and LOTS must be positive")
    require_lot_quantity(ORDER_QTY, "configured entry quantity", allow_zero=False)
    init_client()
    lock = acquire_single_instance_lock()
    try:
        PremiumCrashBot().run()
    finally:
        try:
            release_single_instance_lock(lock)
        except Exception as exc:
            logging.exception("Could not release lock: %s", exc)


if __name__ == "__main__":
    main()
