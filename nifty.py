"""
===============================================================
Rolling LTQ PVT – Nifty ATM Options
===============================================================

WHAT THIS DOES:

1. Fetches the option chain every OPTION_CHAIN_INTERVAL seconds.
2. Subscribes to ATM, ATM-1, ATM+1 (CE + PE) — cumulative,
   never unsubscribes old symbols.
3. Rolling LTQ PVT is computed for ALL subscribed symbols so
   the dashboard shows live values everywhere.
   CSV is logged ONLY for the current ATM CE/PE.
4. Rolling LTQ PVT:
       price_return = (ltp - prev_ltp) / prev_ltp
       contribution = ltq * price_return
   - If current LTP == previous LTP, the tick is skipped
     (contribution would be 0 anyway). No duplicate rows.
   - If ltq == 0, the tick is skipped entirely.
   - Window: WINDOW_MINUTES of actual trade time (LTT).
5. CSV log columns:
       Timestamp, Underlying_LTP, ATM_Strike, Symbol,
       prev_ltp, current_ltp, ltq, Rolling_LTQ_PVT
6. Terminal shows live: spot, ATM strike, ATM CE/PE PVT,
   plus a rolling feed of recent ATM PVT samples.
7. NO TRADING. NO SIGNALS. Just compute + log.
===============================================================
"""

import time
import threading
import csv
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_SYMBOL = "NSE:NIFTY50-INDEX"
STRIKE_INTERVAL = 50
ATM_STRIKE_COUNT = 1        # ATM + ATM±1  -> 3 strikes -> 6 symbols

WINDOW_MINUTES = 5          # rolling window length (trade time)
OPTION_CHAIN_INTERVAL = 15  # seconds between option chain refreshes
DASHBOARD_INTERVAL = 1.0    # seconds between terminal redraws
RECENT_SAMPLES = 3         # how many recent PVT samples to show

ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcW9peGcxVnFTQVI1bW9IbG54MmNpNnVpSXAwN29WNDg4RGNpSGRYVVlBQXlfaHFJLWFDT1o4ZU9qdDJib1ZRcFZ0NDJRWHZuQklHbVp2VURmYzlvRkJLSTQyS2thR2FselBMY3JMMGdiRWxPUUI4OD0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI0NDIxYTAyNDU1ZWQyOTg3NzZhYzdiMzRmYzAzODJmYjk2MTE4Y2M1NzE3Y2NlMWE0ZWJkYmU0YyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFBMzE0OTUiLCJhcHBUeXBlIjoyMDAsImV4cCI6MTc4OTA4NjYwMCwiaWF0IjoxNzg5MDEzMDg4LCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3ODkwMTMwODgsInN1YiI6ImFjY2Vzc190b2tlbiJ9.QRcqtxIXYVp2OWSMj_WaBTfYmJfBYw0_QSXKhzoFISc"

CLIENT_ID = "J0ENYFX9ME-200"

LOG_FILE = "atm_rolling_pvt.csv"


# ============================================================
# ROLLING LTQ PVT TRACKER
# ============================================================

class RollingLTQPVT:
    """
    LTT-driven rolling PVT for one symbol.

        contribution = ltq * (ltp - prev_ltp) / prev_ltp

    Rules:
      - If current LTP == previous LTP  ->  skip (contrib is 0 anyway)
      - If ltq == 0                     ->  skip (filtered upstream too)
    """

    def __init__(self, window_seconds=300):
        self.window_seconds = window_seconds
        self.window = deque()        # (ltt_datetime, contribution)
        self.value = 0.0
        self.prev_ltt = None
        self.prev_ltp = None
        self.prev_ltq = None
        self.last_contribution = 0.0
        self.last_ltt = None
        self.first_ltt = None
        self.tick_count = 0
        self.contrib_count = 0

    def _expire(self, now_ltt):
        cutoff = now_ltt - timedelta(seconds=self.window_seconds)
        while self.window and self.window[0][0] < cutoff:
            _, old_c = self.window.popleft()
            self.value -= old_c

    def update(self, ltt, ltp, ltq):
        """
        Returns (value, changed, prev_ltp, contribution).

          value        = new rolling PVT value
          changed      = True if a new contribution was added
          prev_ltp     = the LTP that was used as `previous` for
                         this computation (None on first tick)
          contribution = the contribution added (0.0 if skipped)
        """
        self.tick_count += 1

        # First trade: baseline, no contribution
        if self.prev_ltt is None:
            self.prev_ltt = ltt
            self.prev_ltp = ltp
            self.prev_ltq = ltq
            self.last_ltt = ltt
            self.first_ltt = ltt
            return self.value, False, None, 0.0

        # Same price -> skip contribution, but still advance clock & expire
        if ltp == self.prev_ltp:
            self.prev_ltt = ltt
            self.last_ltt = ltt
            self.prev_ltq = ltq
            self._expire(ltt)
            return self.value, False, self.prev_ltp, 0.0

        # New price -> compute contribution
        prev_ltp = self.prev_ltp
        price_return = (ltp - prev_ltp) / prev_ltp
        contribution = ltq * price_return

        self.window.append((ltt, contribution))
        self.value += contribution
        self.last_contribution = contribution
        self.last_ltt = ltt
        self.contrib_count += 1

        self.prev_ltt = ltt
        self.prev_ltp = ltp
        self.prev_ltq = ltq

        self._expire(ltt)
        return self.value, True, prev_ltp, contribution

    def window_ready(self):
        """
        True once we have WINDOW_MINUTES of actual trade time.
        """
        if self.first_ltt is None or self.last_ltt is None:
            return False
        return (
            (self.last_ltt - self.first_ltt).total_seconds()
            >= self.window_seconds
        )


# ============================================================
# GLOBAL STATE
# ============================================================

# symbol -> deque of (ltt_datetime, ltp, ltq)  — all subscribed symbols
trade_data = defaultdict(lambda: deque(maxlen=20000))

# Subscribed symbols (cumulative)
subscribed_symbols = set()

# Current ATM CE / PE
atm_symbols = set()
atm_strike_current = None
underlying_ltp = None

# Rolling LTQ PVT trackers — for EVERY subscribed symbol.
# CSV logging still restricted to ATM only.
atm_pvt_trackers = {}

# Recent ATM PVT samples for terminal
recent_samples = deque(maxlen=500)

# Locks
data_lock = threading.Lock()
csv_lock = threading.Lock()
state_lock = threading.Lock()

# Websocket
fyers_ws = None

# Error
last_error = ""


# ============================================================
# HELPERS
# ============================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_atm_strike(spot_price):
    return round(spot_price / STRIKE_INTERVAL) * STRIKE_INTERVAL


def format_num(value, decimals=2):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def short_symbol(sym):
    if not sym:
        return "N/A"
    try:
        base = sym.split(":")[-1]
        if base.startswith("NIFTY") and len(base) > 5:
            rest = base[5:]
            if rest.endswith("CE"):
                return f"NIFTY {rest[:-2]} CE"
            if rest.endswith("PE"):
                return f"NIFTY {rest[:-2]} PE"
        return base
    except Exception:
        return sym


def add_log_line(msg):
    global last_error
    print(msg)
    if "ERROR" in msg.upper() or "EXCEPTION" in msg.upper():
        last_error = msg


# ============================================================
# CSV
# ============================================================

CSV_HEADER = [
    "Timestamp",
    "Underlying_LTP",
    "ATM_Strike",
    "Symbol",
    "Prev_LTP",
    "Current_LTP",
    "LTQ",
    "Rolling_LTQ_PVT",
]


def initialize_csv():
    file_exists = os.path.isfile(LOG_FILE)
    if not file_exists:
        with csv_lock:
            with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                f.flush()
    print("=" * 80)
    print("CSV LOG FILE:")
    print(os.path.abspath(LOG_FILE))
    print("=" * 80)


def log_pvt(timestamp, symbol, prev_ltp, current_ltp, ltq, rolling_pvt):
    """
    Write one row + append to terminal feed.
    Only called for ATM CE/PE when a new contribution was added.
    """
    spot = underlying_ltp
    strike = atm_strike_current

    recent_samples.append({
        "ts": timestamp,
        "spot": spot,
        "atm": strike,
        "symbol": symbol,
        "prev_ltp": prev_ltp,
        "ltp": current_ltp,
        "ltq": ltq,
        "pvt": rolling_pvt,
    })

    try:
        with csv_lock:
            with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    spot if spot is not None else "",
                    strike if strike is not None else "",
                    symbol,
                    f"{prev_ltp:.2f}" if prev_ltp is not None else "",
                    f"{current_ltp:.2f}" if current_ltp is not None else "",
                    ltq if ltq is not None else "",
                    f"{rolling_pvt:.6f}",
                ])
                f.flush()
    except Exception as e:
        add_log_line(f"[CSV ERROR] {e}")


# ============================================================
# WEBSOCKET CALLBACKS
# ============================================================

def onmessage(message):
    try:
        if message.get("type") != "sf":
            return

        symbol = message["symbol"]
        ltp = message["ltp"]
        ltq = message["last_traded_qty"]
        ltt = message["last_traded_time"]

        # ---- Skip zero-quantity heartbeats ----
        if ltq == 0:
            return

        trade_time = datetime.fromtimestamp(ltt)

        # store tick (for all subscribed symbols)
        with data_lock:
            trade_data[symbol].append((trade_time, ltp, ltq))

        # ====================================================
        # PVT TRACKER
        #
        # Created and updated for EVERY subscribed symbol.
        # CSV is logged ONLY for the current ATM CE/PE.
        # ====================================================
        tracker = atm_pvt_trackers.get(symbol)
        if tracker is None:
            tracker = RollingLTQPVT(
                window_seconds=WINDOW_MINUTES * 60
            )
            atm_pvt_trackers[symbol] = tracker

        pvt_value, changed, prev_ltp, contribution = tracker.update(
            trade_time, ltp, ltq
        )

        # Log to CSV only if this symbol is ATM AND a new
        # contribution was actually added.
        with state_lock:
            is_atm = symbol in atm_symbols

        if is_atm and changed:
            log_pvt(
                timestamp=trade_time,
                symbol=symbol,
                prev_ltp=prev_ltp,
                current_ltp=ltp,
                ltq=ltq,
                rolling_pvt=pvt_value,
            )

    except Exception as e:
        add_log_line(f"[WEBSOCKET CALLBACK ERROR] {e}")


def onerror(message):
    global last_error
    last_error = f"[WEBSOCKET ERROR] {message}"
    add_log_line(last_error)


def onclose(message):
    add_log_line(f"[WEBSOCKET CLOSED] {message}")


def onopen():
    try:
        if subscribed_symbols:
            syms = list(subscribed_symbols)
            fyers_ws.subscribe(symbols=syms, data_type="SymbolUpdate")
            add_log_line(f"[WEBSOCKET] Subscribed {len(syms)} symbols")
        else:
            add_log_line("[WEBSOCKET] No symbols yet.")
        fyers_ws.keep_running()
    except Exception as e:
        add_log_line(f"[ONOPEN ERROR] {e}")


# ============================================================
# OPTION CHAIN
# ============================================================

def fetch_option_chain_and_update():
    global subscribed_symbols, atm_symbols
    global underlying_ltp, atm_strike_current

    try:
        fyers = fyersModel.FyersModel(
            client_id=CLIENT_ID, token=ACCESS_TOKEN,
            is_async=False, log_path=""
        )

        data = {
            "symbol": INDEX_SYMBOL,
            "strikecount": ATM_STRIKE_COUNT,
            "timestamp": "",
            "greeks": "0",
        }

        response = fyers.optionchain(data=data)

        if response.get("s") != "ok":
            add_log_line(f"[OPTION CHAIN ERROR] {response}")
            return

        chain = response.get("data", {}).get("optionsChain", [])

        # underlying
        underlying = None
        for item in chain:
            if (item.get("option_type") == ""
                    and item.get("strike_price") == -1):
                underlying = item
                break

        if not underlying:
            add_log_line("[OPTION CHAIN] Underlying not found")
            return

        spot = underlying.get("ltp") or underlying.get("fp")
        if not spot:
            add_log_line("[OPTION CHAIN] No spot price")
            return

        atm_strike = get_atm_strike(spot)

        with state_lock:
            underlying_ltp = spot
            atm_strike_current = atm_strike

        strikes_wanted = {
            atm_strike - STRIKE_INTERVAL,
            atm_strike,
            atm_strike + STRIKE_INTERVAL,
        }

        chain_map = {}
        for item in chain:
            ot = item.get("option_type")
            sp = item.get("strike_price")
            if ot not in ("CE", "PE"):
                continue
            try:
                spi = int(round(float(sp)))
            except (TypeError, ValueError):
                continue
            if spi not in strikes_wanted:
                continue
            chain_map.setdefault(spi, {})[ot] = item.get("symbol")

        atm_pair = chain_map.get(atm_strike, {})
        atm_ce = atm_pair.get("CE")
        atm_pe = atm_pair.get("PE")

        if not atm_ce or not atm_pe:
            add_log_line(
                f"[OPTION CHAIN] ATM {atm_strike} missing CE/PE"
            )
            return

        new_atm = {atm_ce, atm_pe}

        aux = set()
        for strike in (
            atm_strike - STRIKE_INTERVAL,
            atm_strike + STRIKE_INTERVAL,
        ):
            pair = chain_map.get(strike, {})
            for ot in ("CE", "PE"):
                s = pair.get(ot)
                if s:
                    aux.add(s)

        new_all = new_atm | aux

        with state_lock:
            old_atm = set(atm_symbols)

        if fyers_ws:
            to_sub = new_all - subscribed_symbols
            if to_sub:
                try:
                    fyers_ws.subscribe(
                        symbols=list(to_sub),
                        data_type="SymbolUpdate"
                    )
                    add_log_line(
                        f"[SUBSCRIBE] "
                        f"{[short_symbol(s) for s in to_sub]}"
                    )
                except Exception as e:
                    add_log_line(f"[SUBSCRIBE ERROR] {e}")

            subscribed_symbols.update(new_all)

            with state_lock:
                atm_symbols = set(new_atm)

            changed = (old_atm != new_atm)
            add_log_line(
                f"[ATM UPDATE] "
                f"Spot:{format_num(spot)} "
                f"ATM:{atm_strike} "
                f"CE:{short_symbol(atm_ce)} "
                f"PE:{short_symbol(atm_pe)} "
                f"AUX:{[short_symbol(s) for s in sorted(aux)]} "
                f"Total subs:{len(subscribed_symbols)}"
                f"{'  <<< ATM CHANGED' if changed else ''}"
            )
        else:
            subscribed_symbols.update(new_all)
            with state_lock:
                atm_symbols = set(new_atm)

            add_log_line(
                f"[SYMBOL SET] WS not connected. "
                f"ATM:{[short_symbol(s) for s in new_atm]} "
                f"AUX:{[short_symbol(s) for s in aux]}"
            )

    except Exception as e:
        add_log_line(f"[OPTION CHAIN EXCEPTION] {e}")


def periodic_update():
    while True:
        try:
            fetch_option_chain_and_update()
        except Exception as e:
            add_log_line(f"[PERIODIC UPDATE ERROR] {e}")
        time.sleep(OPTION_CHAIN_INTERVAL)


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_loop():
    while True:
        try:
            time.sleep(DASHBOARD_INTERVAL)
            clear_screen()
            now = datetime.now()
            W = 150
            lines = []

            with state_lock:
                spot = underlying_ltp
                atm = atm_strike_current
                atm_snapshot = set(atm_symbols)

            atm_list = sorted(atm_snapshot)
            atm_ce = next((s for s in atm_list if s.endswith("CE")), None)
            atm_pe = next((s for s in atm_list if s.endswith("PE")), None)

            lines.append("=" * W)
            lines.append(
                "ROLLING LTQ PVT (LTT-based) – NIFTY ATM OPTIONS"
            )
            lines.append(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"CSV : {os.path.abspath(LOG_FILE)}")
            lines.append("=" * W)

            lines.append(f"Underlying LTP  : {format_num(spot)}")
            lines.append(
                f"ATM Strike      : {atm}   "
                f"(interval={STRIKE_INTERVAL})"
            )
            lines.append(f"ATM CE          : {short_symbol(atm_ce)}")
            lines.append(f"ATM PE          : {short_symbol(atm_pe)}")
            lines.append(
                f"Subscriptions   : {len(subscribed_symbols)} symbols "
                f"(cumulative, never unsubscribed)"
            )
            lines.append(
                f"Window          : {WINDOW_MINUTES} min of actual "
                f"trade time (LTT)"
            )
            lines.append(
                "PVT tracking    : ALL symbols  |  "
                "CSV logged      : ATM CE/PE only"
            )
            lines.append("-" * W)

            # PVT table
            lines.append(
                f"{'Symbol':<24}"
                f"{'Tag':<5}"
                f"{'Ticks':<8}"
                f"{'Last':<10}"
                f"{'Rolling_LTQ_PVT':<20}"
                f"{'Δ last':<12}"
                f"{'Window':<10}"
                f"{'Last LTT':<10}"
            )
            lines.append("-" * W)

            with data_lock:
                syms_snapshot = sorted(subscribed_symbols)

            for sym in syms_snapshot:
                with data_lock:
                    if trade_data[sym]:
                        latest_ltt = trade_data[sym][-1][0]
                    else:
                        latest_ltt = now
                    cutoff = latest_ltt - timedelta(minutes=WINDOW_MINUTES)
                    ticks = sum(
                        1 for it in trade_data[sym] if it[0] >= cutoff
                    )
                    last_ltp = (
                        trade_data[sym][-1][1]
                        if trade_data[sym] else None
                    )

                tracker = atm_pvt_trackers.get(sym)
                if tracker is not None:
                    pvt_text = f"{tracker.value:.6f}"
                    delta_text = f"{tracker.last_contribution:+.6f}"
                    window_ready = (
                        "YES" if tracker.window_ready() else "no"
                    )
                    last_ltt = (
                        tracker.last_ltt.strftime("%H:%M:%S")
                        if tracker.last_ltt else "-"
                    )
                else:
                    pvt_text = "-"
                    delta_text = "-"
                    window_ready = "-"
                    last_ltt = "-"

                tag = "ATM" if sym in atm_snapshot else "AUX"

                lines.append(
                    f"{short_symbol(sym):<24}"
                    f"{tag:<5}"
                    f"{ticks:<8}"
                    f"{format_num(last_ltp):<10}"
                    f"{pvt_text:<20}"
                    f"{delta_text:<12}"
                    f"{window_ready:<10}"
                    f"{last_ltt:<10}"
                )

            if not syms_snapshot:
                lines.append("NO SYMBOLS SUBSCRIBED YET")

            lines.append("-" * W)

            # Recent samples (ATM only)
            lines.append(f"RECENT ATM PVT SAMPLES (last {RECENT_SAMPLES}):")
            lines.append(
                f"{'Time':<10}"
                f"{'Spot':<12}"
                f"{'ATM':<8}"
                f"{'Symbol':<22}"
                f"{'Prev':<10}"
                f"{'Curr':<10}"
                f"{'LTQ':<8}"
                f"{'Rolling_PVT':<16}"
            )
            snap = list(recent_samples)[-RECENT_SAMPLES:]
            if not snap:
                lines.append("  (no ATM PVT samples yet)")
            else:
                for it in snap:
                    lines.append(
                        f"{it['ts'].strftime('%H:%M:%S'):<10}"
                        f"{format_num(it['spot']):<12}"
                        f"{str(it['atm'] if it['atm'] is not None else '-'):<8}"
                        f"{short_symbol(it['symbol']):<22}"
                        f"{format_num(it['prev_ltp']):<10}"
                        f"{format_num(it['ltp']):<10}"
                        f"{str(it['ltq'] if it['ltq'] is not None else '-'):<8}"
                        f"{it['pvt']:.6f}"
                    )

            lines.append("-" * W)
            if last_error:
                lines.append(f"LAST ERROR: {last_error}")
                lines.append("-" * W)

            lines.append("=" * W)
            lines.append("NO TRADING · NO SIGNALS · COMPUTE + LOG ONLY")
            lines.append("=" * W)

            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

        except Exception as e:
            add_log_line(f"[DASHBOARD ERROR] {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    global fyers_ws

    initialize_csv()

    add_log_line("[START] Rolling LTQ PVT logger started.")
    add_log_line(
        f"[CONFIG] Window={WINDOW_MINUTES} min (LTT) | "
        f"ATM refresh={OPTION_CHAIN_INTERVAL}s | "
        f"Strikes=ATM,ATM±{ATM_STRIKE_COUNT} | "
        f"PVT tracked for ALL symbols | "
        f"CSV logged for ATM CE/PE only | "
        f"Skip if LTP unchanged | "
        f"Skip if LTQ==0 | "
        f"No trading"
    )

    threading.Thread(
        target=periodic_update, daemon=True,
        name="OptionChainThread"
    ).start()

    threading.Thread(
        target=dashboard_loop, daemon=True,
        name="DashboardThread"
    ).start()

    try:
        fyers_ws = data_ws.FyersDataSocket(
            access_token=ACCESS_TOKEN,
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=onopen,
            on_close=onclose,
            on_error=onerror,
            on_message=onmessage,
        )
        add_log_line("[WEBSOCKET] Connecting...")
        fyers_ws.connect()
    except Exception as e:
        add_log_line(f"[WEBSOCKET CONNECT ERROR] {e}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        add_log_line("[SYSTEM] Stopping...")
        try:
            if fyers_ws:
                fyers_ws.close()
        except Exception as e:
            add_log_line(f"[CLOSE ERROR] {e}")
        add_log_line("[SYSTEM] Disconnected.")


if __name__ == "__main__":
    main()