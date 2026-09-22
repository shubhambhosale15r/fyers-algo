"""
Delta-Adjusted Option Residual Breadth (Fyers API)

Instead of a naive "did the option go up or down" breadth line, this asks:

    Did the option move MORE or LESS than its delta says it should have,
    given how much NIFTY itself moved?

    ExpectedMove   = Delta_(t-1) * (NIFTY_t - NIFTY_(t-1))
    ActualMove     = LTP_t - LTP_(t-1)
    OptionResidual = ActualMove - ExpectedMove

    Residual > 0  -> advance   (option outperformed what its delta predicted)
    Residual < 0  -> decline   (option underperformed what its delta predicted)

This strips out the "obvious" part of the move driven purely by NIFTY's own
move, leaving (roughly) the IV / demand / positioning component per strike.
Net breadth = (advances - declines) across the strikes in your ATM band.

Usage:
    python option_residual_breadth.py                  # run continuously
    python option_residual_breadth.py --iterations 5    # run 5 ticks then stop
    python option_residual_breadth.py --inspect         # print raw schema once
"""

import argparse
import csv
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, time as dtime

import config
from fyers_auth import get_fyers_client

IST_MARKET_OPEN = dtime(9, 15)
IST_MARKET_CLOSE = dtime(15, 30)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptionPoint:
    symbol: str
    strike: float
    option_type: str  # "CE" or "PE"
    ltp: float
    delta: float


@dataclass
class Snapshot:
    ts: datetime
    underlying_ltp: float
    options: dict = field(default_factory=dict)  # (strike, type) -> OptionPoint


# ---------------------------------------------------------------------------
# Fetching + parsing
# ---------------------------------------------------------------------------

def fetch_raw_chain(client):
    payload = {
        "symbol": config.UNDERLYING_SYMBOL,
        "strikecount": config.STRIKE_COUNT,
        "timestamp": "",
    }
    resp = client.optionchain(data=payload)
    if resp.get("s") != "ok" and resp.get("code") not in (200, None):
        # Fyers uses "s":"ok"/"error"; surface anything unexpected loudly.
        raise RuntimeError(f"optionchain call failed: {resp}")
    return resp


def _extract_delta(opt: dict):
    """Delta may come back flat on the option dict or nested under 'greeks',
    depending on Fyers API version/plan. Try both; return None if absent."""
    if "delta" in opt and opt["delta"] not in (None, ""):
        return float(opt["delta"])
    greeks = opt.get("greeks") or {}
    if isinstance(greeks, dict) and greeks.get("delta") not in (None, ""):
        return float(greeks["delta"])
    return None


def parse_snapshot(raw: dict) -> Snapshot:
    data = raw.get("data", {})
    chain = data.get("optionsChain", [])

    underlying_ltp = None
    options = {}

    for opt in chain:
        opt_type = opt.get("option_type", "")
        if opt_type not in ("CE", "PE"):
            # The underlying index itself is usually the first row with option_type == ""
            if underlying_ltp is None and opt.get("ltp"):
                underlying_ltp = float(opt["ltp"])
            continue

        strike = opt.get("strike_price")
        ltp = opt.get("ltp")
        delta = _extract_delta(opt)
        if strike is None or ltp is None or delta is None:
            continue  # incomplete row (e.g. illiquid strike with blank greeks)

        key = (float(strike), opt_type)
        options[key] = OptionPoint(
            symbol=opt.get("symbol", ""),
            strike=float(strike),
            option_type=opt_type,
            ltp=float(ltp),
            delta=float(delta),
        )

    if underlying_ltp is None:
        # Fallback: some accounts don't return the index row inline.
        raise RuntimeError(
            "Could not find underlying LTP in option-chain response. "
            "Run with --inspect to see the raw payload."
        )

    return Snapshot(ts=datetime.now(), underlying_ltp=underlying_ltp, options=options)


def inspect_schema(client):
    raw = fetch_raw_chain(client)
    chain = raw.get("data", {}).get("optionsChain", [])
    print(f"Total rows returned: {len(chain)}\n")
    shown = 0
    for opt in chain:
        if opt.get("option_type") in ("CE", "PE"):
            print(f"--- sample {opt.get('option_type')} row ---")
            for k, v in opt.items():
                print(f"  {k}: {v}")
            print()
            shown += 1
        if shown >= 2:
            break
    if shown == 0:
        print("No CE/PE rows found — printing first raw row instead:")
        print(chain[0] if chain else raw)


# ---------------------------------------------------------------------------
# Strike-band selection
# ---------------------------------------------------------------------------

def strike_interval(strikes):
    uniq = sorted(set(strikes))
    if len(uniq) < 2:
        return 50.0
    diffs = Counter(round(b - a, 2) for a, b in zip(uniq, uniq[1:]))
    return diffs.most_common(1)[0][0]


def select_band(options: dict, underlying_ltp: float, band: int):
    strikes = [k[0] for k in options.keys()]
    if not strikes:
        return options
    interval = strike_interval(strikes)
    atm = min(strikes, key=lambda s: abs(s - underlying_ltp))
    lo, hi = atm - band * interval, atm + band * interval
    return {k: v for k, v in options.items() if lo - 1e-6 <= k[0] <= hi + 1e-6}


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_tick(prev: Snapshot, curr: Snapshot):
    delta_nifty = curr.underlying_ltp - prev.underlying_ltp
    rows = []

    common_keys = set(prev.options.keys()) & set(curr.options.keys())
    for key in common_keys:
        p, c = prev.options[key], curr.options[key]

        if abs(p.delta) < config.MIN_ABS_DELTA:
            continue

        actual_move = c.ltp - p.ltp
        expected_move = p.delta * delta_nifty
        residual = actual_move - expected_move
        eps = max(config.MIN_ABS_EPSILON, config.REL_EPSILON * p.ltp)

        if residual > eps:
            direction = "advance"
        elif residual < -eps:
            direction = "decline"
        else:
            direction = "flat"

        rows.append({
            "ts": curr.ts.isoformat(timespec="seconds"),
            "symbol": c.symbol,
            "strike": c.strike,
            "type": c.option_type,
            "delta_used": p.delta,
            "delta_nifty": delta_nifty,
            "ltp_prev": p.ltp,
            "ltp_curr": c.ltp,
            "actual_move": actual_move,
            "expected_move": expected_move,
            "residual": residual,
            "direction": direction,
        })

    return delta_nifty, rows


def summarize(rows):
    ce_adv = sum(1 for r in rows if r["type"] == "CE" and r["direction"] == "advance")
    ce_dec = sum(1 for r in rows if r["type"] == "CE" and r["direction"] == "decline")
    ce_flat = sum(1 for r in rows if r["type"] == "CE" and r["direction"] == "flat")
    pe_adv = sum(1 for r in rows if r["type"] == "PE" and r["direction"] == "advance")
    pe_dec = sum(1 for r in rows if r["type"] == "PE" and r["direction"] == "decline")
    pe_flat = sum(1 for r in rows if r["type"] == "PE" and r["direction"] == "flat")

    net_breadth = (ce_adv + pe_adv) - (ce_dec + pe_dec)
    return {
        "ce_adv": ce_adv, "ce_dec": ce_dec, "ce_flat": ce_flat,
        "pe_adv": pe_adv, "pe_dec": pe_dec, "pe_flat": pe_flat,
        "net_breadth": net_breadth,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def append_summary_log(path, ts, underlying_ltp, delta_nifty, summary, cumulative):
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp", "underlying_ltp", "delta_nifty",
                "ce_adv", "ce_dec", "ce_flat",
                "pe_adv", "pe_dec", "pe_flat",
                "net_breadth", "cumulative_breadth",
            ])
        writer.writerow([
            ts.isoformat(timespec="seconds"), underlying_ltp, round(delta_nifty, 2),
            summary["ce_adv"], summary["ce_dec"], summary["ce_flat"],
            summary["pe_adv"], summary["pe_dec"], summary["pe_flat"],
            summary["net_breadth"], cumulative,
        ])


def append_detail_log(path, rows):
    if not path or not rows:
        return
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Market hours guard
# ---------------------------------------------------------------------------

def market_is_open(now: datetime = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:  # Sat/Sun
        return False
    t = now.time()
    return IST_MARKET_OPEN <= t <= IST_MARKET_CLOSE


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(interval: int, iterations, ignore_market_hours: bool, client):
    print(f"Watching {config.UNDERLYING_SYMBOL} | ATM band ±{config.ATM_STRIKE_BAND} strikes "
          f"| tick = {interval}s\n")

    prev = None
    cumulative = 0
    i = 0

    while iterations is None or i < iterations:
        if not ignore_market_hours and not market_is_open():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed, waiting...")
            time.sleep(min(interval, 60))
            continue

        raw = fetch_raw_chain(client)
        curr = parse_snapshot(raw)
        curr.options = select_band(curr.options, curr.underlying_ltp, config.ATM_STRIKE_BAND)

        if prev is not None:
            delta_nifty, rows = compute_tick(prev, curr)
            summary = summarize(rows)
            cumulative += summary["net_breadth"]

            append_summary_log(config.LOG_FILE, curr.ts, curr.underlying_ltp, delta_nifty,
                                summary, cumulative)
            append_detail_log(config.DETAIL_LOG_FILE, rows)

            print(
                f"[{curr.ts.strftime('%H:%M:%S')}] NIFTY {curr.underlying_ltp:8.2f} "
                f"(Δ{delta_nifty:+.2f}) | "
                f"CE adv/dec {summary['ce_adv']:>2}/{summary['ce_dec']:<2} | "
                f"PE adv/dec {summary['pe_adv']:>2}/{summary['pe_dec']:<2} | "
                f"net {summary['net_breadth']:+3d} | cum {cumulative:+4d}"
            )
        else:
            print(f"[{curr.ts.strftime('%H:%M:%S')}] NIFTY {curr.underlying_ltp:8.2f} "
                  f"| baseline snapshot taken, computing residuals from next tick")

        prev = curr
        i += 1
        if iterations is None or i < iterations:
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Delta-adjusted option residual breadth")
    parser.add_argument("--interval", type=int, default=config.POLL_INTERVAL_SECONDS,
                         help="seconds between snapshots")
    parser.add_argument("--iterations", type=int, default=None,
                         help="stop after N snapshots (default: run forever)")
    parser.add_argument("--inspect", action="store_true",
                         help="print raw option-chain JSON for 2 sample rows and exit "
                              "(use this first to confirm the delta field name in your response)")
    parser.add_argument("--ignore-market-hours", action="store_true",
                         help="poll even outside 09:15-15:30 IST (useful for testing)")
    parser.add_argument("--relogin", action="store_true",
                         help="force a fresh Fyers login instead of using the cached token")
    args = parser.parse_args()

    client = get_fyers_client(force_new=args.relogin)

    if args.inspect:
        inspect_schema(client)
        return

    run(args.interval, args.iterations, args.ignore_market_hours, client)


if __name__ == "__main__":
    main()
