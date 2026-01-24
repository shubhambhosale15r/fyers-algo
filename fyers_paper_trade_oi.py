import time
import datetime
from fyers_apiv3 import fyersModel
from pprint import pprint

# ==========================================================
# CONFIGURATION
# ==========================================================
CLIENT_ID = "0F5WWD1SBL-100"
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcGROWnpVVC1Ud0k1dnRyT0twODkxdDdxNHl5OGJFTEluSjZxQm1qbHc3WlFPNFJ5N0hqNGpuRk5JTnhtYzFuc3hCS1lxdmpVeGVZMkt1NHI1VFV6ZUxPWVprbThSZWZ2ZUpTYkNSYWp2QlhFc3pGYz0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI3MjcwYzY1YTdkY2ExYzAxZDNjOTg3NjRmZmViNjAyOWRiY2Y5ODUyMmMzNGViMzRjYTUxYTAwZSIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFBMzE0OTUiLCJhcHBUeXBlIjoxMDAsImV4cCI6MTc2OTMwMTAwMCwiaWF0IjoxNzY5MjY0NzU1LCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3NjkyNjQ3NTUsInN1YiI6ImFjY2Vzc190b2tlbiJ9.gLeQcciHm-skDICqWqww3y7JEpMHIfdG8iTEOQ0N_9s"
UNDERLYING_INDEX = "NSE:NIFTY50-INDEX"
OPTION_CHAIN_SYMBOL = "NSE:NIFTY50-INDEX"

STRIKE_GAP = 50
QTY = 65 # NIFTY lot size

PCR_HIGH = 1.5
PCR_LOW = 0.5

PCR_SL_HIGH = 1.3   # Exit PE sell
PCR_SL_LOW = 0.7    # Exit CE sell

CHECK_INTERVAL = 60  # seconds

# Set to None to disable time-based blocking
NO_NEW_TRADES_AFTER = None
# NO_NEW_TRADES_AFTER = datetime.time(15, 0)  # enable if needed

# ==========================================================
# FYERS INIT
# ==========================================================
print("Initializing FYERS client...")
fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    is_async=False,
    log_path=""
)
print("FYERS client initialized successfully!")

# ==========================================================
# PAPER TRADE STATE
# ==========================================================
position = None
trades = []

# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================
def now_time():
    return datetime.datetime.now().time()

def round_to_atm(price):
    return round(price / STRIKE_GAP) * STRIKE_GAP

# ==========================================================
# FETCH DATA FUNCTIONS
# ==========================================================
def get_spot_price():
    data = fyers.quotes({"symbols": UNDERLYING_INDEX})
    lp = data["d"][0]["v"]["lp"]
    print(f"[DEBUG] Spot price fetched: {lp}")
    return lp

def get_option_chain(strikecount=10):
    data = {"symbol": OPTION_CHAIN_SYMBOL, "strikecount": strikecount, "timestamp": ""}
    try:
        resp = fyers.optionchain(data=data)
        if resp.get("s") != "ok":
            print(f"[DEBUG] Option chain error | response: {resp}")
            return []
        options_chain = resp.get("data", {}).get("optionsChain", [])
        if not options_chain:
            print("[DEBUG] Option chain empty in data")
        else:
            print(f"[DEBUG] Option chain fetched | Rows: {len(options_chain)}")
        return options_chain
    except Exception as e:
        print(f"[ERROR] Failed to fetch option chain: {e}")
        return []

# ==========================================================
# PARSE OPTION CHAIN
# ==========================================================
def build_strike_map(options_chain):
    strikes = {}
    for row in options_chain:
        strike = row.get("strike_price")
        opt_type = row.get("option_type")
        if strike == -1 or opt_type not in ("CE", "PE"):
            continue
        if strike not in strikes:
            strikes[strike] = {}
        strikes[strike][opt_type] = {
            "ltp": row.get("ltp", 0),
            "oi": row.get("oi", 0),
            "symbol": row.get("symbol")
        }
    print(f"[DEBUG] Grouped strikes: {len(strikes)}")
    return strikes

# ==========================================================
# PCR CALCULATION
# ==========================================================
def calculate_pcr(strike_map):
    pcr_data = []
    for strike, d in strike_map.items():
        ce = d.get("CE")
        pe = d.get("PE")
        if not ce or not pe or ce["oi"] == 0:
            continue
        pcr = pe["oi"] / ce["oi"]
        pcr_data.append({
            "strike": strike,
            "pcr": round(pcr, 2),
            "ce_ltp": ce["ltp"],
            "pe_ltp": pe["ltp"],
            "ce_symbol": ce["symbol"],
            "pe_symbol": pe["symbol"]
        })
    print(f"[DEBUG] PCR calculated for {len(pcr_data)} strikes")
    return pcr_data

# ==========================================================
# FETCH ATP FOR STRIKES
# ==========================================================
def get_avg_atp(symbols):
    data = {"symbols": ",".join(symbols)}
    resp = fyers.quotes(data=data)
    atp_map = {}
    if resp.get("s") == "ok":
        for item in resp.get("d", []):
            atp_map[item["n"]] = item["v"].get("atp", None)
    else:
        print(f"[DEBUG] Quotes API error: {resp}")
    return atp_map

def calculate_avg_atp(pcr_data, strikes, side):
    symbols = [x[f"{side}_symbol"] for x in pcr_data if x["strike"] in strikes]
    atp_map = get_avg_atp(symbols)
    ltps = [atp_map[s] for s in symbols if s in atp_map and atp_map[s] is not None]
    if len(ltps) != 3:
        print(f"[DEBUG] ATP calculation failed for {side.upper()} | LTPs={ltps}")
        return None
    avg_atp = sum(ltps)/3
    print(f"[DEBUG] ATP {side.upper()} | Strikes={strikes} | ATPs={ltps} | Avg ATP={avg_atp}")
    return avg_atp

# ==========================================================
# TREND CHECK
# ==========================================================
def strikes_around_atm(atm):
    return [atm - STRIKE_GAP, atm, atm + STRIKE_GAP]

def get_ltp_for_symbols(symbols):
    data = {"symbols": ",".join(symbols)}
    resp = fyers.quotes(data=data)
    ltp_map = {}
    if resp.get("s") == "ok":
        for item in resp.get("d", []):
            ltp_map[item["n"]] = item["v"]["lp"]
    else:
        print(f"[DEBUG] Quotes API error: {resp}")
    return ltp_map

def trend_check(pcr_data, atm):
    strikes = strikes_around_atm(atm)
    ce_symbols = [x["ce_symbol"] for x in pcr_data if x["strike"] in strikes]
    pe_symbols = [x["pe_symbol"] for x in pcr_data if x["strike"] in strikes]

    if len(ce_symbols) != 3 or len(pe_symbols) != 3:
        print("[DEBUG] Not enough strikes for trend calculation")
        return None, None, None, None, None

    ltp_map = get_ltp_for_symbols(ce_symbols + pe_symbols)
    avg_ce_ltp = sum(ltp_map[s] for s in ce_symbols)/3
    avg_pe_ltp = sum(ltp_map[s] for s in pe_symbols)/3

    avg_ce_atp = calculate_avg_atp(pcr_data, strikes, "ce")
    avg_pe_atp = calculate_avg_atp(pcr_data, strikes, "pe")
    if avg_ce_atp is None or avg_pe_atp is None:
        return None, None, None, None, None

    trend = None
    if avg_ce_ltp > avg_ce_atp and avg_pe_ltp < avg_pe_atp:
        trend = "BULLISH"
    elif avg_ce_ltp < avg_ce_atp and avg_pe_ltp > avg_pe_atp:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    print(f"[DEBUG] Trend check | Trend={trend} | avg CE LTP={avg_ce_ltp} | avg CE ATP={avg_ce_atp} | avg PE LTP={avg_pe_ltp} | avg PE ATP={avg_pe_atp}")
    return trend, avg_ce_ltp, avg_pe_ltp, avg_ce_atp, avg_pe_atp

# ==========================================================
# SELECT PCR STRIKE FOR TRADE
# ==========================================================
def select_pcr_strike(pcr_data, atm, trend):
    if trend not in ("BULLISH", "BEARISH"):
        return None
    extreme_strikes = [
        x for x in pcr_data
        if (trend == "BULLISH" and x["pcr"] > PCR_HIGH) or
           (trend == "BEARISH" and x["pcr"] < PCR_LOW)
    ]
    if not extreme_strikes:
        print("[DEBUG] No PCR extremes for trade")
        return None
    selected = min(extreme_strikes, key=lambda x: abs(x["strike"] - atm))
    print(f"[DEBUG] Selected strike for trade: {selected['strike']} | PCR={selected['pcr']}")
    return selected

# ==========================================================
# PAPER TRADE FUNCTIONS
# ==========================================================
def enter_trade(strike, side, price, pcr, symbol):
    global position
    position = {
        "strike": strike,
        "side": side,
        "entry_price": price,
        "entry_pcr": pcr,
        "entry_time": datetime.datetime.now(),
        "qty": QTY,
        "symbol": symbol
    }
    print(f"[ENTRY] SELL {side} {strike} @ {price} | PCR={pcr}")

def exit_trade(price, pcr):
    global position, trades
    exit_time = datetime.datetime.now()
    pnl = (position["entry_price"] - price) * position["qty"]
    trade = {
        **position,
        "exit_price": price,
        "exit_pcr": pcr,
        "exit_time": exit_time,
        "pnl": round(pnl, 2),
        "holding_minutes": round((exit_time - position["entry_time"]).seconds/60, 2)
    }
    trades.append(trade)
    print(f"[EXIT] {trade['side']} {trade['strike']} | PnL=₹{trade['pnl']} | Hold={trade['holding_minutes']} min")
    position = None

# ==========================================================
# STRATEGY SUMMARY
# ==========================================================
def print_summary():
    if not trades:
        print("No trades executed")
        return
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    print("\n========== STRATEGY SUMMARY ==========")
    print(f"Total Trades : {len(trades)}")
    print(f"Wins         : {len(wins)}")
    print(f"Losses       : {len(losses)}")
    print(f"Win Rate     : {len(wins)/len(trades)*100:.2f}%")
    print(f"Total PnL    : ₹{round(total_pnl,2)}")
    print(f"Max Profit   : ₹{max((t['pnl'] for t in wins), default=0)}")
    print(f"Max Loss     : ₹{min((t['pnl'] for t in losses), default=0)}")

# ==========================================================
# MAIN LOOP
# ==========================================================
print("🚀 FYERS PCR PAPER TRADING STARTED (FULL DEBUG MODE)")

while True:
    try:
        print("\n======== NEW CYCLE ========")
        spot = get_spot_price()
        atm = round_to_atm(spot)
        print(f"[DEBUG] ATM Strike: {atm}")

        option_chain = get_option_chain()
        if not option_chain:
            print("[DEBUG] Option chain empty — skipping cycle")
            time.sleep(CHECK_INTERVAL)
            continue

        strike_map = build_strike_map(option_chain)
        pcr_data = calculate_pcr(strike_map)

        # ================= EXIT =================
        if position:
            curr = next((x for x in pcr_data if x["strike"] == position["strike"]), None)
            if curr:
                if position["side"] == "PE" and curr["pcr"] < PCR_SL_HIGH:
                    exit_trade(curr["pe_ltp"], curr["pcr"])
                elif position["side"] == "CE" and curr["pcr"] > PCR_SL_LOW:
                    exit_trade(curr["ce_ltp"], curr["pcr"])

            # Real-time unrealized PnL
            ltp_map = get_ltp_for_symbols([position["symbol"]])
            current_ltp = ltp_map.get(position["symbol"], position["entry_price"])
            unrealized_pnl = round((position["entry_price"] - current_ltp) * position["qty"], 2)
            print(f"[DEBUG] Current position: {position} | Current LTP: {current_ltp} | Unrealized PnL: ₹{unrealized_pnl}")

        # ================= ENTRY =================
        if not position and (NO_NEW_TRADES_AFTER is None or now_time() <= NO_NEW_TRADES_AFTER):
            trend, avg_ce_ltp, avg_pe_ltp, avg_ce_atp, avg_pe_atp = trend_check(pcr_data, atm)
            if trend in ("BULLISH", "BEARISH"):
                selected = select_pcr_strike(pcr_data, atm, trend)
                if selected:
                    side = "PE" if trend=="BULLISH" else "CE"
                    price = selected["pe_ltp"] if side=="PE" else selected["ce_ltp"]
                    symbol = selected["pe_symbol"] if side=="PE" else selected["ce_symbol"]
                    print(f"[DEBUG] Ready to enter trade | Trend={trend} | Side={side} | Strike={selected['strike']} | Price={price} | PCR={selected['pcr']}")
                    enter_trade(selected["strike"], side, price, selected["pcr"], symbol)
                else:
                    print("[DEBUG] No suitable PCR strike — skipping entry")
            else:
                print("[DEBUG] Trend neutral — skipping entry")

        print(f"[DEBUG] Total trades so far: {len(trades)}")
        time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("Manual stop received")
        print_summary()
        break
    except Exception as e:
        print(f"[ERROR] Unhandled exception: {e}")
        time.sleep(30)
