import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from fyers_apiv3 import fyersModel
import logging
import uuid
import os
from functools import wraps, lru_cache
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QComboBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QTextEdit,
    QStatusBar,
    QMessageBox,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont


# Constants
class Config:
    REFRESH_INTERVAL = 10  # seconds
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
    datefmt="%Y-%m-%d %H:%M:%S",
)


class TradingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fyers Algo Trading")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize session state
        self.init_session_state()

        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        # Create sidebar
        self.create_sidebar()

        # Create main content area
        self.create_main_content()

        # Initialize timer for auto-refresh
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(Config.REFRESH_INTERVAL * 1000)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Application started")

    def init_session_state(self):
        defaults = {
            "option_chain_api_count": 0,
            "quote_api_count": 0,
            "api_window_start": datetime.now(),
            "last_signal": None,
            "last_signal_order_ids": [],
            "paper_trades": [],
            "paper_positions": {},
            "paper_pnl": {"realized": 0.0, "unrealized": 0.0},
            "trade_history": [],
            "paper_trade_active": False,
            "pnl_update_time": datetime.now(),
            "paper_last_signal": None,
            "paper_last_signal_order_ids": [],
            "live_last_signal": None,
            "live_last_signal_order_ids": [],
            "signal_history": [],
            "cid": "",
            "token": "",
            "current_page": "trading",
        }

        for key, value in defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def handle_errors(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            retries = 0
            while retries < Config.MAX_RETRIES:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    retries += 1
                    logging.error(
                        f"Error in {func.__name__} (attempt {retries}): {str(e)}",
                        exc_info=True,
                    )
                    if retries < Config.MAX_RETRIES:
                        time.sleep(Config.RETRY_DELAY)
                    else:
                        self.update_status(
                            f"Operation failed after {Config.MAX_RETRIES} attempts: {str(e)}"
                        )
                        return None

        return wrapper

    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout()
        sidebar.setLayout(sidebar_layout)

        # Settings group
        settings_group = QGroupBox("⚙️ Trading Settings")
        settings_layout = QVBoxLayout()

        self.auto_trade_check = QCheckBox("Enable Auto Trade")
        settings_layout.addWidget(self.auto_trade_check)

        self.paper_trade_check = QCheckBox("📝 Enable Paper Trading")
        settings_layout.addWidget(self.paper_trade_check)

        settings_group.setLayout(settings_layout)
        sidebar_layout.addWidget(settings_group)

        # Navigation group
        nav_group = QGroupBox("📊 Navigation")
        nav_layout = QVBoxLayout()

        self.trading_btn = QPushButton("📈 Trading Dashboard")
        self.trading_btn.clicked.connect(lambda: self.switch_page("trading"))
        nav_layout.addWidget(self.trading_btn)

        self.signal_history_btn = QPushButton("📜 Signal History")
        self.signal_history_btn.clicked.connect(
            lambda: self.switch_page("signal_history")
        )
        nav_layout.addWidget(self.signal_history_btn)

        self.paper_trading_btn = QPushButton("📊 Paper Trading")
        self.paper_trading_btn.clicked.connect(
            lambda: self.switch_page("paper_trading")
        )
        nav_layout.addWidget(self.paper_trading_btn)

        nav_group.setLayout(nav_layout)
        sidebar_layout.addWidget(nav_group)

        # Credentials group
        cred_group = QGroupBox("🔐 API Credentials")
        cred_layout = QVBoxLayout()

        self.cid_label = QLabel("Client ID:")
        self.cid_input = QLineEdit()
        self.cid_input.setText(self.cid)
        cred_layout.addWidget(self.cid_label)
        cred_layout.addWidget(self.cid_input)

        self.token_label = QLabel("Access Token:")
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setText(self.token)
        cred_layout.addWidget(self.token_label)
        cred_layout.addWidget(self.token_input)

        self.sym_label = QLabel("Choose Index:")
        self.sym_combo = QComboBox()
        self.sym_combo.addItems(["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX"])
        cred_layout.addWidget(self.sym_label)
        cred_layout.addWidget(self.sym_combo)

        self.save_creds_btn = QPushButton("Save Credentials")
        self.save_creds_btn.clicked.connect(self.save_credentials)
        cred_layout.addWidget(self.save_creds_btn)

        cred_group.setLayout(cred_layout)
        sidebar_layout.addWidget(cred_group)

        # API Usage group
        api_group = QGroupBox("📊 API Usage")
        api_layout = QVBoxLayout()

        self.oc_label = QLabel(f"Option Chain API: {self.option_chain_api_count}")
        api_layout.addWidget(self.oc_label)

        self.qc_label = QLabel(f"Quote API: {self.quote_api_count}")
        api_layout.addWidget(self.qc_label)

        self.total_label = QLabel(
            f"Total: {self.option_chain_api_count + self.quote_api_count}/{Config.API_RATE_LIMIT} per minute"
        )
        api_layout.addWidget(self.total_label)

        api_group.setLayout(api_layout)
        sidebar_layout.addWidget(api_group)

        sidebar_layout.addStretch()
        self.main_layout.addWidget(sidebar)

    def create_main_content(self):
        self.content_area = QTabWidget()
        self.main_layout.addWidget(self.content_area)

        # Create tabs
        self.trading_tab = QWidget()
        self.signal_history_tab = QWidget()
        self.paper_trading_tab = QWidget()

        self.content_area.addTab(self.trading_tab, "📈 Trading")
        self.content_area.addTab(self.signal_history_tab, "📜 Signal History")
        self.content_area.addTab(self.paper_trading_tab, "📊 Paper Trading")

        # Setup each tab
        self.setup_trading_tab()
        self.setup_signal_history_tab()
        self.setup_paper_trading_tab()

        # Initially show trading tab
        self.content_area.setCurrentIndex(0)

    def setup_trading_tab(self):
        layout = QVBoxLayout()
        self.trading_tab.setLayout(layout)

        # Header
        self.trading_header = QLabel("📈 Live Trading")
        self.trading_header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(self.trading_header)

        # Underlying LTP
        self.underlying_ltp_label = QLabel("Underlying LTP: -")
        self.underlying_ltp_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.underlying_ltp_label)

        # Option chain table
        self.option_chain_table = QTableWidget()
        self.option_chain_table.setColumnCount(15)
        self.option_chain_table.setHorizontalHeaderLabels(
            [
                "Strike",
                "CE Symbol",
                "CE LTP",
                "CE Vol",
                "CE OI",
                "PE Symbol",
                "PE LTP",
                "PE Vol",
                "PE OI",
                "CE LTPCh",
                "CE LTPChP",
                "PE LTPCh",
                "PE LTPChP",
                "CE OICh",
                "PE OICh",
            ]
        )
        self.option_chain_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        layout.addWidget(self.option_chain_table)

        # Signal display
        self.signal_group = QGroupBox("Signal Information")
        signal_layout = QVBoxLayout()

        self.signal_label = QLabel("Current Signal: -")
        self.signal_label.setFont(QFont("Arial", 14, QFont.Bold))
        signal_layout.addWidget(self.signal_label)

        self.ratio_label = QLabel("Ratio: -")
        signal_layout.addWidget(self.ratio_label)

        self.volume_ratio_label = QLabel("Volume Ratio: -")
        signal_layout.addWidget(self.volume_ratio_label)

        self.straddle_sentiment = QLabel("Straddle Sentiment: -")
        signal_layout.addWidget(self.straddle_sentiment)

        self.signal_group.setLayout(signal_layout)
        layout.addWidget(self.signal_group)

        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh Data")
        self.refresh_btn.clicked.connect(self.refresh_data)
        layout.addWidget(self.refresh_btn)

    def setup_signal_history_tab(self):
        layout = QVBoxLayout()
        self.signal_history_tab.setLayout(layout)

        # Header
        self.signal_history_header = QLabel("📜 Signal History")
        self.signal_history_header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(self.signal_history_header)

        # Signal history table
        self.signal_history_table = QTableWidget()
        self.signal_history_table.setColumnCount(4)
        self.signal_history_table.setHorizontalHeaderLabels(
            ["Timestamp", "Signal", "Underlying LTP", "Expiry Date"]
        )
        self.signal_history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        layout.addWidget(self.signal_history_table)

        # Export button
        self.export_signal_btn = QPushButton("💾 Export to CSV")
        self.export_signal_btn.clicked.connect(self.export_signal_history)
        layout.addWidget(self.export_signal_btn)

    def setup_paper_trading_tab(self):
        layout = QVBoxLayout()
        self.paper_trading_tab.setLayout(layout)

        # Header
        self.paper_trading_header = QLabel("📊 Paper Trading Dashboard")
        self.paper_trading_header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(self.paper_trading_header)

        # PnL metrics
        pnl_layout = QHBoxLayout()

        self.realized_pnl_label = QLabel("💰 Realized PnL: ₹0.00")
        self.realized_pnl_label.setFont(QFont("Arial", 12))
        pnl_layout.addWidget(self.realized_pnl_label)

        self.unrealized_pnl_label = QLabel("📊 Unrealized PnL: ₹0.00")
        self.unrealized_pnl_label.setFont(QFont("Arial", 12))
        pnl_layout.addWidget(self.unrealized_pnl_label)

        self.total_pnl_label = QLabel("💵 Total PnL: ₹0.00")
        self.total_pnl_label.setFont(QFont("Arial", 12, QFont.Bold))
        pnl_layout.addWidget(self.total_pnl_label)

        layout.addLayout(pnl_layout)

        # Positions table
        self.positions_group = QGroupBox("📊 Current Positions")
        positions_layout = QVBoxLayout()

        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(7)
        self.positions_table.setHorizontalHeaderLabels(
            ["Symbol", "Action", "Qty", "Entry Price", "Current Price", "PnL", "Signal"]
        )
        self.positions_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        positions_layout.addWidget(self.positions_table)

        self.positions_group.setLayout(positions_layout)
        layout.addWidget(self.positions_group)

        # Trade history table
        self.trade_history_group = QGroupBox("📜 Trade History")
        trade_history_layout = QVBoxLayout()

        self.trade_history_table = QTableWidget()
        self.trade_history_table.setColumnCount(8)
        self.trade_history_table.setHorizontalHeaderLabels(
            ["Timestamp", "Symbol", "Action", "Qty", "Price", "Signal", "Type", "PnL"]
        )
        self.trade_history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        trade_history_layout.addWidget(self.trade_history_table)

        self.trade_history_group.setLayout(trade_history_layout)
        layout.addWidget(self.trade_history_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.export_trades_btn = QPushButton("💾 Export Trade History")
        self.export_trades_btn.clicked.connect(self.export_trade_history)
        button_layout.addWidget(self.export_trades_btn)

        self.reset_paper_btn = QPushButton("🔄 Reset Paper Trading")
        self.reset_paper_btn.clicked.connect(self.reset_paper_trading)
        button_layout.addWidget(self.reset_paper_btn)

        layout.addLayout(button_layout)

    @handle_errors
    def enforce_rate_limit(self):
        now = datetime.now()
        window = now - self.api_window_start
        total_calls = self.option_chain_api_count + self.quote_api_count

        if window > timedelta(seconds=Config.API_WINDOW):
            self.api_window_start = now
            self.option_chain_api_count = 0
            self.quote_api_count = 0
        elif total_calls >= Config.API_RATE_LIMIT:
            wait = Config.API_WINDOW - window.seconds
            logging.warning(f"Approaching rate limit, waiting {wait}s...")
            self.update_status(f"Approaching rate limit, waiting {wait}s...")
            time.sleep(wait)
            self.api_window_start = datetime.now()
            self.option_chain_api_count = 0
            self.quote_api_count = 0

    @handle_errors
    @lru_cache(maxsize=32)
    def get_option_chain_data(self, cid, token, sym, expiry_ts=""):
        self.enforce_rate_limit()
        self.option_chain_api_count += 1
        fy = fyersModel.FyersModel(
            client_id=cid, token=token, is_async=False, log_path=""
        )
        resp = fy.optionchain(data={"symbol": sym, "timestamp": expiry_ts})
        if not resp or resp.get("code") != 200:
            error_msg = f"Option chain error ({expiry_ts or 'nearest'}): {resp.get('message', 'Unknown')}"
            logging.error(error_msg)
            self.update_status(error_msg)
            return None
        return resp["data"]["optionsChain"]

    @handle_errors
    def get_underlying_ltp(self, cid, token, sym):
        self.enforce_rate_limit()
        self.quote_api_count += 1
        fy = fyersModel.FyersModel(
            client_id=cid, token=token, is_async=False, log_path=""
        )
        resp = fy.quotes(data={"symbols": sym})
        if not resp or resp.get("code") != 200 or not resp.get("d"):
            error_msg = f"Quotes API error ({resp.get('code')}): {resp.get('message', 'Unknown')}"
            logging.error(error_msg)
            self.update_status(error_msg)
            return None
        return resp["d"][0]["v"]["lp"]

    @handle_errors
    def get_symbol_ltp(self, cid, token, symbol):
        self.enforce_rate_limit()
        self.quote_api_count += 1
        fy = fyersModel.FyersModel(
            client_id=cid, token=token, is_async=False, log_path=""
        )
        resp = fy.quotes(data={"symbols": symbol})
        if not resp or resp.get("code") != 200 or not resp.get("d"):
            error_msg = (
                f"Quotes API error for {symbol}: {resp.get('message', 'Unknown')}"
            )
            logging.error(error_msg)
            self.update_status(error_msg)
            return None
        return resp["d"][0]["v"]["lp"]

    # def compute_signals(self, merged_df, atm_strike, ltp):
    #     ratio = 0
    #     volume_ratio = 0
    #     final_pcp_bias_ratio = 0
    #     try:
    #         # ATM prices
    #         atm_row = merged_df[merged_df["Strike"] == atm_strike]
    #         atm_ce_price = atm_row["CE_ltp"].values[0]
    #         atm_pe_price = atm_row["PE_ltp"].values[0]
    #         atm_ce_vol = atm_row["CE_Vol"].values[0]
    #         atm_pe_vol = atm_row["PE_Vol"].values[0]
    #         atm_moneyflow_ce = atm_ce_price * atm_ce_vol
    #         atm_moneyflow_pe = atm_pe_price * atm_pe_vol

    #         atm_strikes = sorted(merged_df["Strike"].dropna().unique().tolist())
    #         idx_atm = atm_strikes.index(atm_strike)
    #         atm_minus1 = atm_strikes[max(0, idx_atm - 1) : idx_atm]
    #         atm_plus1 = atm_strikes[idx_atm + 1 : min(len(atm_strikes), idx_atm + 2)]
    #         atm_plus1_df = merged_df[merged_df["Strike"].isin(atm_plus1)]
    #         atm_minus1_df = merged_df[merged_df["Strike"].isin(atm_minus1)]

    #         # calculate put call parity (PCP) bias ratios
    #         pcp_bias_ratios = []

    #         # Loop through atm-1, atm, atm+1
    #         for strike in [
    #             atm_minus1_df["Strike"].values[0] if not atm_minus1_df.empty else atm_strike,
    #             atm_strike,
    #             atm_plus1_df["Strike"].values[0] if not atm_plus1_df.empty else atm_strike,
    #         ]:
    #             row = merged_df[merged_df["Strike"] == strike]
    #             if not row.empty:
    #                 ce_ltp = row["CE_ltp"].values[0]
    #                 pe_ltp = row["PE_ltp"].values[0]

    #                 # PCP difference
    #                 pcp_diff = (ce_ltp - pe_ltp) - (ltp - strike)

    #                 # Average option price for normalization
    #                 avg_option_price = (ce_ltp + pe_ltp) / 2

    #                 if avg_option_price != 0:
    #                     pcp_bias = pcp_diff / avg_option_price
    #                     # Cap to [-1, 1] to prevent spikes
    #                     pcp_bias = max(-1, min(1, pcp_bias))
    #                 else:
    #                     pcp_bias = 0

    #                 pcp_bias_ratios.append(pcp_bias)
    #                 # print(f"PCP Bias for {strike}: {pcp_bias:.2f}")

    #         # You can then average across atm-1, atm, atm+1
    #         final_pcp_bias_ratio = (
    #             sum(pcp_bias_ratios) / len(pcp_bias_ratios) if pcp_bias_ratios else 0
    #         )

    #         # Define ITM strikes
    #         strikes = sorted(merged_df["Strike"].dropna().unique().tolist())
    #         idx = strikes.index(atm_strike)
    #         ce_itm_strikes = strikes[max(0, idx - 5) : idx]
    #         pe_itm_strikes = strikes[idx + 1 : min(len(strikes), idx + 6)]

    #         # Filter DF
    #         ce_itm_df = merged_df[merged_df["Strike"].isin(ce_itm_strikes)]
    #         pe_itm_df = merged_df[merged_df["Strike"].isin(pe_itm_strikes)]

    #         # Compute moneyflow
    #         itm_ce_moneyflow = (ce_itm_df["CE_ltp"] * ce_itm_df["CE_Vol"]).sum()
    #         itm_pe_moneyflow = (pe_itm_df["PE_ltp"] * pe_itm_df["PE_Vol"]).sum()

    #         # Compute volume
    #         itm_ce_vol = ce_itm_df["CE_Vol"].sum()
    #         itm_pe_vol = pe_itm_df["PE_Vol"].sum()

    #         # Core directional ratio
    #         numerator = itm_ce_moneyflow - itm_pe_moneyflow
    #         denominator = abs(itm_ce_moneyflow) + abs(itm_pe_moneyflow)
    #         ratio = numerator / denominator if denominator != 0 else 0

    #         # Volume bias
    #         volume_ratio = (
    #             (itm_ce_vol - itm_pe_vol) / (itm_ce_vol + itm_pe_vol)
    #             if (itm_ce_vol + itm_pe_vol) != 0
    #             else 0
    #         )
    #         print(f"ratios: ratio={ratio:.2f}, volume_ratio={volume_ratio:.2f}, final_pcp_bias_ratio={final_pcp_bias_ratio:.2f}")
    #         # Signal logic
    #         if ratio > 0.33 and volume_ratio > 0.33 and final_pcp_bias_ratio > 0:
    #             signal = "BUY"
    #         elif ratio < -0.33 and volume_ratio < -0.33 and final_pcp_bias_ratio < 0:
    #             signal = "SELL"
    #         else:
    #             signal = "SIDEWAYS"

    #         logging.info(f"Signal Computation: ratio={ratio:.2f}, signal={signal}")
    #     except Exception as e:
    #         logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
    #         signal = "SIDEWAYS"
    #         ratio = 0
    #         volume_ratio = 0
    #         final_pcp_bias_ratio = 0
        
    #     # Always update UI with ratios, even on error
    #     self.ratio_label.setText(f"Ratio: {ratio:.2f}")
    #     self.volume_ratio_label.setText(f"Volume Ratio: {volume_ratio:.2f}")
    #     self.pcp_ratio_label.setText(f"PCP Bias Ratio: {final_pcp_bias_ratio:.2f}")

    #     return signal, ratio
    def compute_signals(self, merged_df, atm_strike, ltp):
        try:
            # Helper: safely get from a Series, else default
            def safe_val(series, default=0):
                if series.empty:
                    return default
                val = series.iloc[0]
                return default if pd.isna(val) else val

            # Ensure Strike list
            strikes = sorted(merged_df["Strike"].dropna().unique().tolist())
            if not strikes:
                logging.error("No valid strikes in data")
                return "SIDEWAYS", 0

            # ATM strike fallback
            if atm_strike not in strikes:
                atm_strike = min(strikes, key=lambda s: abs(s - atm_strike))

            idx_atm = strikes.index(atm_strike)

            # Fill NaNs with zeros
            merged_df = merged_df.fillna(0)

            # Precompute moneyflow and straddles
            merged_df["CE_moneyflow"] = merged_df["CE_ltp"] * merged_df["CE_Vol"]
            merged_df["PE_moneyflow"] = merged_df["PE_ltp"] * merged_df["PE_Vol"]
            merged_df["Straddle"] = merged_df["CE_ltp"] + merged_df["PE_ltp"]

            # ATM row values
            atm_row = merged_df[merged_df["Strike"] == atm_strike]
            atm_ce_price = safe_val(atm_row["CE_ltp"])
            atm_pe_price = safe_val(atm_row["PE_ltp"])
            atm_ce_vol = safe_val(atm_row["CE_Vol"])
            atm_pe_vol = safe_val(atm_row["PE_Vol"])

            # Define ITM strikes for CE & PE (5 strikes each side)
            ce_itm_strikes = strikes[max(0, idx_atm - 5):idx_atm]
            pe_itm_strikes = strikes[idx_atm + 1:min(len(strikes), idx_atm + 6)]

            ce_itm_df = merged_df[merged_df["Strike"].isin(ce_itm_strikes)]
            pe_itm_df = merged_df[merged_df["Strike"].isin(pe_itm_strikes)]

            # ==== STRADDLE SENTIMENT ====
            # Use same range as ITM strikes for both sides
            broad_strikes = ce_itm_strikes + [atm_strike] + pe_itm_strikes
            broad_df = merged_df[merged_df["Strike"].isin(broad_strikes)]
            
            # Calculate average straddle prices for each side
            ce_side_avg = broad_df[broad_df["Strike"].isin(ce_itm_strikes)]["Straddle"].mean()
            pe_side_avg = broad_df[broad_df["Strike"].isin(pe_itm_strikes)]["Straddle"].mean()

            # Determine straddle sentiment
            if pe_side_avg > ce_side_avg:
                straddle_sentiment = "positive"
            elif ce_side_avg > pe_side_avg:
                straddle_sentiment = "negative"
            else:
                straddle_sentiment = "neutral"

            # ==== RATIO & VOLUME RATIO ====
            # Calculate money flow (price * volume)
            itm_ce_moneyflow = ce_itm_df["CE_moneyflow"].sum()
            itm_pe_moneyflow = pe_itm_df["PE_moneyflow"].sum()

            # Calculate volume
            itm_ce_vol = ce_itm_df["CE_Vol"].sum()
            itm_pe_vol = pe_itm_df["PE_Vol"].sum()

            # Core directional ratio (money flow based)
            numerator = itm_ce_moneyflow - itm_pe_moneyflow
            denominator = abs(itm_ce_moneyflow) + abs(itm_pe_moneyflow)
            ratio = numerator / denominator if denominator != 0 else 0

            # Volume ratio
            volume_ratio = ((itm_ce_vol - itm_pe_vol) / (itm_ce_vol + itm_pe_vol)) if (itm_ce_vol + itm_pe_vol) != 0 else 0

            # ==== SIGNAL LOGIC ====
            if (ratio > 0.33 and 
                volume_ratio > 0.33 and 
                straddle_sentiment == "positive"):
                signal = "BUY"
            elif (ratio < -0.33 and 
                volume_ratio < -0.33 and 
                straddle_sentiment == "negative"):
                signal = "SELL"
            else:
                signal = "SIDEWAYS"

            # Update UI with the metrics
            self.ratio_label.setText(f"Ratio: {ratio:.2f}")
            self.volume_ratio_label.setText(f"Volume Ratio: {volume_ratio:.2f}")
            self.straddle_sentiment.setText(f"Straddle Sentiment: {straddle_sentiment}")

            logging.info(f"Signal Computation: ratio={ratio:.2f}, volume_ratio={volume_ratio:.2f}, "
                        f"straddle_sentiment={straddle_sentiment}, signal={signal}")
            
            return signal, ratio
        
        except Exception as e:
            logging.error(f"Error in compute_signals: {str(e)}", exc_info=True)
            return "SIDEWAYS", 0
    def select_strikes_atm_half_price(self, chain, atm_strike):
        try:
            df = pd.DataFrame(chain)
            if df.empty or atm_strike is None:
                return None, None, None, None

            ce = df[df["option_type"] == "CE"].copy()
            pe = df[df["option_type"] == "PE"].copy()

            # Find ATM PE and CE rows
            atm_pe_row = pe[pe["strike_price"] == atm_strike]
            atm_ce_row = ce[ce["strike_price"] == atm_strike]

            sell_pe_row = atm_pe_row.iloc[0].to_dict() if not atm_pe_row.empty else None
            sell_ce_row = atm_ce_row.iloc[0].to_dict() if not atm_ce_row.empty else None

            # Find half-price strikes with vectorized operations
            buy_pe_row = None
            if sell_pe_row and "ltp" in sell_pe_row and sell_pe_row["ltp"] > 0:
                target = sell_pe_row["ltp"] / 4
                pe_candidates = pe[pe["strike_price"] != atm_strike].copy()
                pe_candidates["ltp_diff"] = (pe_candidates["ltp"] - target).abs()
                if not pe_candidates.empty:
                    buy_pe_row = (
                        pe_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()
                    )

            buy_ce_row = None
            if sell_ce_row and "ltp" in sell_ce_row and sell_ce_row["ltp"] > 0:
                target = sell_ce_row["ltp"] / 4
                ce_candidates = ce[ce["strike_price"] != atm_strike].copy()
                ce_candidates["ltp_diff"] = (ce_candidates["ltp"] - target).abs()
                if not ce_candidates.empty:
                    buy_ce_row = (
                        ce_candidates.nsmallest(1, "ltp_diff").iloc[0].to_dict()
                    )

            return sell_pe_row, buy_pe_row, sell_ce_row, buy_ce_row

        except Exception as e:
            logging.error(
                f"Error in select_strikes_atm_half_price: {str(e)}", exc_info=True
            )
            return None, None, None, None

    def handle_paper_trade(self, signal, sell_pe, buy_pe, sell_ce, buy_ce):
        try:
            orders = []
            lot = Config.LOT_SIZE
            timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            current_signal = self.paper_last_signal

            # Skip if signal is the same as last time
            if current_signal == signal and self.paper_positions:
                logging.info(
                    f"[PAPER] Signal {signal} is already active. No new trades."
                )
                return []

            # Case 1: BUY → SELL (Exit BUY, Enter SELL)
            if current_signal == "BUY" and signal == "SELL":
                for symbol, position in list(self.paper_positions.items()):
                    close_action = "SELL" if position["action"] == "BUY" else "BUY"
                    orders.append(
                        {
                            "symbol": symbol,
                            "qty": position["qty"],
                            "action": close_action,
                            "price": position["current_price"],
                            "type": "CLOSE",
                        }
                    )
                logging.info("[PAPER] Exiting BUY positions for SELL signal")

                if buy_ce and sell_ce:
                    orders.append(
                        {
                            "symbol": buy_ce["symbol"],
                            "qty": lot,
                            "side": 1,
                            "action": "BUY",
                            "price": buy_ce["ltp"],
                        }
                    )
                    orders.append(
                        {
                            "symbol": sell_ce["symbol"],
                            "qty": lot,
                            "side": -1,
                            "action": "SELL",
                            "price": sell_ce["ltp"],
                        }
                    )

            # Case 2: SELL → BUY (Exit SELL, Enter BUY)
            elif current_signal == "SELL" and signal == "BUY":
                for symbol, position in list(self.paper_positions.items()):
                    close_action = "SELL" if position["action"] == "BUY" else "BUY"
                    orders.append(
                        {
                            "symbol": symbol,
                            "qty": position["qty"],
                            "action": close_action,
                            "price": position["current_price"],
                            "type": "CLOSE",
                        }
                    )
                logging.info("[PAPER] Exiting SELL positions for BUY signal")

                if buy_pe and sell_pe:
                    orders.append(
                        {
                            "symbol": buy_pe["symbol"],
                            "qty": lot,
                            "side": 1,
                            "action": "BUY",
                            "price": buy_pe["ltp"],
                        }
                    )
                    orders.append(
                        {
                            "symbol": sell_pe["symbol"],
                            "qty": lot,
                            "side": -1,
                            "action": "SELL",
                            "price": sell_pe["ltp"],
                        }
                    )

            # Case 3: New signal (no existing position)
            elif current_signal is None and signal in ["BUY", "SELL"]:
                if signal == "BUY" and buy_pe and sell_pe:
                    orders.append(
                        {
                            "symbol": buy_pe["symbol"],
                            "qty": lot,
                            "side": 1,
                            "action": "BUY",
                            "price": buy_pe["ltp"],
                        }
                    )
                    orders.append(
                        {
                            "symbol": sell_pe["symbol"],
                            "qty": lot,
                            "side": -1,
                            "action": "SELL",
                            "price": sell_pe["ltp"],
                        }
                    )
                elif signal == "SELL" and buy_ce and sell_ce:
                    orders.append(
                        {
                            "symbol": buy_ce["symbol"],
                            "qty": lot,
                            "side": 1,
                            "action": "BUY",
                            "price": buy_ce["ltp"],
                        }
                    )
                    orders.append(
                        {
                            "symbol": sell_ce["symbol"],
                            "qty": lot,
                            "side": -1,
                            "action": "SELL",
                            "price": sell_ce["ltp"],
                        }
                    )

            # Case 4: SIDEWAYS (do nothing)
            elif signal == "SIDEWAYS":
                logging.info("[PAPER] SIDEWAYS: Holding existing positions")
                return []

            # Execute orders
            placed_ids = []
            for order in orders:
                trade_id = str(uuid.uuid4())[:8]
                trade = {
                    "id": trade_id,
                    "timestamp": timestamp,
                    "symbol": order["symbol"],
                    "qty": order["qty"],
                    "action": order["action"],
                    "price": order["price"],
                    "signal": signal,
                    "type": order.get("type", "OPEN"),
                }
                self.update_paper_positions(trade)
                self.trade_history.append(trade)
                placed_ids.append(trade_id)
                logging.info(
                    f"[PAPER] {order['action']} {order['qty']} of {order['symbol']} "
                    f"at {order['price']}"
                )

            if placed_ids and signal in ["BUY", "SELL"]:
                self.paper_last_signal = signal

            return placed_ids

        except Exception as e:
            logging.error(f"Error in handle_paper_trade: {str(e)}", exc_info=True)
            return []

    def update_paper_positions(self, trade):
        try:
            symbol = trade["symbol"]
            if trade["type"] == "OPEN":
                self.paper_positions[symbol] = {
                    "symbol": trade["symbol"],
                    "id": trade["id"],
                    "entry_time": trade["timestamp"],
                    "qty": trade["qty"],
                    "action": trade["action"],
                    "side": 1 if trade["action"] == "BUY" else -1,
                    "entry_price": trade["price"],
                    "current_price": trade["price"],
                    "signal": trade["signal"],
                }
            else:
                if symbol in self.paper_positions:
                    position = self.paper_positions[symbol]

                    # Calculate PnL
                    if position["action"] == "BUY":
                        pnl = (trade["price"] - position["entry_price"]) * position[
                            "qty"
                        ]
                    else:
                        pnl = (position["entry_price"] - trade["price"]) * position[
                            "qty"
                        ]

                    self.paper_pnl["realized"] += pnl
                    trade["pnl"] = pnl

                    del self.paper_positions[symbol]

                    if not self.paper_positions:
                        self.last_signal = None
        except Exception as e:
            logging.error(f"Error in update_paper_positions: {str(e)}", exc_info=True)

    @handle_errors
    def update_unrealized_pnl(self, cid, token):
        if not self.paper_positions:
            self.paper_pnl["unrealized"] = 0.0
            return

        total_unrealized = 0.0
        for symbol, position in self.paper_positions.items():
            ltp = self.get_symbol_ltp(cid, token, symbol)
            if ltp is None:
                ltp = position.get("current_price", position["entry_price"])

            position["current_price"] = ltp

            if position["action"] == "BUY":
                pnl = (ltp - position["entry_price"]) * position["qty"]
            else:
                pnl = (position["entry_price"] - ltp) * position["qty"]

            total_unrealized += pnl

        self.paper_pnl["unrealized"] = total_unrealized
        self.pnl_update_time = datetime.now()

    @handle_errors
    def update_position_prices(self, cid, token):
        if not self.paper_positions:
            return

        for symbol, position in self.paper_positions.items():
            ltp = self.get_symbol_ltp(cid, token, symbol)
            if ltp is not None:
                position["current_price"] = ltp

    def handle_basket_orders_atomic(
        self, signal, sell_pe, buy_pe, sell_ce, buy_ce, fyers
    ):
        try:
            orders = []
            lot = Config.LOT_SIZE
            current_signal = self.last_signal

            # Skip if signal is the same as last time
            if current_signal == signal:
                logging.info(f"Signal {signal} is already active. No new trades.")
                return []

            # Case 1: BUY → SELL (Exit BUY, Enter SELL)
            if current_signal == "BUY" and signal == "SELL":
                try:
                    positions = fyers.positions().get("netPositions", [])
                    for p in positions:
                        if p.get("segment") == "NFO" and p.get("qty") != 0:
                            side = -1 if p["qty"] > 0 else 1
                            orders.append(
                                {
                                    "symbol": p["symbol"],
                                    "qty": abs(int(p["qty"])),
                                    "type": 2,
                                    "side": side,
                                    "productType": "MARGIN",
                                    "validity": "DAY",
                                }
                            )
                    logging.info("[LIVE] Exiting BUY positions for SELL signal")
                except Exception as e:
                    logging.error(f"Failed to exit positions: {e}", exc_info=True)
                    return []

                if buy_ce and sell_ce:
                    orders.append(
                        {
                            "symbol": buy_ce["symbol"],
                            "qty": lot,
                            "type": 2,
                            "side": 1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                        }
                    )
                    orders.append(
                        {
                            "symbol": sell_ce["symbol"],
                            "qty": lot,
                            "type": 2,
                            "side": -1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                        }
                    )

            # Case 2: SELL → BUY (Exit SELL, Enter BUY)
            elif current_signal == "SELL" and signal == "BUY":
                try:
                    positions = fyers.positions().get("netPositions", [])
                    for p in positions:
                        if p.get("segment") == "NFO" and p.get("qty") != 0:
                            side = -1 if p["qty"] > 0 else 1
                            orders.append(
                                {
                                    "symbol": p["symbol"],
                                    "qty": abs(int(p["qty"])),
                                    "type": 2,
                                    "side": side,
                                    "productType": "MARGIN",
                                    "validity": "DAY",
                                }
                            )
                    logging.info("[LIVE] Exiting SELL positions for BUY signal")
                except Exception as e:
                    logging.error(f"Failed to exit positions: {e}", exc_info=True)
                    return []

                if buy_pe and sell_pe:
                    orders.append(
                        {
                            "symbol": buy_pe["symbol"],
                            "qty": lot,
                            "type": 2,
                            "side": 1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                        }
                    )
                    orders.append(
                        {
                            "symbol": sell_pe["symbol"],
                            "qty": lot,
                            "type": 2,
                            "side": -1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                        }
                    )

            # Case 3: New signal (no existing position)
            elif current_signal is None and signal in ["BUY", "SELL"]:
                if signal == "BUY" and buy_pe and sell_pe:
                    orders.append(
                        {
                            "symbol": buy_pe["symbol"],
                            "qty": lot,
                            "type": 2,
                            "side": 1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                        }
                    )
                    orders.append(
                        {
                            "symbol": sell_pe["symbol"],
                            "qty": lot,
                            "type": 2,
                            "side": -1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                        }
                    )
                elif signal == "SELL" and buy_ce and sell_ce:
                    orders.append(
                        {
                            "symbol": buy_ce["symbol"],
                            "qty": lot,
                            "type": 2,
                            "side": 1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                        }
                    )
                    orders.append(
                        {
                            "symbol": sell_ce["symbol"],
                            "qty": lot,
                            "type": 2,
                            "side": -1,
                            "productType": "MARGIN",
                            "validity": "DAY",
                        }
                    )

            # Case 4: SIDEWAYS (do nothing)
            elif signal == "SIDEWAYS":
                logging.info("[LIVE] SIDEWAYS: Holding positions")
                return []

            # Execute orders
            placed_ids = []
            for order in orders:
                oid, status = self.place_order_and_check(fyers, order)
                if oid is None or status in [1, 5]:
                    logging.error(f"Order failed: {order['symbol']}")
                    for pid in placed_ids:
                        self.cancel_order(fyers, pid)
                    return []
                placed_ids.append(oid)

            if placed_ids:
                self.last_signal = signal
                logging.info(f"[LIVE] {signal} basket executed!")
                self.update_status(f"[LIVE] {signal} basket executed!")

            return placed_ids

        except Exception as e:
            logging.error(
                f"Error in handle_basket_orders_atomic: {str(e)}", exc_info=True
            )
            return []

    @handle_errors
    def cancel_order(self, fyers, order_id):
        try:
            resp = fyers.cancel_order(data={"id": order_id})
            logging.info(f"Cancelled order {order_id}: {resp}")
            return resp
        except Exception as e:
            logging.error(
                f"Exception while cancelling order {order_id}: {e}", exc_info=True
            )
            raise

    @handle_errors
    def get_order_status_by_id(self, fyers, order_id):
        try:
            orderbook = fyers.orderbook()
            for o in orderbook.get("orderBook", []):
                if o.get("id") == order_id:
                    return o.get("status")
            return None
        except Exception as e:
            logging.warning(f"Could not fetch order status for {order_id}: {e}")
            return None

    @handle_errors
    def place_order_and_check(self, fyers, order):
        try:
            resp = fyers.place_order(data=order)
            logging.info(f"Order placed for {order['symbol']}: {resp}")

            if resp.get("s") == "ok":
                oid = resp.get("id")
                time.sleep(2)  # Wait for order to process
                status = self.get_order_status_by_id(fyers, oid)
                if status in [1, 5]:  # Rejected or canceled
                    logging.error(
                        f"Order {oid} was rejected or canceled. Status: {status}"
                    )
                    return None, status
                logging.info(f"Order {oid} status: {status}")
                return oid, status
            else:
                logging.error(f"Order failed: {resp}")
                return None, None
        except Exception as e:
            logging.exception(f"Place order error: {e}")
            raise

    def has_open_orders_for_last_signal(self, fyers):
        ids = self.live_last_signal_order_ids
        if not ids:
            return False
        try:
            ob = fyers.orderbook().get("orderBook", [])
            return any(o.get("id") in ids and o.get("status") == 2 for o in ob)
        except Exception as e:
            logging.warning(f"Could not check open orders: {e}")
            return False

    def update_option_chain_table(self, data, atm_strike):
        self.option_chain_table.setRowCount(len(data))

        for row_idx, (_, row) in enumerate(data.iterrows()):
            # Strike price
            strike_item = QTableWidgetItem(str(row["Strike"]))
            if row["Strike"] == atm_strike:
                strike_item.setBackground(QColor(255, 255, 0))  # Yellow for ATM

            self.option_chain_table.setItem(row_idx, 0, strike_item)

            # CE data
            self.option_chain_table.setItem(
                row_idx, 1, QTableWidgetItem(str(row["CE_symbol"]))
            )
            self.option_chain_table.setItem(
                row_idx, 2, QTableWidgetItem(f"{row['CE_ltp']:.2f}")
            )
            self.option_chain_table.setItem(
                row_idx, 3, QTableWidgetItem(str(row["CE_Vol"]))
            )
            self.option_chain_table.setItem(
                row_idx, 4, QTableWidgetItem(str(row["CE_OI"]))
            )

            # PE data
            self.option_chain_table.setItem(
                row_idx, 5, QTableWidgetItem(str(row["PE_symbol"]))
            )
            self.option_chain_table.setItem(
                row_idx, 6, QTableWidgetItem(f"{row['PE_ltp']:.2f}")
            )
            self.option_chain_table.setItem(
                row_idx, 7, QTableWidgetItem(str(row["PE_Vol"]))
            )
            self.option_chain_table.setItem(
                row_idx, 8, QTableWidgetItem(str(row["PE_OI"]))
            )

            # Changes
            self.option_chain_table.setItem(
                row_idx, 9, QTableWidgetItem(f"{row['CE_LTPCh']:.2f}")
            )
            self.option_chain_table.setItem(
                row_idx, 10, QTableWidgetItem(f"{row['CE_LTPChP']:.2f}")
            )
            self.option_chain_table.setItem(
                row_idx, 11, QTableWidgetItem(f"{row['PE_LTPCh']:.2f}")
            )
            self.option_chain_table.setItem(
                row_idx, 12, QTableWidgetItem(f"{row['PE_LTPChP']:.2f}")
            )
            self.option_chain_table.setItem(
                row_idx, 13, QTableWidgetItem(str(row["CE_OICh"]))
            )
            self.option_chain_table.setItem(
                row_idx, 14, QTableWidgetItem(str(row["PE_OICh"]))
            )

    def update_signal_history_table(self):
        if not self.signal_history:
            return

        self.signal_history_table.setRowCount(len(self.signal_history))

        for row_idx, record in enumerate(reversed(self.signal_history)):
            self.signal_history_table.setItem(
                row_idx, 0, QTableWidgetItem(record["timestamp"])
            )

            signal_item = QTableWidgetItem(record["signal"])
            if record["signal"] == "BUY":
                signal_item.setForeground(QColor(0, 128, 0))  # Green
            elif record["signal"] == "SELL":
                signal_item.setForeground(QColor(255, 0, 0))  # Red
            self.signal_history_table.setItem(row_idx, 1, signal_item)

            self.signal_history_table.setItem(
                row_idx, 2, QTableWidgetItem(f"{record['underlying_ltp']:.2f}")
            )
            self.signal_history_table.setItem(
                row_idx, 3, QTableWidgetItem(str(record["expiry_date"]))
            )

    def update_paper_trading_tab(self):
        # Update PnL
        realized = self.paper_pnl["realized"]
        unrealized = self.paper_pnl["unrealized"]
        total = realized + unrealized

        self.realized_pnl_label.setText(f"💰 Realized PnL: ₹{realized:.2f}")
        self.unrealized_pnl_label.setText(f"📊 Unrealized PnL: ₹{unrealized:.2f}")
        self.total_pnl_label.setText(f"💵 Total PnL: ₹{total:.2f}")

        # Update positions table
        self.positions_table.setRowCount(len(self.paper_positions))
        for row_idx, (symbol, position) in enumerate(self.paper_positions.items()):
            if position["action"] == "BUY":
                pnl = (position["current_price"] - position["entry_price"]) * position[
                    "qty"
                ]
            else:
                pnl = (position["entry_price"] - position["current_price"]) * position[
                    "qty"
                ]

            self.positions_table.setItem(
                row_idx, 0, QTableWidgetItem(position["symbol"])
            )
            self.positions_table.setItem(
                row_idx, 1, QTableWidgetItem(position["action"])
            )
            self.positions_table.setItem(
                row_idx, 2, QTableWidgetItem(str(position["qty"]))
            )
            self.positions_table.setItem(
                row_idx, 3, QTableWidgetItem(f"₹{position['entry_price']:.2f}")
            )
            self.positions_table.setItem(
                row_idx, 4, QTableWidgetItem(f"₹{position['current_price']:.2f}")
            )
            self.positions_table.setItem(row_idx, 5, QTableWidgetItem(f"₹{pnl:.2f}"))
            self.positions_table.setItem(
                row_idx, 6, QTableWidgetItem(position["signal"])
            )

            # Color PnL
            pnl_item = self.positions_table.item(row_idx, 5)
            if pnl > 0:
                pnl_item.setForeground(QColor(0, 128, 0))  # Green
            elif pnl < 0:
                pnl_item.setForeground(QColor(255, 0, 0))  # Red

        # Update trade history
        self.trade_history_table.setRowCount(len(self.trade_history))
        for row_idx, trade in enumerate(reversed(self.trade_history)):
            self.trade_history_table.setItem(
                row_idx, 0, QTableWidgetItem(trade["timestamp"])
            )
            self.trade_history_table.setItem(
                row_idx, 1, QTableWidgetItem(trade["symbol"])
            )

            action_item = QTableWidgetItem(trade["action"])
            if trade["action"] == "BUY":
                action_item.setForeground(QColor(0, 128, 0))  # Green
            else:
                action_item.setForeground(QColor(255, 0, 0))  # Red
            self.trade_history_table.setItem(row_idx, 2, action_item)

            self.trade_history_table.setItem(
                row_idx, 3, QTableWidgetItem(str(trade["qty"]))
            )
            self.trade_history_table.setItem(
                row_idx, 4, QTableWidgetItem(f"₹{trade['price']:.2f}")
            )
            self.trade_history_table.setItem(
                row_idx, 5, QTableWidgetItem(trade["signal"])
            )
            self.trade_history_table.setItem(
                row_idx, 6, QTableWidgetItem(trade.get("type", "OPEN"))
            )

            if "pnl" in trade:
                pnl_item = QTableWidgetItem(f"₹{trade['pnl']:.2f}")
                if trade["pnl"] > 0:
                    pnl_item.setForeground(QColor(0, 128, 0))  # Green
                elif trade["pnl"] < 0:
                    pnl_item.setForeground(QColor(255, 0, 0))  # Red
                self.trade_history_table.setItem(row_idx, 7, pnl_item)

    @handle_errors
    def update_all_data(self, cid, token, sym, paper_trade_enabled):
        fy = fyersModel.FyersModel(
            client_id=cid, token=token, is_async=False, log_path=""
        )
        self.enforce_rate_limit()
        self.option_chain_api_count += 1

        # Get expiry data
        meta = fy.optionchain(data={"symbol": sym, "timestamp": ""})
        if meta.get("code") != 200:
            logging.error("Failed to fetch expiry data")
            return None, None, None

        # Process expiry dates
        exp = sorted(
            [(e["date"], e.get("expiry")) for e in meta["data"]["expiryData"]],
            key=lambda x: datetime.strptime(x[0], "%d-%m-%Y"),
        )
        today = datetime.today().date()
        dates = [datetime.strptime(d, "%d-%m-%Y").date() for d, _ in exp]
        idx = next((i for i, d in enumerate(dates) if d >= today), 0)
        curr_date, curr_ts = exp[idx]

        # Get underlying LTP
        ltp = self.get_underlying_ltp(cid, token, sym)

        # Get option chain data
        chain = self.get_option_chain_data(cid, token, sym, expiry_ts=curr_ts)

        # Update paper trading positions if enabled
        if paper_trade_enabled and self.paper_positions:
            self.update_position_prices(cid, token)
            self.update_unrealized_pnl(cid, token)

        return ltp, chain, curr_date

    def switch_page(self, page_name):
        self.current_page = page_name
        if page_name == "trading":
            self.content_area.setCurrentIndex(0)
        elif page_name == "signal_history":
            self.content_area.setCurrentIndex(1)
            self.update_signal_history_table()
        elif page_name == "paper_trading":
            self.content_area.setCurrentIndex(2)
            self.update_paper_trading_tab()

    def save_credentials(self):
        self.cid = self.cid_input.text()
        self.token = self.token_input.text()
        self.update_status("Credentials saved")

    def refresh_data(self):
        if not self.cid or not self.token:
            self.update_status("Please enter credentials first")
            return

        try:
            sym = self.sym_combo.currentText()
            ltp, chain, curr_date = self.update_all_data(
                self.cid, self.token, sym, self.paper_trade_check.isChecked()
            )

            if ltp:
                self.underlying_ltp_label.setText(f"Underlying LTP: {ltp}")

            if chain:
                signal, atm = self.format_and_show(
                    chain, f"Current Expiry: {curr_date}", ltp, show_signals=True
                )

                # Record signal history
                if signal and ltp:
                    timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    self.signal_history.append(
                        {
                            "timestamp": timestamp,
                            "signal": signal,
                            "underlying_ltp": ltp,
                            "expiry_date": curr_date,
                        }
                    )
                    self.update_signal_history_table()

                # Select strikes
                sell_pe, buy_pe, sell_ce, buy_ce = self.select_strikes_atm_half_price(
                    chain, atm
                )

                # Trading logic
                if signal in ("BUY", "SELL"):
                    if self.paper_trade_check.isChecked():
                        logging.info(
                            f"📢 Signal detected: {signal} - Attempting paper trade"
                        )
                        ids = self.handle_paper_trade(
                            signal, sell_pe, buy_pe, sell_ce, buy_ce
                        )
                        self.paper_last_signal_order_ids = ids
                        self.update_paper_trading_tab()
                    elif self.auto_trade_check.isChecked():
                        fy = fyersModel.FyersModel(
                            client_id=self.cid,
                            token=self.token,
                            is_async=False,
                            log_path="",
                        )
                        if (
                            self.live_last_signal == signal
                            and self.has_open_orders_for_last_signal(fy)
                        ):
                            self.update_status(f"Basket for {signal} already open.")
                        else:
                            ids = self.handle_basket_orders_atomic(
                                signal, sell_pe, buy_pe, sell_ce, buy_ce, fy
                            )
                            self.live_last_signal = signal
                            self.live_last_signal_order_ids = ids
                    else:
                        self.update_status(f"Auto trade disabled; signal: {signal}.")
                elif signal == "SIDEWAYS":
                    if self.paper_trade_check.isChecked():
                        self.handle_paper_trade(
                            signal, sell_pe, buy_pe, sell_ce, buy_ce
                        )
                        self.update_paper_trading_tab()
                    elif self.auto_trade_check.isChecked():
                        fy = fyersModel.FyersModel(
                            client_id=self.cid,
                            token=self.token,
                            is_async=False,
                            log_path="",
                        )
                        self.handle_basket_orders_atomic(
                            signal, sell_pe, buy_pe, sell_ce, buy_ce, fy
                        )

            # Update API usage
            self.oc_label.setText(f"Option Chain API: {self.option_chain_api_count}")
            self.qc_label.setText(f"Quote API: {self.quote_api_count}")
            self.total_label.setText(
                f"Total: {self.option_chain_api_count + self.quote_api_count}/{Config.API_RATE_LIMIT} per minute"
            )

            self.update_status("Data refreshed successfully")
        except Exception as e:
            self.update_status(f"Error refreshing data: {str(e)}")
            logging.error(f"Error in refresh_data: {str(e)}", exc_info=True)

    def format_and_show(self, chain, title, ltp, show_signals=False):
        df = pd.DataFrame(chain)
        if df.empty:
            self.update_status(f"No data for {title}")
            return None, None

        # Process CE and PE data
        ce = df[df["option_type"] == "CE"]
        pe = df[df["option_type"] == "PE"]

        # Define columns and rename them
        cols = [
            "symbol",
            "strike_price",
            "oi",
            "volume",
            "ltp",
            "ask",
            "bid",
            "ltpch",
            "ltpchp",
            "oich",
            "oichp",
            "prev_oi",
        ]

        ce_df = ce[cols].rename(
            columns={
                "strike_price": "Strike",
                "oi": "OI",
                "volume": "Vol",
                "ltpch": "LTPCh",
                "ltpchp": "LTPChP",
                "oich": "OICh",
                "oichp": "OIChP",
                "prev_oi": "PrevOI",
            }
        )

        pe_df = pe[cols].rename(
            columns={
                "strike_price": "Strike",
                "oi": "OI",
                "volume": "Vol",
                "ltpch": "LTPCh",
                "ltpchp": "LTPChP",
                "oich": "OICh",
                "oichp": "OIChP",
                "prev_oi": "PrevOI",
            }
        )

        # Merge CE and PE data
        merged = (
            pd.merge(
                ce_df.add_prefix("CE_"),
                pe_df.add_prefix("PE_"),
                left_on="CE_Strike",
                right_on="PE_Strike",
                how="outer",
            )
            .rename(columns={"CE_Strike": "Strike"})
            .drop("PE_Strike", axis=1)
        )

        # Find ATM strike
        atm = None
        if ltp is not None and not merged["Strike"].dropna().empty:
            atm = min(merged["Strike"].dropna(), key=lambda x: abs(x - ltp))

        # Update option chain table
        self.update_option_chain_table(merged, atm)

        if show_signals and atm is not None:
            signal, ratio = self.compute_signals(merged, atm, ltp)

            # Update signal display
            self.signal_label.setText(f"Current Signal: {signal}")
            if signal == "BUY":
                self.signal_label.setStyleSheet("color: green;")
            elif signal == "SELL":
                self.signal_label.setStyleSheet("color: red;")
            else:
                self.signal_label.setStyleSheet("color: orange;")

            return signal, atm
        return None, atm

    def export_signal_history(self):
        if not self.signal_history:
            QMessageBox.information(self, "Export", "No signal history to export")
            return

        try:
            df = pd.DataFrame(self.signal_history)
            filename = f"signal_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            QMessageBox.information(
                self, "Export", f"Signal history exported to {filename}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def export_trade_history(self):
        if not self.trade_history:
            QMessageBox.information(self, "Export", "No trade history to export")
            return

        try:
            df = pd.DataFrame(self.trade_history)
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            QMessageBox.information(
                self, "Export", f"Trade history exported to {filename}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def reset_paper_trading(self):
        reply = QMessageBox.question(
            self,
            "Reset Paper Trading",
            "Are you sure you want to reset all paper trading data?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.paper_trades = []
            self.paper_positions = {}
            self.paper_pnl = {"realized": 0.0, "unrealized": 0.0}
            self.trade_history = []
            self.last_signal = None
            self.update_paper_trading_tab()
            QMessageBox.information(self, "Reset", "Paper trading data has been reset")

    def update_status(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_bar.showMessage(f"[{timestamp}] {message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TradingApp()
    window.show()
    sys.exit(app.exec_())
