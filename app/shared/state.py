import threading
from collections import deque
import pandas as pd

class AppState:
    def __init__(self):
        self._lock = threading.RLock()
        self._is_running = False
        self._logs = deque(maxlen=100)  # Store last 100 log messages
        self.ltp = None
        self.atm = None
        self.signal = None
        self.symbol = None  # Add symbol attribute
        self.option_chain_df = pd.DataFrame()

    def start(self, symbol: str):
        with self._lock:
            self._is_running = True
            self.symbol = symbol
            self.add_log(f"Trading loop started for {symbol}.")

    def stop(self):
        with self._lock:
            self._is_running = False
            self.add_log("Trading loop stopped.")

    def is_running(self):
        with self._lock:
            return self._is_running

    def add_log(self, message):
        with self._lock:
            self._logs.append(message)

    def get_logs(self):
        with self._lock:
            return list(self._logs)

    def update_live_data(self, ltp, atm, signal, option_chain_data):
        with self._lock:
            self.ltp = ltp
            self.atm = atm
            self.signal = signal
            self.option_chain_df = pd.DataFrame(option_chain_data)

    def get_status(self):
        with self._lock:
            return {
                "is_running": self._is_running,
                "ltp": self.ltp,
                "atm": self.atm,
                "signal": self.signal,
                "symbol": self.symbol,
                "logs": list(self._logs),
                "option_chain": self.option_chain_df.to_dict('records')
            }
