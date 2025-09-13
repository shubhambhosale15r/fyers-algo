import threading
from collections import deque
import pandas as pd

class AppState:
    def __init__(self):
        self._lock = threading.RLock()
        self._is_running = False
        self._logs = deque(maxlen=100)
        self.ltp = None
        self.atm = None
        self.signal = None
        self.symbol = None
        self.last_updated = None
        self.signal_history = deque(maxlen=20)
        self.order_history = deque(maxlen=20)
        self.option_chain_df = pd.DataFrame()

    def start(self, symbol: str):
        with self._lock:
            self._is_running = True
            self.symbol = symbol
            # Reset histories for a new session
            self.signal_history.clear()
            self.order_history.clear()
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

    def update_live_data(self, ltp, atm, signal, option_chain_data, timestamp):
        with self._lock:
            self.ltp = ltp
            self.atm = atm
            self.signal = signal
            self.option_chain_df = pd.DataFrame(option_chain_data)
            self.last_updated = timestamp
            if signal:
                self.signal_history.appendleft({"timestamp": timestamp, "signal": signal, "ltp": ltp})

    def add_order(self, order_details):
        with self._lock:
            self.order_history.appendleft(order_details)

    def get_status(self):
        with self._lock:
            return {
                "is_running": self._is_running,
                "ltp": self.ltp,
                "atm": self.atm,
                "signal": self.signal,
                "symbol": self.symbol,
                "last_updated": self.last_updated,
                "logs": list(self._logs),
                "signal_history": list(self.signal_history),
                "order_history": list(self.order_history),
                "option_chain": self.option_chain_df.to_dict('records')
            }
