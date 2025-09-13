import streamlit as st
import requests
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh

# --- Page Configuration ---
st.set_page_config(
    page_title="Fyers Algo-Trading UI",
    page_icon="📈",
    layout="wide",
)

# --- App Constants ---
API_URL = "http://127.0.0.1:8000"
SYMBOLS = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX"
}

# --- Helper Functions to Interact with Backend ---

def get_status():
    try:
        r = requests.get(f"{API_URL}/api/status")
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None # Error is handled in the main UI

def get_settings():
    try:
        r = requests.get(f"{API_URL}/api/settings")
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None # Error is handled in the main UI

def post_settings(settings):
    try:
        r = requests.post(f"{API_URL}/api/settings", json=settings)
        r.raise_for_status()
        st.success("Settings updated successfully!")
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error updating settings: {e}")
        return None

def start_bot(symbol: str):
    try:
        r = requests.post(f"{API_URL}/api/start", json={"symbol": symbol})
        r.raise_for_status()
        st.success(r.json().get("message", "Bot started successfully!"))
    except requests.exceptions.RequestException as e:
        st.error(f"Error starting bot: {e.response.json().get('detail', str(e))}")

def stop_bot():
    try:
        r = requests.post(f"{API_URL}/api/stop")
        r.raise_for_status()
        st.warning("Bot stopped.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error stopping bot: {e.response.json().get('detail', str(e))}")


# --- UI Layout ---

st.title("📈 Fyers Algo-Trading Dashboard")

# --- Settings Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    # --- UI Refresh Settings ---
    st.subheader("UI Settings")
    ui_refresh_interval = st.number_input("UI Refresh Interval (seconds)", min_value=5, max_value=60, value=5, step=1)

    # --- Trading Settings ---
    st.subheader("Trading Settings")
    settings = get_settings()
    if settings:
        # Symbol Selection
        symbol_key = st.selectbox("Select Underlying", options=list(SYMBOLS.keys()), index=0)

        # Other Settings
        client_id = st.text_input("Client ID", value=settings.get("client_id", ""), type="password")
        access_token = st.text_input("Access Token", value=settings.get("access_token", ""), type="password")
        lot_size = st.number_input("Lot Size", value=settings.get("lot_size", 75), min_value=1)
        backend_refresh_interval = st.number_input("Backend Refresh Interval (seconds)", value=settings.get("refresh_interval", 30), min_value=5)

        if st.button("Save Settings"):
            new_settings = {
                "client_id": client_id,
                "access_token": access_token,
                "lot_size": lot_size,
                "refresh_interval": backend_refresh_interval
            }
            post_settings(new_settings)
    else:
        st.warning("Could not load settings from backend.")
        symbol_key = "NIFTY" # Default

# Auto-refresh the page based on user setting
st_autorefresh(interval=ui_refresh_interval * 1000, key="datarefresh")


# --- Main Content ---
status_data = get_status()

if status_data:
    is_running = status_data.get("is_running", False)
    active_symbol = status_data.get("symbol", "N/A")
    status_text = f"Running for {active_symbol}" if is_running else "Stopped"
    status_color = "green" if is_running else "red"

    st.markdown(f"## Bot Status: <font color='{status_color}'>{status_text}</font>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Trading", disabled=is_running):
            start_bot(SYMBOLS[symbol_key])
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("🛑 Stop Trading", disabled=not is_running):
            stop_bot()
            time.sleep(1)
            st.rerun()

    st.markdown("---")

    # --- Live Data Dashboard ---
    st.subheader("📊 Live Data")
    last_updated = status_data.get("last_updated", "N/A")
    st.caption(f"Last updated: {last_updated}")

    ltp = status_data.get('ltp', 'N/A')
    atm = status_data.get('atm', 'N/A')
    signal = status_data.get('signal', 'N/A')

    col1, col2, col3 = st.columns(3)
    col1.metric("Underlying LTP", ltp)
    col2.metric("ATM Strike", atm)
    col3.metric("Signal", signal)

    # --- Logs ---
    st.subheader("📜 Logs")
    logs = status_data.get("logs", [])
    st.text_area("Live Logs", value="\n".join(logs), height=200, disabled=True)

    # --- History Sections ---
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔔 Signal History")
        signal_history = status_data.get("signal_history", [])
        if signal_history:
            st.dataframe(pd.DataFrame(signal_history), height=200)
        else:
            st.info("No signals generated yet.")

    with col2:
        st.subheader("🛒 Order History")
        order_history = status_data.get("order_history", [])
        if order_history:
            st.dataframe(pd.DataFrame(order_history), height=200)
        else:
            st.info("No orders placed yet.")

    # --- Option Chain ---
    st.markdown("---")
    st.subheader("⛓️ Option Chain Data")
    option_chain_data = status_data.get("option_chain", [])
    if option_chain_data:
        df = pd.DataFrame(option_chain_data)
        st.dataframe(df, height=300)
    else:
        st.info("No option chain data available yet.")

else:
    st.error("Could not connect to the backend. Please ensure the backend server is running.")
