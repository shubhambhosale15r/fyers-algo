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

# --- Backend API Configuration ---
API_URL = "http://127.0.0.1:8000"

# --- Helper Functions to Interact with Backend ---
def get_status():
    try:
        response = requests.get(f"{API_URL}/api/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None

def get_settings():
    try:
        response = requests.get(f"{API_URL}/api/settings")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching settings: {e}")
        return None

def post_settings(settings):
    try:
        response = requests.post(f"{API_URL}/api/settings", json=settings)
        response.raise_for_status()
        st.success("Settings updated successfully!")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error updating settings: {e}")
        return None

def start_bot():
    try:
        response = requests.post(f"{API_URL}/api/start")
        response.raise_for_status()
        st.success("Bot started successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error starting bot: {e.response.json().get('detail', str(e))}")

def stop_bot():
    try:
        response = requests.post(f"{API_URL}/api/stop")
        response.raise_for_status()
        st.warning("Bot stopped.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error stopping bot: {e.response.json().get('detail', str(e))}")


# --- UI Layout ---

# Auto-refresh the page every 5 seconds
st_autorefresh(interval=5 * 1000, key="datarefresh")

st.title("📈 Fyers Algo-Trading Dashboard")

# --- Settings Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    settings = get_settings()
    if settings:
        client_id = st.text_input("Client ID", value=settings.get("client_id", ""), type="password")
        access_token = st.text_input("Access Token", value=settings.get("access_token", ""), type="password")
        lot_size = st.number_input("Lot Size", value=settings.get("lot_size", 75), min_value=1)

        if st.button("Save Settings"):
            new_settings = {
                "client_id": client_id,
                "access_token": access_token,
                "lot_size": lot_size,
            }
            post_settings(new_settings)
    else:
        st.warning("Could not load settings from backend.")

# --- Main Content ---
status_data = get_status()

if status_data:
    is_running = status_data.get("is_running", False)
    status_text = "Running" if is_running else "Stopped"
    status_color = "green" if is_running else "red"

    st.markdown(f"## Bot Status: <font color='{status_color}'>{status_text}</font>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Trading", disabled=is_running):
            start_bot()
            time.sleep(1) # Give a moment for the status to update
            st.rerun()

    with col2:
        if st.button("🛑 Stop Trading", disabled=not is_running):
            stop_bot()
            time.sleep(1)
            st.rerun()

    st.markdown("---")

    # --- Live Data Dashboard ---
    st.subheader("📊 Live Data")
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

    # --- Option Chain ---
    st.subheader("⛓️ Option Chain Data")
    option_chain_data = status_data.get("option_chain", [])
    if option_chain_data:
        df = pd.DataFrame(option_chain_data)
        st.dataframe(df, height=300)
    else:
        st.info("No option chain data available yet.")

else:
    st.error("Could not connect to the backend. Please ensure the backend is running.")
