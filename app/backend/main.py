import sys
import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Add the project root directory to the Python path to allow imports from app.shared
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.state import AppState
from shared.trading_logic import (
    trading_loop,
    SessionState,
    load_credentials,
    Config
)

# --- FastAPI App Initialization ---
app = FastAPI(title="Trading Bot API")

# --- Singleton State Management ---
app_state = AppState()
session_state = SessionState()

# --- API Models for Request/Response ---
class Settings(BaseModel):
    lot_size: int
    client_id: str
    access_token: str
    refresh_interval: int

class StartRequest(BaseModel):
    symbol: str

# --- API Endpoints ---

@app.on_event("startup")
def on_startup():
    """Load credentials from the JSON file when the application starts."""
    try:
        load_credentials(session_state)
        app_state.add_log("Credentials loaded on startup.")
    except Exception as e:
        app_state.add_log(f"Could not load credentials on startup: {e}")

@app.post("/api/start", summary="Start the Trading Bot")
def start_trading(start_request: StartRequest, background_tasks: BackgroundTasks):
    """Starts the automated trading loop for a given symbol."""
    if app_state.is_running():
        raise HTTPException(status_code=400, detail="Trading is already running.")

    if not session_state.cid or not session_state.token:
        raise HTTPException(status_code=400, detail="Credentials are not set. Please set them via POST /api/settings.")

    app_state.start(symbol=start_request.symbol)
    background_tasks.add_task(trading_loop, session_state, app_state, start_request.symbol)

    return {"message": f"Trading started successfully for {start_request.symbol}."}


@app.post("/api/stop", summary="Stop the Trading Bot")
def stop_trading():
    """Stops the automated trading loop."""
    if not app_state.is_running():
        raise HTTPException(status_code=400, detail="Trading is not running.")

    app_state.stop()
    return {"message": "Trading will stop after the current cycle."}


@app.get("/api/status", summary="Get Live Status")
def get_status():
    """Retrieves the current status of the trading bot."""
    return app_state.get_status()


@app.get("/api/settings", response_model=Settings, summary="Get Current Settings")
def get_settings():
    """Retrieves the current trading settings."""
    return {
        "lot_size": Config.LOT_SIZE,
        "client_id": session_state.cid,
        "access_token": session_state.token,
        "refresh_interval": Config.REFRESH_INTERVAL
    }


@app.post("/api/settings", summary="Update Settings")
def update_settings(settings: Settings):
    """Updates the trading settings and saves credentials."""
    Config.LOT_SIZE = settings.lot_size
    Config.REFRESH_INTERVAL = settings.refresh_interval
    session_state.cid = settings.client_id
    session_state.token = settings.access_token

    try:
        with open('fyers_credentials.json', 'w') as f:
            json.dump({'client_id': settings.client_id, 'access_token': settings.access_token}, f)
        app_state.add_log("Credentials saved to fyers_credentials.json")
    except Exception as e:
        app_state.add_log(f"Error saving credentials: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving credentials: {e}")

    app_state.add_log(f"Settings updated: Lot Size={Config.LOT_SIZE}, Refresh Interval={Config.REFRESH_INTERVAL}s")
    return {"message": "Settings updated successfully."}
