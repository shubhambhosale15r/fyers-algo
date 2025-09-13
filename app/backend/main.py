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
# These objects are created once and shared across all API requests.
app_state = AppState()
session_state = SessionState()

# --- API Models for Request/Response ---
class Settings(BaseModel):
    lot_size: int
    client_id: str
    access_token: str

# --- API Endpoints ---

@app.on_event("startup")
def on_startup():
    """
    Load credentials from the JSON file when the application starts.
    This assumes the server is run from the project's root directory.
    """
    try:
        load_credentials(session_state)
        app_state.add_log("Credentials loaded on startup.")
    except Exception as e:
        app_state.add_log(f"Could not load credentials on startup: {e}")

@app.post("/api/start", summary="Start the Trading Bot")
def start_trading(background_tasks: BackgroundTasks):
    """
    Starts the automated trading loop as a background task.
    """
    if app_state.is_running():
        raise HTTPException(status_code=400, detail="Trading is already running.")

    if not session_state.cid or not session_state.token:
        raise HTTPException(status_code=400, detail="Credentials are not set. Please set them via POST /api/settings.")

    app_state.start()
    # Use FastAPI's BackgroundTasks to run the loop without blocking.
    background_tasks.add_task(trading_loop, session_state, app_state)

    return {"message": "Trading started successfully."}


@app.post("/api/stop", summary="Stop the Trading Bot")
def stop_trading():
    """
    Stops the automated trading loop.
    """
    if not app_state.is_running():
        raise HTTPException(status_code=400, detail="Trading is not running.")

    app_state.stop()
    return {"message": "Trading will stop after the current cycle."}


@app.get("/api/status", summary="Get Live Status")
def get_status():
    """
    Retrieves the current status of the trading bot, including logs and live data.
    """
    return app_state.get_status()


@app.get("/api/settings", response_model=Settings, summary="Get Current Settings")
def get_settings():
    """
    Retrieves the current trading settings.
    """
    return {
        "lot_size": Config.LOT_SIZE,
        "client_id": session_state.cid,
        "access_token": session_state.token
    }


@app.post("/api/settings", summary="Update Settings")
def update_settings(settings: Settings):
    """
    Updates the trading settings and saves credentials to the JSON file.
    """
    Config.LOT_SIZE = settings.lot_size
    session_state.cid = settings.client_id
    session_state.token = settings.access_token

    # Save the updated credentials back to the file.
    try:
        # Assumes the server is run from the project root directory.
        with open('fyers_credentials.json', 'w') as f:
            json.dump({'client_id': settings.client_id, 'access_token': settings.access_token}, f)
        app_state.add_log("Credentials saved to fyers_credentials.json")
    except Exception as e:
        app_state.add_log(f"Error saving credentials: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving credentials: {e}")

    app_state.add_log(f"Settings updated: Lot Size={Config.LOT_SIZE}")
    return {"message": "Settings updated successfully."}
