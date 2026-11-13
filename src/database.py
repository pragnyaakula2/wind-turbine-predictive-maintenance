"""
MongoDB Database Interface & Authentication Manager
Handles user authentication, password hashing, and prediction audit history.
Includes seamless in-memory fallback if MongoDB is not running.
"""

import atexit
import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGO_DB_NAME", "wind_turbine_db")
AUTH_SALT = b"wt_gpms_secure_salt_2026"

_client: Optional[Any] = None

# In-Memory Fallback Store (Used when MongoDB is offline)
FALLBACK_USERS = {
    "admin": {
        "username": "admin",
        "full_name": "Lead Maintenance Engineer",
        "role": "Lead Engineer",
        "password_hash": hashlib.sha256("adminpassword123".encode("utf-8") + AUTH_SALT).hexdigest(),
    },
    "operator": {
        "username": "operator",
        "full_name": "Turbine Field Operator",
        "role": "Field Operator",
        "password_hash": hashlib.sha256("operatorpass123".encode("utf-8") + AUTH_SALT).hexdigest(),
    },
}
FALLBACK_HISTORY: List[Dict[str, Any]] = []


def hash_password(password: str) -> str:
    """Computes SHA-256 hash of password with fixed system salt."""
    return hashlib.sha256(password.encode("utf-8") + AUTH_SALT).hexdigest()


def get_client():
    """Returns singleton MongoClient instance or None if unreachable."""
    global _client
    if not PYMONGO_AVAILABLE:
        return None
    if _client is not None:
        try:
            _client.admin.command("ping")
            return _client
        except Exception:
            _client = None
    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000)
        _client.admin.command("ping")
        return _client
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception):
        _client = None
        return None


def get_db():
    """Returns database connection or None if MongoDB is unreachable."""
    client = get_client()
    if client is not None:
        return client[DATABASE_NAME]
    return None


def close_client():
    """Closes the MongoClient connection on program exit."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


atexit.register(close_client)


def is_mongodb_connected() -> bool:
    """Returns True if MongoDB server responds to ping."""
    return get_db() is not None


def init_db() -> bool:
    """Seeds default user accounts into MongoDB if collection is empty."""
    db = get_db()
    if db is None:
        return False
    try:
        users_col = db["users"]
        if users_col.count_documents({}) == 0:
            seed_users = [
                {
                    "username": "admin",
                    "full_name": "Lead Maintenance Engineer",
                    "role": "Lead Engineer",
                    "password_hash": hash_password("adminpassword123"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "username": "operator",
                    "full_name": "Turbine Field Operator",
                    "role": "Field Operator",
                    "password_hash": hash_password("operatorpass123"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            ]
            users_col.insert_many(seed_users)
        return True
    except Exception:
        return False


def verify_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Verifies user credentials against MongoDB (or in-memory fallback).
    Returns user details dict on success, None on failure.
    """
    input_hash = hash_password(password)
    db = get_db()

    if db is not None:
        try:
            user = db["users"].find_one({"username": username})
            if user and user.get("password_hash") == input_hash:
                return {
                    "username": user["username"],
                    "full_name": user.get("full_name", username),
                    "role": user.get("role", "Engineer"),
                    "backend": "MongoDB",
                }
            return None
        except Exception:
            pass  # Fall back to in-memory check

    # In-memory fallback authentication
    fallback_user = FALLBACK_USERS.get(username)
    if fallback_user and fallback_user["password_hash"] == input_hash:
        return {
            "username": fallback_user["username"],
            "full_name": fallback_user["full_name"],
            "role": fallback_user["role"],
            "backend": "Local Fallback",
        }
    return None


def register_user(
    username: str,
    password: str,
    full_name: str = "",
    role: str = "Engineer",
) -> Tuple[bool, str]:
    """
    Registers a new user in MongoDB (and local fallback store).
    Returns (True, success_message) or (False, error_message).
    """
    clean_username = username.strip()
    if not clean_username or len(clean_username) < 3:
        return False, "Username must be at least 3 characters long."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."

    db = get_db()
    if db is not None:
        try:
            if db["users"].find_one({"username": clean_username}):
                return False, f"Username '{clean_username}' already exists. Please choose another."
            new_doc = {
                "username": clean_username,
                "full_name": full_name.strip() or clean_username,
                "role": role.strip() or "Engineer",
                "password_hash": hash_password(password),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            db["users"].insert_one(new_doc)
            FALLBACK_USERS[clean_username] = new_doc
            return True, "Account created successfully! You can now sign in."
        except Exception as e:
            return False, f"Registration error: {str(e)}"

    # In-memory fallback registration
    if clean_username in FALLBACK_USERS:
        return False, f"Username '{clean_username}' already exists."
    new_doc = {
        "username": clean_username,
        "full_name": full_name.strip() or clean_username,
        "role": role.strip() or "Engineer",
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    FALLBACK_USERS[clean_username] = new_doc
    return True, "Account registered successfully (local mode)! You can now sign in."


# Auto-seed database on import if connected
try:
    init_db()
except Exception:
    pass


def log_prediction(
    username: str,
    sensor_inputs: Dict[str, Any],
    predictions: Dict[str, Any],
    consensus: str,
) -> bool:
    """Logs a single real-time prediction event into MongoDB (or fallback)."""
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "operator": username,
        "sensor_inputs": sensor_inputs,
        "classical_ml": predictions.get("ml_status", "N/A"),
        "deep_learning": predictions.get("dl_status", "N/A"),
        "quantum_ml": predictions.get("qml_status", "N/A"),
        "consensus_verdict": consensus,
    }

    db = get_db()
    if db is not None:
        try:
            db["prediction_history"].insert_one(record)
            return True
        except Exception:
            pass

    # Save to in-memory fallback history
    FALLBACK_HISTORY.insert(0, record)
    if len(FALLBACK_HISTORY) > 50:
        FALLBACK_HISTORY.pop()
    return True


def get_prediction_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieves recent prediction logs from MongoDB or in-memory fallback."""
    db = get_db()
    if db is not None:
        try:
            cursor = db["prediction_history"].find(
                {}, {"_id": 0}
            ).sort("timestamp", pymongo.DESCENDING).limit(limit)
            return list(cursor)
        except Exception:
            pass
    return FALLBACK_HISTORY[:limit]
