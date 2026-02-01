"""
Config version registry - Database-backed training config versioning system.
Supports versioning, linking models to configs, and fallback to TrainingConfig defaults.
"""
import os
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import inspect

logger = logging.getLogger(__name__)

# Import TrainingConfig for defaults
from config import TrainingConfig


def _default_db_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "data", "model_registry.db")


def _get_conn(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def _init_database(db_path: str) -> None:
    """Ensure config tables exist."""
    from model_registry import _init_database as init_model_registry
    init_model_registry(db_path)  # This will create all tables including config tables


def allocate_config_version_id(db_path: Optional[str] = None) -> str:
    """
    Allocate a new config version ID: config-YYYY-MM-dd-N where N is per-day auto-increment.
    
    Args:
        db_path: Registry DB path (default: data/model_registry.db).
    
    Returns:
        Config version ID string, e.g. "config-2025-01-31-1".
    """
    db_path = db_path or _default_db_path()
    _init_database(db_path)
    today = datetime.now().strftime("%Y-%m-%d")
    prefix = f"config-{today}-"
    
    # Get existing config version IDs for today
    indexes = []
    if os.path.isfile(db_path):
        with _get_conn(db_path) as conn:
            cur = conn.execute(
                "SELECT config_version_id FROM training_configs WHERE config_version_id LIKE ?",
                (f"{prefix}%",)
            )
            for row in cur.fetchall():
                version_id = row[0]
                try:
                    idx_part = version_id[len(prefix):]
                    if idx_part.isdigit():
                        indexes.append(int(idx_part))
                except ValueError:
                    continue
    
    next_index = max(indexes, default=0) + 1
    config_version_id = f"{prefix}{next_index}"
    return config_version_id


def config_object_to_dict(config_obj: Any) -> Dict[str, Any]:
    """
    Convert TrainingConfig object (or any object) to flattened dict.
    Handles nested structures like MODEL_CONFIGS by flattening with dot notation.
    
    Args:
        config_obj: TrainingConfig instance or class
    
    Returns:
        Flattened dict with dot-notation keys (e.g., "MODEL_CONFIGS.5m.sequence_length")
    """
    result = {}
    
    # Get all attributes that are not methods or private
    for attr_name in dir(config_obj):
        if attr_name.startswith('_') or callable(getattr(config_obj, attr_name, None)):
            continue
        
        try:
            value = getattr(config_obj, attr_name)
            
            # Skip None values
            if value is None:
                continue
            
            # Handle nested dicts (like MODEL_CONFIGS)
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        # Double nesting (e.g., MODEL_CONFIGS.5m.sequence_length)
                        for inner_key, inner_value in nested_value.items():
                            key = f"{attr_name}.{nested_key}.{inner_key}"
                            result[key] = json.dumps(inner_value)
                    else:
                        # Single nesting
                        key = f"{attr_name}.{nested_key}"
                        result[key] = json.dumps(nested_value)
            elif isinstance(value, (list, tuple)):
                # Serialize lists/tuples as JSON
                result[attr_name] = json.dumps(value)
            else:
                # Primitive types: convert to string for storage
                result[attr_name] = json.dumps(value)
        except Exception as e:
            logger.warning(f"Failed to serialize config attribute {attr_name}: {e}")
            continue
    
    return result


def config_dict_to_object(config_dict: Dict[str, Any]) -> Any:
    """
    Convert flattened config dict back to TrainingConfig-like object.
    Reconstructs nested structures from dot-notation keys.
    
    Args:
        config_dict: Flattened dict with dot-notation keys
    
    Returns:
        Object with attributes matching TrainingConfig structure
    """
    class ConfigObject:
        """Dynamic config object that mimics TrainingConfig."""
        pass
    
    obj = ConfigObject()
    
    # Track nested structures
    nested_dicts = {}
    
    for key, value_json in config_dict.items():
        try:
            value = json.loads(value_json)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, try to parse as primitive
            try:
                # Try to infer type
                if value_json.lower() == 'true':
                    value = True
                elif value_json.lower() == 'false':
                    value = False
                elif value_json.replace('.', '', 1).isdigit():
                    value = float(value_json) if '.' in value_json else int(value_json)
                else:
                    value = value_json
            except:
                value = value_json
        
        # Handle dot notation (e.g., MODEL_CONFIGS.5m.sequence_length)
        if '.' in key:
            parts = key.split('.')
            if len(parts) == 2:
                # Single nesting: ATTR.subkey
                attr_name, subkey = parts
                if not hasattr(obj, attr_name):
                    setattr(obj, attr_name, {})
                getattr(obj, attr_name)[subkey] = value
            elif len(parts) == 3:
                # Double nesting: ATTR.key1.key2
                attr_name, key1, key2 = parts
                if not hasattr(obj, attr_name):
                    setattr(obj, attr_name, {})
                if key1 not in getattr(obj, attr_name):
                    getattr(obj, attr_name)[key1] = {}
                getattr(obj, attr_name)[key1][key2] = value
            else:
                # More nesting - store as-is for now
                setattr(obj, key.replace('.', '_'), value)
        else:
            # Top-level attribute
            setattr(obj, key, value)
    
    return obj


def get_default_config() -> Dict[str, Any]:
    """
    Get default config from TrainingConfig class.
    Reads all attributes and returns as flattened dict.
    
    Returns:
        Flattened config dict
    """
    return config_object_to_dict(TrainingConfig)


def init_from_config_file(description: str = "Initial config from TrainingConfig", db_path: Optional[str] = None) -> str:
    """
    Initialize/migrate: Import current TrainingConfig class values into database as a new version.
    Always creates a new config version, even if configs already exist.
    
    Args:
        description: Description for the config version
        db_path: Registry DB path
    
    Returns:
        Config version ID of the newly created version
    """
    db_path = db_path or _default_db_path()
    _init_database(db_path)
    
    # Always create new config version from TrainingConfig
    config_dict = get_default_config()
    config_version_id = allocate_config_version_id(db_path)
    
    now = datetime.utcnow().isoformat() + "Z"
    
    with _get_conn(db_path) as conn:
        # Insert config version record
        conn.execute(
            "INSERT INTO training_configs (config_version_id, created_at, description, is_active) VALUES (?, ?, ?, ?)",
            (config_version_id, now, description, 1)
        )
        
        # Insert config data
        for config_key, config_value in config_dict.items():
            conn.execute(
                "INSERT INTO training_config_data (config_version_id, config_key, config_value) VALUES (?, ?, ?)",
                (config_version_id, config_key, config_value)
            )
        
        conn.commit()
    
    logger.info(f"Initialized config from TrainingConfig: {config_version_id}")
    return config_version_id


def create_config_version(config_dict: Dict[str, Any], description: Optional[str] = None, db_path: Optional[str] = None) -> str:
    """
    Create a new config version from a dict.
    
    Args:
        config_dict: Config as dict (can be nested or flattened)
        description: Optional description
        db_path: Registry DB path
    
    Returns:
        Created config version ID
    """
    db_path = db_path or _default_db_path()
    _init_database(db_path)
    
    # Flatten config if needed (check if it has nested structures)
    flattened = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # Nested dict - flatten with dot notation
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict):
                    # Double nesting
                    for inner_key, inner_value in nested_value.items():
                        flat_key = f"{key}.{nested_key}.{inner_key}"
                        flattened[flat_key] = json.dumps(inner_value)
                else:
                    flat_key = f"{key}.{nested_key}"
                    flattened[flat_key] = json.dumps(nested_value)
        else:
            flattened[key] = json.dumps(value)
    
    config_version_id = allocate_config_version_id(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    
    with _get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO training_configs (config_version_id, created_at, description, is_active) VALUES (?, ?, ?, ?)",
            (config_version_id, now, description, 1)
        )
        
        for config_key, config_value in flattened.items():
            conn.execute(
                "INSERT INTO training_config_data (config_version_id, config_key, config_value) VALUES (?, ?, ?)",
                (config_version_id, config_key, config_value)
            )
        
        conn.commit()
    
    logger.info(f"Created config version: {config_version_id}")
    return config_version_id


def get_config_version(config_version_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve a config version as flattened dict.
    
    Args:
        config_version_id: Config version ID
        db_path: Registry DB path
    
    Returns:
        Flattened config dict, or None if not found
    """
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return None
    
    _init_database(db_path)
    
    with _get_conn(db_path) as conn:
        # Check if config exists and is active
        cur = conn.execute(
            "SELECT is_active FROM training_configs WHERE config_version_id = ?",
            (config_version_id,)
        )
        row = cur.fetchone()
        if row is None or row[0] == 0:
            return None
        
        # Get all config data
        cur = conn.execute(
            "SELECT config_key, config_value FROM training_config_data WHERE config_version_id = ?",
            (config_version_id,)
        )
        config_dict = {}
        for row in cur.fetchall():
            config_dict[row[0]] = row[1]
    
    return config_dict if config_dict else None


def get_config_or_default(config_version_id: Optional[str] = None, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get config from database or fallback to TrainingConfig defaults.
    
    Args:
        config_version_id: Config version ID (None = use defaults)
        db_path: Registry DB path
    
    Returns:
        Config dict (from DB or defaults)
    """
    if config_version_id is None:
        return get_default_config()
    
    config_dict = get_config_version(config_version_id, db_path)
    if config_dict is None:
        logger.warning(f"Config version {config_version_id} not found, using defaults")
        return get_default_config()
    
    return config_dict


def list_config_versions(include_inactive: bool = False, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all config versions.
    
    Args:
        include_inactive: Include inactive (deleted) configs
        db_path: Registry DB path
    
    Returns:
        List of config version info dicts
    """
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return []
    
    _init_database(db_path)
    
    query = "SELECT config_version_id, created_at, description, is_active FROM training_configs"
    if not include_inactive:
        query += " WHERE is_active = 1"
    query += " ORDER BY created_at DESC"
    
    result = []
    with _get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(query):
            # Count models using this config
            cur = conn.execute(
                "SELECT COUNT(*) FROM model_config_links WHERE config_version_id = ?",
                (row["config_version_id"],)
            )
            model_count = cur.fetchone()[0]
            
            result.append({
                "config_version_id": row["config_version_id"],
                "created_at": row["created_at"],
                "description": row["description"],
                "is_active": bool(row["is_active"]),
                "model_count": model_count,
            })
    
    return result


def update_config_version(config_version_id: str, updates_dict: Dict[str, Any], description: Optional[str] = None, db_path: Optional[str] = None) -> str:
    """
    Update a config version by creating a new version with updates.
    
    Args:
        config_version_id: Source config version ID
        updates_dict: Dict of updates (can be nested or flattened)
        description: Optional description for new version
        db_path: Registry DB path
    
    Returns:
        New config version ID
    """
    db_path = db_path or _default_db_path()
    _init_database(db_path)
    
    # Get existing config
    existing_config = get_config_version(config_version_id, db_path)
    if existing_config is None:
        raise ValueError(f"Config version {config_version_id} not found")
    
    # Merge updates
    # Flatten updates if needed
    flattened_updates = {}
    for key, value in updates_dict.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict):
                    for inner_key, inner_value in nested_value.items():
                        flat_key = f"{key}.{nested_key}.{inner_key}"
                        flattened_updates[flat_key] = json.dumps(inner_value)
                else:
                    flat_key = f"{key}.{nested_key}"
                    flattened_updates[flat_key] = json.dumps(nested_value)
        else:
            flattened_updates[key] = json.dumps(value)
    
    # Merge with existing
    merged_config = existing_config.copy()
    merged_config.update(flattened_updates)
    
    # Create new version
    return create_config_version(
        {k: json.loads(v) for k, v in merged_config.items()},
        description=description or f"Updated from {config_version_id}",
        db_path=db_path
    )


def delete_config_version(config_version_id: str, db_path: Optional[str] = None) -> bool:
    """
    Soft delete a config version (set is_active=0).
    
    Args:
        config_version_id: Config version ID
        db_path: Registry DB path
    
    Returns:
        True if deleted, False if not found
    """
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return False
    
    _init_database(db_path)
    
    with _get_conn(db_path) as conn:
        cur = conn.execute(
            "UPDATE training_configs SET is_active = 0 WHERE config_version_id = ?",
            (config_version_id,)
        )
        conn.commit()
        return cur.rowcount > 0


def link_model_to_config(
    model_version_id: str,
    config_version_id: str,
    symbol: str,
    timeframe: str,
    db_path: Optional[str] = None
) -> None:
    """
    Link a model version to a config version.
    
    Args:
        model_version_id: Model version ID
        config_version_id: Config version ID
        symbol: Trading pair symbol
        timeframe: Timeframe
        db_path: Registry DB path
    """
    db_path = db_path or _default_db_path()
    _init_database(db_path)
    
    now = datetime.utcnow().isoformat() + "Z"
    
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_config_links (model_version_id, config_version_id, symbol, timeframe, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_version_id, symbol, timeframe) DO UPDATE SET
                config_version_id = excluded.config_version_id,
                created_at = excluded.created_at
            """,
            (model_version_id, config_version_id, symbol, timeframe, now)
        )
        conn.commit()
    
    logger.debug(f"Linked model {model_version_id} ({symbol} {timeframe}) to config {config_version_id}")


def get_config_for_model(
    model_version_id: str,
    symbol: str,
    timeframe: str,
    db_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get config version used for a specific model.
    
    Args:
        model_version_id: Model version ID
        symbol: Trading pair symbol
        timeframe: Timeframe
        db_path: Registry DB path
    
    Returns:
        Config dict, or None if not linked
    """
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return None
    
    _init_database(db_path)
    
    with _get_conn(db_path) as conn:
        cur = conn.execute(
            "SELECT config_version_id FROM model_config_links WHERE model_version_id = ? AND symbol = ? AND timeframe = ?",
            (model_version_id, symbol, timeframe)
        )
        row = cur.fetchone()
        if row is None:
            return None
        
        return get_config_version(row[0], db_path)


def get_models_for_config(config_version_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all models trained with a specific config version.
    
    Args:
        config_version_id: Config version ID
        db_path: Registry DB path
    
    Returns:
        List of model info dicts
    """
    db_path = db_path or _default_db_path()
    if not os.path.isfile(db_path):
        return []
    
    _init_database(db_path)
    
    result = []
    with _get_conn(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT model_version_id, symbol, timeframe, created_at FROM model_config_links WHERE config_version_id = ? ORDER BY created_at DESC",
            (config_version_id,)
        )
        for row in cur.fetchall():
            result.append({
                "model_version_id": row["model_version_id"],
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "created_at": row["created_at"],
            })
    
    return result
