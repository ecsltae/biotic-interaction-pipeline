"""
Configuration management for biotic-pipeline.

Reads config.toml from the working directory (or a path passed explicitly).
"""

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8003
    workers: int = 1
    log_level: str = "info"


class ModelConfig(BaseModel):
    model_dir: str = ""
    device: str = "auto"
    threshold: float = 0.25
    max_length: int = 256


class DataConfig(BaseModel):
    interaction_dict: str = "data/interaction_dict.csv"
    species_dict: str = "data/species_dict.csv"


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()


_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Load and return the configuration singleton."""
    global _config
    if _config is None:
        path = Path(config_path) if config_path else Path("config.toml")
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        _config = Config(**raw)
    return _config
