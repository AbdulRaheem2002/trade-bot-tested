"""Load and validate configuration from config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.common.errors import ConfigError
from app.config.settings import TradingConfig

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(path: Path | str | None = None) -> TradingConfig:
    """Read *path* (defaults to project-root ``config.yaml``), parse YAML,
    and return a validated :class:`TradingConfig`.

    Raises :class:`ConfigError` with a descriptive message on any problem.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in {config_path}: {exc}") from exc

    if raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raise ConfigError(f"Config in {config_path} must be a mapping (key-value dict)")

    import os

    # Environment variable overrides
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        raw["telegram_bot_token"] = os.environ["TELEGRAM_BOT_TOKEN"]
    if os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS"):
        raw_ids = os.environ["TELEGRAM_ALLOWED_CHAT_IDS"].split(",")
        raw["telegram_allowed_chat_ids"] = [
            int(x.strip()) if x.strip().lstrip("-").isdigit() else x.strip()
            for x in raw_ids
            if x.strip()
        ]
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        raw["discord_webhook_url"] = os.environ["DISCORD_WEBHOOK_URL"]
    if os.environ.get("DASHBOARD_API_TOKEN"):
        raw["dashboard_api_token"] = os.environ["DASHBOARD_API_TOKEN"]
    if os.environ.get("DASHBOARD_HOST"):
        raw["dashboard_host"] = os.environ["DASHBOARD_HOST"]
    if os.environ.get("DASHBOARD_PORT"):
        raw["dashboard_port"] = int(os.environ["DASHBOARD_PORT"])

    try:
        return TradingConfig(**raw)
    except Exception as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
