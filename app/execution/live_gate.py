"""Live trading safety gates – Phase 9.

Enforces strict, multi-layer gating before live capital execution:
1. Environment variable: LIVE_TRADING_ENABLED=true
2. CLI flag: --mode live
3. Operator confirmation: typing exact phrase at terminal startup
4. Capital cap: config.live.max_capital enforces maximum allocated equity
5. API keys: non-empty EXCHANGE_API_KEY and EXCHANGE_SECRET
6. Permission check: withdrawal permissions must be disabled (refuse startup if enabled;
   prominent notice if exchange does not expose permissions).
"""

from __future__ import annotations

import os
import sys
from typing import Any

from app.common.enums import TradingMode
from app.common.errors import TradingAgentError
from app.common.logging import get_logger
from app.config.settings import TradingConfig

logger = get_logger(__name__)

DEFAULT_CONFIRMATION_PHRASE = "I CONFIRM LIVE TRADING WITH REAL CAPITAL"


class LiveGatingError(TradingAgentError):
    """Raised when any required live trading gate fails."""


def print_security_warning_banner() -> None:
    """Print prominent operator notice when exchange doesn't expose permissions API."""
    banner = (
        "\n"
        "********************************************************************************\n"
        "                     CRITICAL LIVE SECURITY INSTRUCTION                         \n"
        "********************************************************************************\n"
        "The exchange API does not expose programmatic permission inspection.\n"
        "Before proceeding, you MUST manually verify in your exchange API settings:\n"
        "  1. 'Enable Withdrawals' is STRICTLY DISABLED for this API key.\n"
        "  2. IP Access Restriction / IP-whitelisting is TURNED ON.\n"
        "  3. API key only has Spot/Margin Trading permissions enabled.\n"
        "********************************************************************************\n"
    )
    print(banner, file=sys.stderr)
    logger.warning("Printed exchange permissions verification notice to operator.")


async def check_exchange_withdrawal_permissions(exchange: Any) -> bool | None:
    """Inspect exchange API permissions for withdrawals if supported by the exchange.

    Returns:
        True: withdrawals are enabled (SECURITY RISK)
        False: withdrawals are confirmed disabled (SAFE)
        None: exchange API does not expose permissions inspection
    """
    if exchange is None:
        return None

    # 1. CCXT standard fetch_permissions / fetch_api_key_permissions
    for meth_name in (
        "fetch_api_key_permissions",
        "fetch_permissions",
        "fetchApiKeyPermissions",
        "fetchPermissions",
    ):
        meth = getattr(exchange, meth_name, None)
        if callable(meth):
            try:
                res = await meth() if asyncio_iscoroutine(meth) else meth()
                if isinstance(res, dict):
                    # Check common withdrawal keys
                    if res.get("withdraw") is True or res.get("withdrawals") is True:
                        return True
                    if res.get("enableWithdrawals") is True:
                        return True
                    if (
                        res.get("withdraw") is False
                        or res.get("withdrawals") is False
                        or res.get("enableWithdrawals") is False
                    ):
                        return False
            except Exception:
                pass

    # 2. Binance-specific sapiGetAccountApiRestrictions
    sapi_meth = getattr(exchange, "sapiGetAccountApiRestrictions", None)
    if callable(sapi_meth):
        try:
            res = await sapi_meth() if asyncio_iscoroutine(sapi_meth) else sapi_meth()
            if isinstance(res, dict):
                if res.get("enableWithdrawals") is True:
                    return True
                if res.get("enableWithdrawals") is False:
                    return False
        except Exception:
            pass

    return None


def asyncio_iscoroutine(obj: Any) -> bool:
    """Helper to check if a callable or result is a coroutine."""
    import inspect

    return inspect.iscoroutinefunction(obj)


async def verify_live_trading_gates(
    mode: TradingMode | str,
    config: TradingConfig,
    api_key: str | None = None,
    secret: str | None = None,
    env_live_enabled: str | None = None,
    confirmation_input: str | None = None,
    exchange: Any | None = None,
    bypass_terminal_prompt: bool = False,
) -> None:
    """Verify that every live trading gate is satisfied before running live mode.

    All gates are required together:
    - mode must be live
    - LIVE_TRADING_ENABLED=true in environment
    - operator typed exact confirmation phrase
    - config.live.max_capital > 0
    - API keys present and non-empty
    - exchange withdrawal permissions confirmed absent (or prominent warning)
    """
    mode_str = mode.value if isinstance(mode, TradingMode) else str(mode)
    if mode_str != TradingMode.LIVE.value:
        return

    # Gate 1: Environment variable LIVE_TRADING_ENABLED=true
    env_val = env_live_enabled if env_live_enabled is not None else os.environ.get("LIVE_TRADING_ENABLED", "")
    if env_val.strip().lower() != "true":
        raise LiveGatingError(
            "Live gating failed: LIVE_TRADING_ENABLED environment variable is not 'true'."
        )

    # Gate 2: CLI flag / mode parameter is live (already confirmed by caller, validated above)

    # Gate 3: Confirmation phrase typed at startup
    expected_phrase = getattr(config.live, "exact_confirmation_phrase", DEFAULT_CONFIRMATION_PHRASE)
    if confirmation_input is None and not bypass_terminal_prompt:
        try:
            prompt_msg = (
                f"\n[LIVE MODE INITIALIZATION]\n"
                f"Type exactly '{expected_phrase}' to authorize live trading: "
            )
            confirmation_input = input(prompt_msg).strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise LiveGatingError("Live gating failed: Terminal input aborted by operator.") from exc

    if (confirmation_input or "").strip() != expected_phrase:
        raise LiveGatingError(
            f"Live gating failed: Confirmation phrase mismatch. Expected '{expected_phrase}', "
            f"got '{(confirmation_input or '').strip()}'. Live trading aborted."
        )

    # Gate 4: config.live.max_capital caps equity
    max_capital = getattr(config.live, "max_capital", 0.0)
    if max_capital <= 0:
        raise LiveGatingError(
            f"Live gating failed: config.live.max_capital must be > 0, got {max_capital}."
        )

    # Gate 5: API keys present and non-empty
    key = (api_key or os.environ.get("EXCHANGE_API_KEY", "")).strip()
    sec = (secret or os.environ.get("EXCHANGE_SECRET", "")).strip()
    if not key or not sec:
        raise LiveGatingError(
            "Live gating failed: Missing exchange API credentials. "
            "Both EXCHANGE_API_KEY and EXCHANGE_SECRET must be set and non-empty."
        )

    # Gate 6: Withdrawal permissions check
    withdrawals_enabled = await check_exchange_withdrawal_permissions(exchange)
    if withdrawals_enabled is True:
        raise LiveGatingError(
            "FATAL SECURITY RISK: Exchange reports withdrawal permissions enabled for this API key! "
            "Live trading refuses to start until withdrawal permissions are strictly disabled."
        )
    if withdrawals_enabled is None:
        print_security_warning_banner()

    logger.info(
        "All live trading gates successfully verified. Authorized live capital cap: $%.2f",
        max_capital,
    )
