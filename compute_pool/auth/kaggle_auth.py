"""
Kaggle credential management for Compute Pool.

Each "slot" (1 or 2) maps to a separate Kaggle account.
Credentials are stored at:
    ~/.compute-pool/accounts/slot{N}/kaggle.json

Supports both credential formats:
  - Old format: {"username": "...", "key": "<hex-string>"}  → HTTP Basic auth
  - New format: {"username": "...", "key": "KGAT_..."}      → Bearer token auth
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

console = Console()

CREDS_ROOT = Path.home() / ".compute-pool" / "accounts"


def _creds_path(slot: int) -> Path:
    return CREDS_ROOT / f"slot{slot}" / "kaggle.json"


def _is_bearer_token(key: str) -> bool:
    """New Kaggle API tokens start with KGAT_ and use Bearer auth."""
    return key.startswith("KGAT_")


def store_credentials(slot: int, username: str, key: str) -> Path:
    """Persist Kaggle credentials for a slot."""
    path = _creds_path(slot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"username": username, "key": key}, indent=2))
    try:
        path.chmod(0o600)
    except NotImplementedError:
        pass  # Windows — skip
    return path


def load_credentials(slot: int) -> dict[str, str] | None:
    """Load stored credentials for a slot. Returns None if not configured."""
    path = _creds_path(slot)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if "username" in data and "key" in data:
            return data
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def interactive_login(slot: int) -> dict[str, str]:
    """
    Prompt the user for Kaggle credentials interactively.
    Directs them to https://www.kaggle.com/settings to get an API token.
    """
    console.print(f"\n[bold cyan]Compute Pool — Login (Slot {slot})[/bold cyan]")
    console.print("[dim]Get your API key from: https://www.kaggle.com/settings → API → Create New Token[/dim]\n")

    username = Prompt.ask(f"  Kaggle username for slot {slot}").strip()
    key      = Prompt.ask(f"  Kaggle API key for slot {slot}").strip()   # shown as plain text

    if not username or not key:
        raise ValueError("Username and API key must not be empty.")

    # Validate by attempting a lightweight API call
    _validate_credentials(username, key, slot)

    path = store_credentials(slot, username, key)
    console.print(f"\n[green]✓ Slot {slot} authenticated as '{username}'[/green]")
    console.print(f"  Credentials stored at: {path}\n")
    return {"username": username, "key": key}


def _validate_credentials(username: str, key: str, slot: int) -> None:
    """
    Validate Kaggle credentials.

    Kaggle has two token formats:
      - Legacy hex key  → HTTP Basic auth (username, key)
      - New KGAT_ token → Bearer token auth
    """
    import httpx

    bearer = _is_bearer_token(key)
    auth_desc = "Bearer token" if bearer else "Basic auth"
    console.print(f"  [dim]Verifying slot {slot} via {auth_desc}...[/dim]")

    headers = {}
    auth = None
    if bearer:
        headers["Authorization"] = f"Bearer {key}"
    else:
        auth = (username, key)

    try:
        resp = httpx.get(
            "https://www.kaggle.com/api/v1/competitions/list",
            headers=headers,
            auth=auth,
            timeout=15,
        )
        if resp.status_code == 401:
            raise RuntimeError(
                f"Invalid credentials for slot {slot}: 401 Unauthorized.\n"
                "Double-check your username and API key at https://www.kaggle.com/settings"
            )
        if resp.status_code not in (200, 204):
            console.print(
                f"  [yellow]Warning: unexpected status {resp.status_code} "
                f"— credentials saved anyway.[/yellow]"
            )
    except httpx.ConnectError:
        console.print(
            "  [yellow]Warning: could not reach kaggle.com "
            "— credentials saved but not verified.[/yellow]"
        )


def credentials_configured(slot: int) -> bool:
    return load_credentials(slot) is not None

