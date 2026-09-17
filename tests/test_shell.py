"""Unit tests for the interactive GPU shell module."""
import json
import pytest
from unittest import mock
from compute_pool.shell import launch_gpu_shell, _display_shell_panel


class TestShellModule:
    def test_missing_credentials_raises(self, monkeypatch):
        monkeypatch.setattr("compute_pool.shell.load_credentials", lambda slot: None)
        with pytest.raises(ValueError, match="No credentials configured"):
            launch_gpu_shell(slot=1)

    def test_successful_shell_launch(self, monkeypatch):
        monkeypatch.setattr(
            "compute_pool.shell.load_credentials",
            lambda slot: {"username": "testuser", "key": "testkey"}
        )

        mock_api = mock.MagicMock()
        mock_api.kernels_push.return_value = {"error": None}
        mock_api.kernels_logs.return_value = [
            {"data": "WEB_TERMINAL: https://random-subdomain.trycloudflare.com\n"}
        ]
        monkeypatch.setattr("compute_pool.shell._get_authenticated_api", lambda u, k: mock_api)
        monkeypatch.setattr("webbrowser.open", mock.MagicMock())

        res = launch_gpu_shell(slot=1, duration_minutes=30, open_web=False, timeout_seconds=5)
        assert res["web"] == "https://random-subdomain.trycloudflare.com"
        assert res["kernel_ref"] == "testuser/interactive-gpu-terminal"
        mock_api.kernels_push.assert_called_once()

    def test_display_panel_does_not_crash(self):
        # Ensure display panel executes cleanly with mock data
        _display_shell_panel(
            slot=1,
            username="testuser",
            web_url="https://test.trycloudflare.com",
            duration_minutes=60,
        )
