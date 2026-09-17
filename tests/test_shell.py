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
        monkeypatch.setattr(
            "compute_pool.shell._launch_single_slot_proc",
            lambda slot, dur, s_id: {"slot": slot, "username": "testuser", "status": "QUEUED"}
        )
        monkeypatch.setattr("webbrowser.open", mock.MagicMock())

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "https://test-node-shell.trycloudflare.com\n"
        monkeypatch.setattr("httpx.get", lambda url, timeout: mock_resp)

        res = launch_gpu_shell(slot=1, duration_minutes=30, open_web=False, timeout_seconds=5)
        assert res["web"] == "https://test-node-shell.trycloudflare.com"
        assert res["kernel_ref"] == "testuser/interactive-gpu-terminal-s1"

    def test_display_panel_does_not_crash(self):
        _display_shell_panel(
            slot=1,
            username="testuser",
            web_url="https://test.trycloudflare.com",
            duration_minutes=60,
        )

    def test_stop_gpu_shell(self, monkeypatch):
        from compute_pool.shell import stop_gpu_shell

        monkeypatch.setattr(
            "compute_pool.shell.load_credentials",
            lambda slot: {"username": f"user{slot}", "key": "key"} if slot == 1 else None
        )
        mock_api = mock.MagicMock()
        monkeypatch.setattr("compute_pool.shell._get_authenticated_api", lambda u, k: mock_api)

        stop_gpu_shell(slot=1)
        assert mock_api.kernels_push.call_count >= 1

    def test_launch_dual_gpu_shells(self, monkeypatch):
        from compute_pool.shell import launch_dual_gpu_shells

        monkeypatch.setattr(
            "compute_pool.shell.load_credentials",
            lambda slot: {"username": f"user{slot}", "key": "key"}
        )
        monkeypatch.setattr(
            "compute_pool.shell._launch_single_slot_proc",
            lambda slot, dur, s_id: {"slot": slot, "username": f"user{slot}", "status": "QUEUED"}
        )
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "https://dual-test.trycloudflare.com\n"
        monkeypatch.setattr("httpx.get", lambda url, timeout: mock_resp)
        monkeypatch.setattr("webbrowser.open", mock.MagicMock())

        res = launch_dual_gpu_shells(duration_minutes=30, open_web=False, timeout_seconds=5)
        assert "node0" in res
        assert "node1" in res
        assert res["node0"]["web"] == "https://dual-test.trycloudflare.com"


