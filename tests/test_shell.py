"""Unit tests for the interactive GPU shell module."""
import json
import pytest
from unittest import mock
from compute_pool.shell import launch_gpu_shell, launch_cluster_shell, _display_cluster_panel, _display_single_shell_panel, stop_gpu_shell


class TestShellModule:
    def test_missing_credentials_raises(self, monkeypatch):
        monkeypatch.setattr("compute_pool.shell.load_credentials", lambda slot: None)
        with pytest.raises(ValueError, match="No credentials configured"):
            launch_gpu_shell(slot=1)

    def test_successful_single_shell_launch(self, monkeypatch):
        monkeypatch.setattr(
            "compute_pool.shell.load_credentials",
            lambda slot: {"username": "testuser", "key": "testkey"}
        )
        monkeypatch.setattr(
            "compute_pool.shell._push_kernel_payload",
            lambda slot, slug, script: {"slot": slot, "username": "testuser", "status": "QUEUED"}
        )
        monkeypatch.setattr("webbrowser.open", mock.MagicMock())

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "https://test-node-shell.trycloudflare.com\n"
        monkeypatch.setattr("httpx.get", lambda url, timeout: mock_resp)

        res = launch_gpu_shell(slot=1, duration_minutes=30, open_web=False, timeout_seconds=5)
        assert res["web"] == "https://test-node-shell.trycloudflare.com"
        assert res["kernel_ref"] == "testuser/interactive-gpu-terminal-s1"

    def test_successful_cluster_shell_launch(self, monkeypatch):
        monkeypatch.setattr(
            "compute_pool.shell.load_credentials",
            lambda slot: {"username": f"user{slot}", "key": "key"}
        )
        monkeypatch.setattr(
            "compute_pool.shell._push_kernel_payload",
            lambda slot, slug, script: {"slot": slot, "username": f"user{slot}", "status": "QUEUED"}
        )
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "https://master-cluster.trycloudflare.com\n"
        monkeypatch.setattr("httpx.get", lambda url, timeout: mock_resp)
        monkeypatch.setattr("webbrowser.open", mock.MagicMock())

        res = launch_cluster_shell(duration_minutes=30, open_web=False, timeout_seconds=5)
        assert res["web"] == "https://master-cluster.trycloudflare.com"
        assert res["master_ref"] == "user1/interactive-gpu-terminal-s1"
        assert res["worker_ref"] == "user2/interactive-gpu-terminal-s2"

    def test_display_panels_do_not_crash(self):
        _display_cluster_panel(
            master_user="user1",
            worker_user="user2",
            web_url="https://master.trycloudflare.com",
            duration_minutes=60,
        )
        _display_single_shell_panel(
            slot=1,
            username="user1",
            web_url="https://test.trycloudflare.com",
            duration_minutes=60,
        )

    def test_stop_gpu_shell(self, monkeypatch):
        monkeypatch.setattr(
            "compute_pool.shell.load_credentials",
            lambda slot: {"username": f"user{slot}", "key": "key"} if slot == 1 else None
        )
        mock_api = mock.MagicMock()
        monkeypatch.setattr("compute_pool.shell._get_authenticated_api", lambda u, k: mock_api)

        stop_gpu_shell(slot=1)
        assert mock_api.kernels_push.call_count >= 1



