import os
from compute_pool.cluster_pool import gpus, total_vram_gb, is_cluster_online


def test_cluster_pool_gpus():
    gpu_list = gpus()
    assert len(gpu_list) == 4
    assert gpu_list[0]["id"] == 0
    assert gpu_list[3]["id"] == 3
    assert total_vram_gb() == 60.0


def test_is_cluster_online(tmp_path, monkeypatch):
    assert not is_cluster_online()
    marker = tmp_path / ".cluster_worker_url"
    marker.write_text("https://fake.trycloudflare.com")
    monkeypatch.setattr("os.path.exists", lambda path: True if path == "/kaggle/working/.cluster_worker_url" else os.path.exists(path))
    assert is_cluster_online()
