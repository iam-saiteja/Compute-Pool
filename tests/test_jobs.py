"""Unit tests for the job model and local storage."""
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from compute_pool.jobs.model import Job, JobSpec, JobState


class TestJobModel:
    def test_default_id_format(self):
        job = Job(id="job-0")
        assert job.id.startswith("job-")
        assert job.id == "job-0"

    def test_default_state_is_queued(self):
        job = Job()
        assert job.state == JobState.QUEUED

    def test_transition_changes_state(self):
        job = Job()
        job.transition(JobState.SCHEDULING)
        assert job.state == JobState.SCHEDULING

    def test_transition_stores_error(self):
        job = Job()
        job.transition(JobState.FAILED, error="no quota")
        assert job.state == JobState.FAILED
        assert job.error == "no quota"

    def test_to_dict_roundtrip(self):
        spec = JobSpec(name="test-job", script="print('hi')", gpu=True, gpu_memory_gb=16)
        job = Job(spec=spec)
        job.transition(JobState.ASSIGNED)
        job.assigned_slot = 1
        job.assigned_username = "alice"

        d = job.to_dict()
        restored = Job.from_dict(d)

        assert restored.id == job.id
        assert restored.state == JobState.ASSIGNED
        assert restored.assigned_slot == 1
        assert restored.assigned_username == "alice"
        assert restored.spec.name == "test-job"
        assert restored.spec.gpu is True

    def test_from_dict_handles_missing_fields(self):
        """from_dict should not crash on minimal dict."""
        d = {"id": "job-abc12345"}
        job = Job.from_dict(d)
        assert job.id == "job-abc12345"
        assert job.state == JobState.QUEUED


class TestLocalStorage:
    def test_upsert_and_load(self, tmp_path, monkeypatch):
        # Redirect DATA_DIR to tmp_path
        import compute_pool.storage.local as storage
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        job = Job(id="job-0", spec=JobSpec(name="store-test", script="pass"))
        storage.upsert_job(job)

        loaded = storage.load_all_jobs()
        assert len(loaded) == 1
        assert loaded[0].id == "job-0"
        assert loaded[0].spec.name == "store-test"

    def test_get_job_by_index_or_string(self, tmp_path, monkeypatch):
        import compute_pool.storage.local as storage
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        job = Job(id="job-0", spec=JobSpec(name="index-test", script="pass"))
        storage.upsert_job(job)

        assert storage.get_job(0) is not None
        assert storage.get_job("0") is not None
        assert storage.get_job("job-0") is not None
        assert storage.get_job("job-doesnotexist") is None

    def test_delete_and_clear_jobs(self, tmp_path, monkeypatch):
        import compute_pool.storage.local as storage
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        j0 = Job(id="job-0", spec=JobSpec(name="j0", script="pass"))
        j1 = Job(id="job-1", spec=JobSpec(name="j1", script="pass"))
        j1.transition(JobState.COMPLETED)
        storage.upsert_job(j0)
        storage.upsert_job(j1)

        # Delete single job by index
        deleted = storage.delete_job(0)
        assert deleted is not None
        assert deleted.id == "job-0"
        assert len(storage.load_all_jobs()) == 1

        # Clear finished jobs
        cleared = storage.clear_jobs(all_jobs=False)
        assert len(cleared) == 1
        assert len(storage.load_all_jobs()) == 0

    def test_reindex_jobs(self, tmp_path, monkeypatch):
        import compute_pool.storage.local as storage
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        j1 = Job(id="job-old-a", spec=JobSpec(name="a", script="pass"))
        j2 = Job(id="job-old-b", spec=JobSpec(name="b", script="pass"))
        storage.upsert_job(j1)
        storage.upsert_job(j2)

        reindexed = storage.reindex_jobs()
        assert len(reindexed) == 2
        assert reindexed[0].id == "0"
        assert reindexed[1].id == "1"

    def test_upsert_updates_existing(self, tmp_path, monkeypatch):
        import compute_pool.storage.local as storage
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        job = Job(id="job-0", spec=JobSpec(name="update-test", script="pass"))
        storage.upsert_job(job)

        job.transition(JobState.COMPLETED)
        storage.upsert_job(job)

        jobs = storage.load_all_jobs()
        assert len(jobs) == 1
        assert jobs[0].state == JobState.COMPLETED

    def test_stop_remote_job(self, tmp_path, monkeypatch):
        import compute_pool.storage.local as storage
        from compute_pool.jobs.runner import stop_remote_job
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        job = Job(id="job-0", spec=JobSpec(name="running-job", script="pass"))
        job.transition(JobState.RUNNING)
        job.assigned_slot = 1
        job.kaggle_kernel_slug = "test-slug"
        storage.upsert_job(job)

        mock_api = mock.MagicMock()
        monkeypatch.setattr("compute_pool.jobs.runner._get_api_for_slot", lambda s: (mock_api, "testuser"))

        cancelled_job = stop_remote_job(job.id)
        assert cancelled_job.state == JobState.FAILED
        assert cancelled_job.error == "Cancelled by user"
        mock_api.kernels_push.assert_called_once()

