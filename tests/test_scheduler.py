"""Tests for the scheduler."""
from unittest import mock
from compute_pool.jobs.model import Job, JobSpec, JobState
from compute_pool.accounts.manager import AccountStatus


class TestScheduler:
    def _make_status(self, slot, username, connected, hours_left):
        return AccountStatus(
            slot=slot,
            username=username,
            connected=connected,
            estimated_gpu_hours_remaining=hours_left,
        )

    def test_assigns_to_slot_with_most_hours(self, tmp_path, monkeypatch):
        import compute_pool.storage.local as storage
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        statuses = [
            self._make_status(1, "alice", True, 10.0),
            self._make_status(2, "bob",   True, 25.0),
        ]

        import compute_pool.scheduler.simple as sched
        monkeypatch.setattr(sched, "get_all_statuses", lambda: statuses)

        job = Job(spec=JobSpec(name="sched-test", script="pass"))
        storage.upsert_job(job)

        result = sched.schedule_job(job)
        assert result.assigned_slot == 2
        assert result.assigned_username == "bob"
        assert result.state == JobState.ASSIGNED

    def test_fails_when_no_quota(self, tmp_path, monkeypatch):
        import compute_pool.storage.local as storage
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        statuses = [
            self._make_status(1, "alice", True, 0.0),
            self._make_status(2, "bob",   True, 0.0),
        ]

        import compute_pool.scheduler.simple as sched
        monkeypatch.setattr(sched, "get_all_statuses", lambda: statuses)

        job = Job(spec=JobSpec(name="no-quota-test", script="pass"))
        storage.upsert_job(job)

        result = sched.schedule_job(job)
        assert result.state == JobState.FAILED
        assert result.error is not None

    def test_fails_when_both_disconnected(self, tmp_path, monkeypatch):
        import compute_pool.storage.local as storage
        monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
        monkeypatch.setattr(storage, "JOBS_FILE", tmp_path / "jobs.json")

        statuses = [
            self._make_status(1, "alice", False, 30.0),
            self._make_status(2, "bob",   False, 30.0),
        ]

        import compute_pool.scheduler.simple as sched
        monkeypatch.setattr(sched, "get_all_statuses", lambda: statuses)

        job = Job(spec=JobSpec(name="disconnected-test", script="pass"))
        storage.upsert_job(job)

        result = sched.schedule_job(job)
        assert result.state == JobState.FAILED
