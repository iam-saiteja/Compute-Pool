"""Local tests for dynamic loss scaling (no torch needed -- pure Python state machine)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.kaggle.amp import DynamicLossScaler  # noqa: E402


def test_scale_grows_after_consecutive_finite_steps():
    s = DynamicLossScaler(init_scale=100.0, growth_factor=2.0, growth_interval=3)
    assert s.update(True) == 100.0
    assert s.update(True) == 100.0
    assert s.update(True) == 200.0  # 3rd consecutive finite step triggers growth


def test_scale_shrinks_immediately_on_overflow():
    s = DynamicLossScaler(init_scale=100.0, backoff_factor=0.5, growth_interval=3)
    s.update(True)
    s.update(True)
    assert s.update(False) == 50.0  # overflow before reaching growth_interval


def test_overflow_resets_the_growth_counter():
    s = DynamicLossScaler(init_scale=100.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=3)
    s.update(True)
    s.update(True)
    s.update(False)  # resets the streak, scale -> 50
    assert s.update(True) == 50.0
    assert s.update(True) == 50.0
    assert s.update(True) == 100.0  # needs 3 fresh consecutive finite steps


def test_scale_is_clamped_to_min_and_max():
    s = DynamicLossScaler(init_scale=2.0, backoff_factor=0.5, min_scale=1.0)
    for _ in range(5):
        s.update(False)
    assert s.scale == 1.0

    s = DynamicLossScaler(init_scale=2.0 ** 19, growth_factor=2.0, growth_interval=1, max_scale=2.0 ** 20)
    s.update(True)
    assert s.scale == 2.0 ** 20
    s.update(True)
    assert s.scale == 2.0 ** 20  # does not keep growing past max


def _run(name, fn):
    try:
        fn()
        print(f"ok   {name}")
        return True
    except Exception as e:
        print(f"FAIL {name}: {e!r}")
        return False


if __name__ == "__main__":
    results = [
        _run("scale_grows_after_consecutive_finite_steps", test_scale_grows_after_consecutive_finite_steps),
        _run("scale_shrinks_immediately_on_overflow", test_scale_shrinks_immediately_on_overflow),
        _run("overflow_resets_the_growth_counter", test_overflow_resets_the_growth_counter),
        _run("scale_is_clamped_to_min_and_max", test_scale_is_clamped_to_min_and_max),
    ]
    sys.exit(0 if all(results) else 1)
