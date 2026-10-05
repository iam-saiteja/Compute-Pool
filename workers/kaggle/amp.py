"""Dynamic loss scaling for fp16 training without a kernel-level AMP backend.

The pipeline strategy trains in fp16 and needs a loss scale to keep small
gradients from underflowing to zero, but a single fixed scale is fragile: too
high and gradients overflow to inf/nan (wasting the step), too low and small
gradients vanish silently. This is the standard dynamic scaling algorithm
(as in torch.cuda.amp.GradScaler): grow the scale after a run of consecutive
finite steps, and shrink it immediately on the first overflow.

Each side of a split model (e.g. the pipeline's master and worker) should keep
its own DynamicLossScaler instance. They are allowed to diverge -- each is
solving a local numerical-stability problem for its own half of the network,
and a mismatch in which side skips a given step is already handled by the
pipeline's "master and worker disagree on finiteness" log line.
"""


class DynamicLossScaler:
    def __init__(self, init_scale=1024.0, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=50, min_scale=1.0, max_scale=2.0 ** 20):
        self.scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._good_steps = 0

    def update(self, finite):
        """Call once per step with whether this step's gradients were finite."""
        if finite:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self._good_steps = 0
        else:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self._good_steps = 0
        return self.scale

    def state_dict(self):
        """For checkpointing: without this, a resumed run restarts at
        init_scale and has to re-earn any growth from scratch."""
        return {"scale": self.scale, "good_steps": self._good_steps}

    def load_state_dict(self, state):
        self.scale = state["scale"]
        self._good_steps = state.get("good_steps", 0)
