"""Compute Pool Kaggle worker runtime.

Shipped to each worker and imported by the strategy run loops. Keep this a flat
package of standard-library + torch modules so it can be copied to an ephemeral
Kaggle container and run without installation.
"""
from . import amp, checkpoint, strategy, wire

__all__ = ["amp", "checkpoint", "strategy", "wire"]
