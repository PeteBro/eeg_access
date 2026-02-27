"""eeg_access — versioned EEG epoch loading utilities."""

from .getdata.get_trials import EpochLookup
from .getdata.utilities import build_trial_metadata, find_metadata

__all__ = ["EpochLookup", "build_trial_metadata", "find_metadata"]
