"""eeg_access — versioned EEG epoch loading utilities."""

from .getdata.get_trials import TrialHandler
from .getdata.utilities import build_trial_metadata, find_metadata

__all__ = ["TrialHandler", "build_trial_metadata", "find_metadata"]
