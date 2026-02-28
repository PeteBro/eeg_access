"""Shared utilities for dataset discovery and metadata construction."""

import glob
import os
import pandas as pd
import zarr
from pathlib import Path
from dvc.repo import Repo

def resolve_dir(path, start=None):

    start = Path(start).resolve() if start else Path(os.getcwd())
    path = Path(path)
    upward = [p / path for p in [start, *start.parents] if (p / path).is_dir()]
    downward = [p for p in start.rglob(str(path)) if p.is_dir()] if not upward else []

    matches = list({p.resolve() for p in upward + downward})

    if len(matches) == 0:
        raise RuntimeError(f"Could not find directory '{path}'")
    if len(matches) > 1:
        matches_str = "\n  ".join(str(m) for m in matches)
        raise RuntimeError(f"Ambiguous — multiple '{path}' directories found:\n  {matches_str}")

    return matches[0]


def check_islocal(paths):

    return {p: os.path.exists(p) for p in paths}


def check_stale(paths):
    """Return paths that exist locally but are out of date per DVC."""
    if not paths:
        return []
    try:
        with Repo() as repo:
            status = repo.status(targets=[str(Path(p).resolve()) for p in paths])
        if not status:
            return []
        stale_out_paths = set()
        for changes in status.values():
            for change in changes:
                for out_path in change.get('changed outs', {}):
                    stale_out_paths.add(str(Path(out_path).resolve()))
        return [p for p in paths if str(Path(p).resolve()) in stale_out_paths]
    except Exception:
        return []


def fetch_remote(paths):

    missing = [p for p, local in check_islocal(paths).items() if not local]
    if not missing:
        return
    with Repo() as repo:
        try:
            repo.pull(targets=[f'{str(Path(p).resolve())}.dvc' for p in missing])
        except:
            print('DVC fetch failed, check requested filenames or credentials.')


def build_trial_metadata(epochs_root: str) -> pd.DataFrame:
    """Build a trial metadata table from scratch by scanning zarr stores on disk.

    Use this when you have preprocessed your own raw EEG data into zarr epochs
    and need to generate the ``*metadata.tsv`` lookup file that
    :class:`~eeg_access.getdata.get_trials.TrialHandler` expects.

    The function walks *epochs_root* looking for ``sub-*/chunk-*`` zarr stores,
    reads the metadata embedded in each store, and assembles it into a single
    table with one row per trial.  Save the result as a TSV to use it with
    :class:`~eeg_access.getdata.get_trials.TrialHandler`.

    Parameters
    ----------
    epochs_root : str
        Directory containing ``sub-XX/chunk-XX`` zarr stores (e.g.
        ``'/data/eeg_study/v2'``).

    Returns
    -------
    pd.DataFrame
        One row per trial.  Columns are the metadata fields stored inside each
        zarr file plus a ``path`` column with the full path to the zarr store
        that holds that trial's EEG data.

    Examples
    --------
    >>> meta = build_trial_metadata('/data/eeg_study/v2')
    >>> meta.to_csv('/data/eeg_study/v2/epoch_metadata.tsv', sep='\\t')
    """
    records = []
    for subject_dir in sorted(glob.glob(os.path.join(epochs_root, "sub-*"))):
        for chunk_dir in sorted(glob.glob(os.path.join(subject_dir, "chunk-*"))):
            z = zarr.open(chunk_dir, mode="r")
            df = pd.DataFrame(dict(z.attrs))
            df["path"] = chunk_dir
            records.append(df)
    return pd.concat(records, ignore_index=True)
