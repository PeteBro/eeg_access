"""Shared utilities for dataset discovery and metadata construction."""

import glob
import os
import pandas as pd
import zarr


def find_metadata(dataset_root: str, version: str) -> str:
    """Return the path to the metadata TSV for a given data version.

    Searches *dataset_root* recursively for a directory whose name matches
    *version*, then looks inside it for a file matching ``*metadata.tsv``.

    This is shared by all data loaders (epochs, raw, etc.) so that each one
    does not need its own path-discovery logic.

    Parameters
    ----------
    dataset_root : str
        Top-level folder of your dataset (e.g. ``'/data/eeg_study'``).
    version : str
        Name of the version directory to search for (e.g. ``'v2'``).  The
        directory can sit anywhere under *dataset_root*.

    Returns
    -------
    str
        Full path to the matching ``*metadata.tsv`` file.

    Raises
    ------
    FileNotFoundError
        If the version directory does not exist under *dataset_root*, or if
        no ``*metadata.tsv`` file is found inside it.
    ValueError
        If more than one version directory or more than one metadata file is
        found (ambiguous match).
    """
    version_dirs = [
        m for m in glob.glob(os.path.join(dataset_root, "**", version), recursive=True)
        if os.path.isdir(m)
    ]
    if not version_dirs:
        raise FileNotFoundError(
            f"No directory named '{version}' found under '{dataset_root}'"
        )
    if len(version_dirs) > 1:
        raise ValueError(
            f"Multiple directories named '{version}' found under '{dataset_root}':\n"
            + "\n".join(version_dirs)
        )
    tsvs = glob.glob(os.path.join(version_dirs[0], "*metadata.tsv"))
    if not tsvs:
        raise FileNotFoundError(
            f"No '*metadata.tsv' file found in '{version_dirs[0]}'"
        )
    if len(tsvs) > 1:
        raise ValueError(
            f"Multiple metadata files found in '{version_dirs[0]}':\n"
            + "\n".join(tsvs)
            + "\nSpecify the file explicitly to resolve the ambiguity."
        )
    return tsvs[0]


def build_trial_metadata(epochs_root: str) -> pd.DataFrame:
    """Build a trial metadata table from scratch by scanning zarr stores on disk.

    Use this when you have preprocessed your own raw EEG data into zarr epochs
    and need to generate the ``*metadata.tsv`` lookup file that
    :class:`~eeg_access.getdata.get_trials.EpochLookup` expects.

    The function walks *epochs_root* looking for ``sub-*/chunk-*`` zarr stores,
    reads the metadata embedded in each store, and assembles it into a single
    table with one row per trial.  Save the result as a TSV to use it with
    :class:`~eeg_access.getdata.get_trials.EpochLookup`.

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
