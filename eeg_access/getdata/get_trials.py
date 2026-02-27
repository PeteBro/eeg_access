"""Trial metadata lookup and data loader."""

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from ..utilities import find_metadata


class TrialHandler:
    """Load and filter EEG trial data from a versioned zarr-backed dataset.

    Point this class at your dataset root folder and tell it which data version
    you want to work with.  It will find the matching metadata table
    automatically, giving you a simple interface to select and load trials.

    The dataset is expected to follow this layout::

        dataset_root/
        └── .../ (any depth)
            └── <version>/
                ├── *metadata.tsv
                └── sub-XX/
                    └── chunk-XX/   ← zarr stores

    Parameters
    ----------
    dataset_root : str
        Top-level folder of your dataset (e.g. ``'/data/eeg_study'``).
    version : str
        Name of the data version directory to load (e.g. ``'v2'`` or
        ``'preprocessed_v3'``).  The directory can sit anywhere under
        *dataset_root* — it will be located automatically.

    Examples
    --------
    >>> loader = TrialHandler('/data/eeg_study', version='v2')

    >>> # Load a specific subset by filtering inline
    >>> result = loader.get_data(subject='sub-01', shared=True)

    >>> # Or look up a trial table first, then load
    >>> trials = loader.lookup_trials(subject='sub-01', shared=True)
    >>> result = loader.get_data(trials)
    >>> result['data'].shape   # (n_trials, n_channels, n_samples)
    """

#
    def __init__(self, dataset_root: str, version: str):
        """Locate the metadata table for *version* and load it."""
#
        self.store_cache = {}
        lookup_path = find_metadata(dataset_root, version)
        self.metadata = pd.read_csv(lookup_path, sep="\t", index_col=0)

#
    def lookup_trials(self, cond='and', **filters) -> pd.DataFrame:
        """Return a subset of the metadata table matching the given criteria.

        Pass any metadata column as a keyword argument to filter trials.
        Multiple filters are combined with ``cond='and'`` (all criteria must
        match) or ``cond='or'`` (any criterion is enough).  The returned
        DataFrame can be passed directly to :meth:`get_data` or
        :meth:`iter_data`, or inspected before loading.

        Parameters
        ----------
        cond : {'and', 'or'}, optional
            How to combine multiple filters.  ``'and'`` (default) keeps only
            trials that satisfy **all** filters; ``'or'`` keeps trials that
            satisfy **at least one**.
        **filters
            Column name / value pairs.  The value can be a single item or a
            list.  For example ``nsd_id=[1, 2, 3]`` keeps only trials whose
            ``nsd_id`` column is 1, 2, or 3.

        Returns
        -------
        pd.DataFrame
            Filtered metadata table, sorted by the filter columns and with a
            fresh integer index.  Pass this directly to :meth:`get_data`.

        Examples
        --------
        >>> # All shared trials for subject 01
        >>> trials = loader.lookup_trials(subject='sub-01', shared=True)

        >>> # Trials from subject 01 OR subject 02
        >>> trials = loader.lookup_trials(cond='or', subject=['sub-01', 'sub-02'])
        """
        mask = pd.DataFrame(
            np.full(self.metadata.shape, True), columns=self.metadata.columns
        )
        for col, vals in filters.items():
            if not isinstance(vals, (list, tuple, np.ndarray)):
                vals = [vals]
            mask[col] &= self.metadata[col].isin(vals)
        mask = mask.to_numpy()
        if cond == 'and':
            mask = np.all(mask, axis=1)
        if cond == 'or':
            mask = np.any(mask, axis=1)
        return self.metadata[mask].sort_values(list(filters.keys())).reset_index(drop=True)

#
    def get_data(
        self,
        trials: pd.DataFrame = None,
        channels=None,
        tmin: float = None,
        tmax: float = None,
        average_by=None,
        verbose=True,
        cond='and',
        **filters,
    ) -> dict:
        """Load EEG data into memory, with optional inline trial filtering.

        Reads the EEG arrays from disk and returns them as a NumPy array
        together with the corresponding metadata.  Zarr stores are cached
        after the first open, so repeated calls for trials from the same
        store are fast.

        You can supply trials three ways:

        * Pass a pre-built ``trials`` DataFrame (e.g. from
          :meth:`lookup_trials`).
        * Pass filter keyword arguments directly — :meth:`lookup_trials` is
          called internally.
        * Pass neither — all trials in the metadata table are loaded.

        Parameters
        ----------
        trials : pd.DataFrame, optional
            Trial metadata table.  Must contain ``path`` and ``array_index``
            columns.  When omitted, ``**filters`` (if any) are used to build
            the table automatically via :meth:`lookup_trials`.
        channels : list, optional
            Channels to load.  Can be a list of integer indices or channel
            name strings.  When omitted, all channels are returned.
        tmin : float, optional
            Start of the time window in seconds.  Not yet active — the full
            timecourse is always returned for now.
        tmax : float, optional
            End of the time window in seconds.  Not yet active — the full
            timecourse is always returned for now.
        average_by : str or list of str, optional
            Metadata column(s) to average over.  For example
            ``average_by='subject'`` returns one averaged waveform per
            subject instead of one waveform per trial.
        verbose : bool, optional
            Show a progress bar while loading.  Default ``True``.
        cond : {'and', 'or'}, optional
            How to combine multiple ``**filters`` (passed to
            :meth:`lookup_trials`).  Ignored when ``trials`` is provided
            explicitly.  Default ``'and'``.
        **filters
            Column / value pairs forwarded to :meth:`lookup_trials` when
            ``trials`` is not provided.

        Returns
        -------
        dict
            A dictionary with two keys:

            ``'data'``
                NumPy array of shape ``(n_trials, n_channels, n_samples)``
                (or ``(n_groups, n_channels, n_samples)`` when
                ``average_by`` is set), dtype ``float32``.
            ``'metadata'``
                DataFrame with one row per trial (or per group when
                ``average_by`` is set), aligned to the first axis of
                ``'data'``.

        Examples
        --------
        >>> # Inline filtering — no separate lookup_trials call needed
        >>> result = loader.get_data(subject='sub-01', shared=True)
        >>> eeg = result['data']    # shape: (n_trials, n_channels, n_samples)

        >>> # Pass a pre-built trial table
        >>> trials = loader.lookup_trials(nsd_id=[1, 2, 3])
        >>> result = loader.get_data(trials)

        >>> # Average across trials, grouped by subject
        >>> result = loader.get_data(shared=True, average_by='subject')
        """
        if trials is None:
            trials = self.lookup_trials(cond=cond, **filters) if filters else self.metadata.copy()

        stores = trials['path'].unique()
        for path in stores:
            if path not in self.store_cache.keys():
                self.store_cache[path] = zarr.open(path, mode='r')
    #
        #channel_names = self.store_cache[stores[0]].attrs['channel_names']
        #sfreq = self.store_cache[stores[0]].attrs['sfreq']
        #timecourse = self.store_cache[stores[0]].attrs['timecourse']
        #nsamples = self.store_cache[stores[0]].shape[-1]
    #
        #if channels is not none:
        #    if isinstance(channels[0], str):
        #        chan_idcs = np.searchsorted(channels, channel_names)
        #else:
        #    chan_idcs = np.arange(len(channel_names))
    #
        #tmin = np.argmin(timecourse-tmin) if tmin is not none else 0
        #tmax = np.argmin(timecourse-tmin) if tmax is not none else nsamples-1
        #sample_idcs = np.arange(tmin, tmax)#, step)
    #
        store0 = self.store_cache[stores[0]]
        # Zarr always reads a full contiguous trial block; channel / sample
        # subsetting is applied in numpy on the in-memory chunk (cheap).
        chan_sel = channels if channels is not None else slice(None)
        sample_sel = slice(None)  # placeholder until tmin/tmax conversion is wired up
        n_channels = len(channels) if channels is not None else store0.shape[1]
        n_samples = store0.shape[2]
    #
        # Sort by store then array_index for sequential chunk access;
        # record original row position so output order matches input trials
        ordered = trials[['path', 'array_index']].copy()
        ordered['out_row'] = np.arange(len(trials))
        ordered = ordered.sort_values(['path', 'array_index'])
    #
        data_array = np.empty((len(trials), n_channels, n_samples), dtype='float32')
    #
        if verbose:
            prog = tqdm(range(len(trials)), desc='Loading Trials')
        for path, group in ordered.groupby('path', sort=False):
            store = self.store_cache[path]
            arr_idcs = group['array_index'].to_numpy()
            out_rows = group['out_row'].to_numpy()
            data_array[out_rows] = store.oindex[arr_idcs, chan_sel, sample_sel]
            if verbose:
                prog.update(len(out_rows))
    #
        meta = trials.reset_index(drop=True)
    #
        if average_by is not None:
            keys = [average_by] if isinstance(average_by, str) else list(average_by)
            groups = meta.groupby(keys, sort=False)
            data_array = np.stack([data_array[grp.index].mean(axis=0) for _, grp in groups])
            meta = groups.first().drop(columns=['path', 'array_index'], errors='ignore').reset_index()
    #
        return {"data": data_array, "metadata": meta}

#
    def iter_data(
        self,
        trials: pd.DataFrame = None,
        batch_size: int = 64,
        channels=None,
        tmin: float = None,
        tmax: float = None,
        average_by=None,
        sort_lookup=True,
        cond='and',
        **filters,
    ):
        """Iterate over trials in memory-friendly batches.

        Yields successive chunks of loaded EEG data instead of loading
        everything at once.  Useful when your full trial set is too large to
        fit in RAM, or when you want to feed a model batch-by-batch.

        Each yielded item has the same structure as the dict returned by
        :meth:`get_data`: a ``'data'`` array and a ``'metadata'`` DataFrame.

        When ``average_by`` is set, the iterator guarantees that all trials
        belonging to the same group are included in the same batch before
        averaging — groups are never split across batches.

        As with :meth:`get_data`, you can pass a pre-built ``trials`` table,
        supply ``**filters`` to build one inline, or omit both to iterate
        over all trials.

        Parameters
        ----------
        trials : pd.DataFrame, optional
            Trial metadata table.  When omitted, ``**filters`` (if any) are
            used to build it via :meth:`lookup_trials`, or all trials are
            used if no filters are given.
        batch_size : int, optional
            Maximum number of trials (or group rows) to load per batch.
            Default is 64.
        channels : list, optional
            Channels to load (integer indices or name strings).  All channels
            are loaded when omitted.
        tmin : float, optional
            Start of the time window in seconds (not yet active).
        tmax : float, optional
            End of the time window in seconds (not yet active).
        average_by : str or list of str, optional
            Metadata column(s) to average over within each batch.
        sort_lookup : bool, optional
            Sort trials by store path and array index before iterating for
            more efficient sequential disk reads.  Default ``True``.
        cond : {'and', 'or'}, optional
            How to combine multiple ``**filters``.  Ignored when ``trials``
            is provided explicitly.  Default ``'and'``.
        **filters
            Column / value pairs forwarded to :meth:`lookup_trials` when
            ``trials`` is not provided.

        Yields
        ------
        dict
            Same structure as :meth:`get_data`: ``{'data': np.ndarray,
            'metadata': pd.DataFrame}``.

        Examples
        --------
        >>> # Inline filtering
        >>> for batch in loader.iter_data(subject='sub-01', batch_size=32):
        ...     eeg = batch['data']   # shape: (<=32, n_channels, n_samples)
        ...     meta = batch['metadata']
        ...     process(eeg, meta)

        >>> # Iterate with per-stimulus averaging
        >>> trials = loader.lookup_trials(shared=True)
        >>> for batch in loader.iter_data(trials, batch_size=64, average_by='subject'):
        ...     process(batch['data'], batch['metadata'])
        """
        if trials is None:
            trials = self.lookup_trials(cond=cond, **filters) if filters else self.metadata.copy()

        keys = ([average_by] if isinstance(average_by, str) else list(average_by)) if average_by else None
#
        if sort_lookup:
            sort_cols = (keys + ['path', 'array_index']) if keys else ['path', 'array_index']
            trials = trials.sort_values(sort_cols)
#
        if keys:
            # accumulate complete groups into batches, never splitting a group
            batch, count = [], 0
            for _, grp in trials.groupby(keys, sort=False):
                if count + len(grp) > batch_size and batch:
                    yield self.get_data(pd.concat(batch), channels=channels,
                                        tmin=tmin, tmax=tmax, average_by=keys,
                                        verbose=False)
                    batch, count = [], 0
                batch.append(grp)
                count += len(grp)
            if batch:
                yield self.get_data(pd.concat(batch), channels=channels,
                                    tmin=tmin, tmax=tmax, average_by=keys,
                                    verbose=False)
        else:
            for start in range(0, len(trials), batch_size):
                yield self.get_data(
                    trials.iloc[start:start + batch_size],
                    channels=channels, tmin=tmin, tmax=tmax, verbose=False
                )
