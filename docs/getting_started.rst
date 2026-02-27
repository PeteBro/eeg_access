Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install eeg_access

Or install from source::

   git clone <repo-url>
   cd eeg_access
   pip install .

Quick Start
-----------

Load trials from a versioned zarr dataset::

   from eeg_access import TrialHandler

   loader = TrialHandler('/data/eeg_study', version='v2')

   # Load all trials for one subject
   result = loader.get_data(subject='sub-01')
   eeg = result['data']      # shape: (n_trials, n_channels, n_samples)
   meta = result['metadata'] # pd.DataFrame aligned to first axis

   # Filter trials before loading
   trials = loader.lookup_trials(subject='sub-01', shared=True)
   result = loader.get_data(trials)

   # Iterate in batches (memory-friendly)
   for batch in loader.iter_data(subject='sub-01', batch_size=64):
       process(batch['data'])

Building Metadata
-----------------

If you have your own preprocessed zarr epoch stores and need to generate the
metadata TSV that :class:`~eeg_access.TrialHandler` expects::

   from eeg_access import build_trial_metadata

   meta = build_trial_metadata('/data/eeg_study/v2')
   meta.to_csv('/data/eeg_study/v2/epoch_metadata.tsv', sep='\t')

Expected Dataset Layout
-----------------------

.. code-block:: text

   dataset_root/
   └── .../ (any depth)
       └── <version>/
           ├── *metadata.tsv
           └── sub-XX/
               └── chunk-XX/   ← zarr stores
