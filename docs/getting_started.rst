Getting Started
===============

Installation
------------

Clone the repo and install in editable mode:

.. code-block:: bash

   git clone git@github.com:PeteBro/eeg_access.git
   cd eeg_access
   pip install -e .

Usage
-----

Load trials for a subject
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from eeg_access import TrialHandler

   loader = TrialHandler()

   # Look up trials for subject 6, filtering to conditions 5 and 2951
   trials = loader.lookup_trials(subject=6, condition=[5, 2951])

   # Load all matching trials into memory
   # Returns {'data': ndarray, 'metadata': DataFrame}
   result = loader.get_data(trials)

Average by condition (ERP-style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from eeg_access import TrialHandler

   loader = TrialHandler()

   # Look up all shared trials across subjects
   trials = loader.lookup_trials(shared=True)

   # Load data and average across trials within each condition
   # result['data'] shape: (n_conditions, n_channels, n_samples)
   # result['metadata'] has one row per condition
   result = loader.get_data(trials, average_by='condition')

Batch iteration (memory-efficient)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from eeg_access import TrialHandler

   loader = TrialHandler()

   # Look up trials for subject 6
   trials = loader.lookup_trials(subject=6)

   # Iterate in batches of 32 trials — useful for large datasets
   for batch in loader.iter_data(trials, batch_size=32):
       data = batch['data']      # (batch_size, n_channels, n_samples)
       meta = batch['metadata']  # DataFrame aligned to data's first axis
