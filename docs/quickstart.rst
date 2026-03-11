Quickstart
==========

Running a single simulation
----------------------------

.. code-block:: bash

   # GPU ABM: 50 beetles, 365 days, CI + ER enabled
   wings-abm --population 50 --days 365 --ci --er --output result.csv

   # Wright-Fisher model: 50 beetles, 12 generations, all 16 combos
   wings-wfm --run-all --nreps 200

Batch runs on SLURM
--------------------

.. code-block:: bash

   # ABM: 16 combos × 200 reps at 10% initial infection
   sbatch slurm/submit_abm.sh

   # ABM: same at 50% initial infection
   sbatch slurm/submit_abm_05.sh

   # Δp frequency sweep (CI, ER, CI+ER × 19 fractions × 50 reps)
   bash slurm/submit_delta_p.sh

Analysing results
-----------------

.. code-block:: bash

   # 1. Ingest raw CSVs into a combined dataset
   wings-ingest --input-dir /path/to/results --output data/combined.csv

   # 2. Generate publication figures
   wings-plot --model abm --input data/combined.csv

   # 3. Δp analysis
   wings-ingest-dp --input-dir /path/to/delta_p_results --output data/combined_dp.csv
   wings-plot-dp --input data/combined_dp.csv --dt 24 --mode compare
